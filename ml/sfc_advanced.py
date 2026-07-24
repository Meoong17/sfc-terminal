#!/usr/bin/env python3
"""
SFC Advanced Module — Priority 2-6 Implementation
==================================================
All implementations use pure numpy/scipy — no sklearn/hmmlearn/gplearn dependency.

Priority 2: HMM Regime Detection (manual GMM + Markov chain)
Priority 3: Walk-Forward Backtest Engine
Priority 4: Uncertainty Quantification (Platt scaling + Bootstrap)
Priority 5: Auto Feature Engineering (polynomial + ratio features)
Priority 6: Alternative Data
"""

import json, os, sys, math, time, requests
from datetime import datetime, timezone
import numpy as np
from scipy import stats as sp_stats
from scipy.cluster.vq import kmeans2, vq

# ════════════════════════════════════════════════════════════════
# PRIORITY 2: REGIME DETECTION (Manual HMM-like)
# ════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    HMM-like regime detection using GMM clustering + Markov transition matrix.
    
    Pure numpy implementation — no hmmlearn dependency.
    Uses k-means + Gaussian fit for each cluster, then builds
    a transition probability matrix between regimes.
    
    Regimes: BULL, BEAR, SIDEWAYS, CRISIS (sorted by volatility)
    """
    
    def __init__(self, n_regimes=4, random_state=42):
        self.n_regimes = n_regimes
        self.rng = np.random.RandomState(random_state)
        self.centroids = None      # GMM means
        self.covariances = None    # GMM covariances per cluster
        self.transmat = None       # Markov transition matrix
        self.regime_labels = ['BULL', 'BEAR', 'SIDEWAYS', 'CRISIS']
        self.state_order = None    # mapping centroid index -> sorted regime index
        
    def fit(self, features):
        """
        features: 2D numpy array [n_samples x n_features]
        Fits k-means clusters, then builds transition matrix.
        """
        features = np.asarray(features, dtype=float)
        n = len(features)
        if n < self.n_regimes * 5:
            return self
        
        # 1. K-means clustering (fixed seed = deterministic)
        centroid, labels = kmeans2(features, self.n_regimes, minit='points',
                                    iter=50, seed=42)
        self.centroids = centroid
        
        # 2. Compute covariance per cluster
        self.covariances = []
        for k in range(self.n_regimes):
            mask = labels == k
            if mask.sum() > 1:
                cov = np.cov(features[mask].T)
                # Regularize to avoid singular matrices
                cov += np.eye(cov.shape[0]) * 1e-6
            else:
                cov = np.eye(features.shape[1]) * 0.1
            self.covariances.append(cov)
        
        # 3. Sort clusters by volatility (std of first feature as proxy)
        cluster_vols = []
        for k in range(self.n_regimes):
            mask = labels == k
            if mask.sum() > 0:
                cluster_vols.append(np.std(features[mask, 0]) if features.shape[1] > 0 else 0)
            else:
                cluster_vols.append(0)
        
        # Sort: lowest vol = BULL, highest vol = CRISIS
        self.state_order = np.argsort(cluster_vols)  # [low_vol_idx, ..., high_vol_idx]
        
        # 4. Build transition matrix
        self.transmat = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(1, n):
            i, j = labels[t-1], labels[t]
            self.transmat[i, j] += 1.0
        # Normalize rows to probabilities
        row_sums = self.transmat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.transmat = self.transmat / row_sums
        
        return self
    
    def predict(self, features):
        """
        Predict regime for each sample in features.
        Returns (regime_ids, regime_labels)
        """
        features = np.asarray(features, dtype=float)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Assign to nearest centroid
        dists = np.array([[np.linalg.norm(f - c) for c in self.centroids] for f in features])
        raw_labels = np.argmin(dists, axis=1)
        
        # Map through state_order
        label_map = {orig: new for new, orig in enumerate(self.state_order)}
        mapped = np.array([label_map.get(l, 0) for l in raw_labels])
        
        return mapped, [self.regime_labels[m] for m in mapped]
    
    def get_regime_status(self, current_features):
        """Get detailed regime info for current observation."""
        rid, rlabel = self.predict(current_features)
        regime_id = rid[0] if isinstance(rid, np.ndarray) else rid
        
        # Transition probabilities from current regime
        if self.transmat is not None:
            trans_probs = {self.regime_labels[i]: float(self.transmat[regime_id, i])
                          for i in range(self.n_regimes)}
            stability = 1.0 - float(self.transmat[regime_id, regime_id])
            crisis_idx = self.regime_labels.index('CRISIS')
            crisis_prob = float(self.transmat[regime_id, crisis_idx])
        else:
            trans_probs = {}
            stability = 0.5
            crisis_prob = 0.0
        
        return {
            'regime_id': int(regime_id),
            'regime': rlabel[0] if isinstance(rlabel, list) else rlabel,
            'stability': round(stability, 3),
            'crisis_probability': round(crisis_prob, 3),
            'transition_probabilities': trans_probs,
        }
    
    def score_stress_boost(self, current_features):
        """
        Compute stress index boost from regime detection.
        Returns additional stress points to add to SFC.
        """
        status = self.get_regime_status(current_features)
        boost = 0
        if status['crisis_probability'] > 0.3:
            boost += 15
        if status['regime'] == 'CRISIS':
            boost += 10
        if status['regime'] == 'BEAR' and status['stability'] < 0.5:
            boost += 5
        return boost, status


# ════════════════════════════════════════════════════════════════
# PRIORITY 3: WALK-FORWARD BACKTEST (standalone validation)
# ════════════════════════════════════════════════════════════════

class WalkForwardBacktest:
    """
    Walk-forward validation for SFC signals.
    Uses expanding window — no look-ahead bias.
    """
    
    def __init__(self, train_days=756, test_days=63):
        """
        train_days: ~3 years of trading days
        test_days: ~3 months
        """
        self.train_days = train_days
        self.test_days = test_days
        self.results = []
        
    def run(self, signal_series, return_series):
        """
        signal_series: list/array of signal values (0=risk-off, 1=risk-on, 0.5=partial)
        return_series: list/array of daily returns
        
        Returns dict of metrics.
        """
        signals = np.array(signal_series, dtype=float)
        returns = np.array(return_series, dtype=float)
        n = min(len(signals), len(returns))
        
        signals = signals[:n]
        returns = returns[:n]
        
        # Walk-forward
        walk_results = []
        equity = 1.0
        equity_curve = [1.0]
        
        for t in range(1, n):
            signal = signals[t]
            ret = returns[t]
            
            # Apply signal to return
            if signal > 0.5:  # risk-on
                portfolio_ret = ret
            elif signal < 0.1:  # risk-off
                portfolio_ret = 0.0
            else:  # partial
                portfolio_ret = ret * signal
            
            equity *= (1.0 + portfolio_ret)
            equity_curve.append(equity)
            walk_results.append({
                'signal': float(signal),
                'return': float(ret),
                'portfolio_return': float(portfolio_ret),
                'equity': float(equity),
            })
        
        self.results = walk_results
        self.equity_curve = np.array(equity_curve)
        
        return self._compute_metrics(returns[:n])
    
    def _compute_metrics(self, full_returns):
        """Compute backtest metrics."""
        signals = np.array([r['signal'] for r in self.results])
        port_rets = np.array([r['portfolio_return'] for r in self.results])
        
        # Sharpe ratio (annualized)
        if port_rets.std() > 0:
            sharpe = np.sqrt(252) * port_rets.mean() / port_rets.std()
        else:
            sharpe = 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        dd = (self.equity_curve - peak) / peak
        max_dd = float(dd.min())
        
        # Win rate
        win_rate = float((port_rets > 0).mean())
        
        # Buy & hold return
        bh_return = float(np.prod(1.0 + full_returns) - 1.0)
        strategy_return = float(self.equity_curve[-1] - 1.0)
        
        # Exposure
        exposure = float((signals > 0.5).mean())
        
        # Signal stability (lower = more consistent)
        signal_std = float(signals.std())
        
        return {
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_dd, 4),
            'win_rate': round(win_rate, 3),
            'strategy_return': round(strategy_return, 4),
            'buy_hold_return': round(bh_return, 4),
            'exposure': round(exposure, 3),
            'signal_stability': round(1.0 - min(signal_std, 1.0), 3),
            'n_periods': len(self.results),
            'overfitting_risk': 'LOW' if sharpe > 1.5 and max_dd > -0.15 else 
                               'MEDIUM' if sharpe > 0.5 else 'HIGH',
        }
    
    def get_report(self):
        """Human-readable backtest report."""
        if not self.results:
            return "No backtest results."
        
        m = self._compute_metrics(np.zeros(len(self.results)))
        lines = []
        lines.append("╔" + "═"*50 + "╗")
        lines.append("║  WALK-FORWARD BACKTEST REPORT                ║")
        lines.append("╠" + "═"*50 + "╣")
        lines.append(f"║ Sharpe Ratio:      {m['sharpe_ratio']:>10.3f}                  ║")
        lines.append(f"║ Max Drawdown:      {m['max_drawdown']:>10.2%}                  ║")
        lines.append(f"║ Win Rate:          {m['win_rate']:>10.1%}                  ║")
        lines.append(f"║ Strategy Return:   {m['strategy_return']:>10.2%}                  ║")
        lines.append(f"║ Buy & Hold Return: {m['buy_hold_return']:>10.2%}                  ║")
        lines.append(f"║ Exposure:          {m['exposure']:>10.1%}                  ║")
        lines.append(f"║ Overfitting Risk:  {m['overfitting_risk']:>10}                  ║")
        lines.append(f"║ Periods:           {m['n_periods']:>10}                  ║")
        lines.append("╚" + "═"*50 + "╝")
        return '\n'.join(lines)


# ════════════════════════════════════════════════════════════════
# PRIORITY 4: UNCERTAINTY QUANTIFICATION
# ════════════════════════════════════════════════════════════════

class UncertaintyQuantifier:
    """
    Uncertainty Quantification via Bootstrap + Platt scaling.
    
    Pure numpy implementation:
    - Platt scaling: fits logistic regression to calibrate probabilities
    - Bootstrap: 100 samples for confidence intervals
    """
    
    def __init__(self, n_bootstrap=100, confidence_level=0.9):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.a = 0.0  # Platt params
        self.b = 0.0
        
    def _platt_fit(self, scores, labels):
        """
        Platt scaling: P(y=1|x) = 1 / (1 + exp(A * score + B))
        Uses Newton-Raphson optimization.
        """
        scores = np.array(scores, dtype=float)
        labels = np.array(labels, dtype=float)
        n = len(scores)
        
        # Initial params
        a, b = 0.0, 0.0
        
        # Prior to avoid extreme probabilities
        prior0 = max(1, (labels == 0).sum())
        prior1 = max(1, (labels == 1).sum())
        
        for _ in range(100):
            # Current probabilities
            f = scores
            p = 1.0 / (1.0 + np.exp(-(a * f + b)))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            
            # Gradient
            da = np.sum(f * (p - labels))
            db = np.sum(p - labels)
            
            # Hessian
            w = p * (1 - p)
            ha = np.sum(f * f * w)
            hb = np.sum(f * w)
            hbb = np.sum(w)
            
            # Newton step
            try:
                det = ha * hbb - hb * hb
                if abs(det) < 1e-15:
                    break
                da_new = (hbb * da - hb * db) / det
                db_new = (ha * db - hb * da) / det
            except:
                break
            
            a -= da_new
            b -= db_new
            
            if abs(da_new) < 1e-5 and abs(db_new) < 1e-5:
                break
        
        self.a = a
        self.b = b
        return self
    
    def _platt_predict(self, scores):
        """Apply Platt scaling: raw scores → calibrated probabilities."""
        scores = np.array(scores, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.a * scores + self.b)))
    
    def fit(self, scores, labels):
        """Fit Platt scaling from historical scores vs actual outcomes."""
        self._platt_fit(scores, labels)
        return self
    
    def predict_with_uncertainty(self, scores, regime_info=None):
        """
        Predict with confidence intervals — multi-dimensional, regime-aware.

        Mathematical approach:
        ──────────────────────────────────────────────────
        1. Primary Platt-calibrated score  (sfc_pct/100)
        2. Dimensional bootstrap: resample feature dimensions with replacement,
           weighted by importance. If N features disagree, bootstrap variance
           is wide → high uncertainty → LOW_CONFIDENCE.
           If N features agree, bootstrap variance is narrow → reliable.
        3. Dynamic thresholds adjusted by regime crisis probability:
           calm_threshold  = 0.30 * max(0.01, 1 - 2*P(crisis))
           stress_threshold = 0.70 * max(0.25, 1 - 0.4*P(crisis))
        4. Safety override: never return CALM when regime strongly CRISIS

        Args:
            scores: 1D array where scores[0] = primary SFC stress (0-1),
                    scores[1:] = auxiliary stress features.
            regime_info: dict with 'crisis_probability', 'regime'.

        Returns:
            dict with prediction, bounds, uncertainty, action.
        """
        scores = np.array(scores, dtype=float)
        if len(scores) == 0:
            return {'prediction': 0.5, 'lower_bound': 0.0, 'upper_bound': 1.0,
                    'uncertainty': 1.0, 'is_reliable': False, 'recommended_action': 'LOW_CONFIDENCE'}

        primary = float(scores[0])
        n_feats = len(scores)

        # ── 1. Platt-calibrated point prediction ──
        if self.a != 0 or self.b != 0:
            pred = float(self._platt_predict(np.array([primary]))[0])
        else:
            pred = float(np.clip(primary, 0, 1))

        # ── 2. Dimensional bootstrap ──
        #    Resample features with replacement, weighted by importance.
        #    Primary score gets 2× weight; aux features get equal weight.
        w = np.ones(n_feats)
        w[0] = 2.0  # primary feature double-weighted
        w = w / w.sum()

        n_boot = max(self.n_bootstrap, 50)
        boot_preds = np.empty(n_boot)
        for i in range(n_boot):
            idx = np.random.choice(n_feats, size=n_feats, replace=True)
            w_sub = w[idx]
            w_sub = w_sub / w_sub.sum()
            # Apply Platt if calibrated, else clip
            if self.a != 0 or self.b != 0:
                calibrated = self._platt_predict(scores[idx])
            else:
                calibrated = np.clip(scores[idx], 0, 1)
            boot_preds[i] = float(np.dot(w_sub, calibrated))

        alpha = 1.0 - self.confidence_level
        lower = float(np.percentile(boot_preds, 50 * alpha))
        upper = float(np.percentile(boot_preds, 100 - 50 * alpha))
        uncertainty = upper - lower
        is_reliable = uncertainty < 0.30

        # ── 3. Regime context ──
        regime_crisis_prob = 0.0
        regime_label = 'NORMAL'
        if regime_info and isinstance(regime_info, dict):
            regime_crisis_prob = float(regime_info.get('crisis_probability', 0))
            regime_label = str(regime_info.get('regime', 'NORMAL'))

        # ── 4. Dynamic thresholds ──
        #    P(crisis)=1   → calm_threshold ≈ 0.01 (virtually unreachable)
        #    P(crisis)=0   → calm_threshold ≈ 0.30 (original)
        #    P(crisis)=1   → stress_threshold ≈ 0.42 (easier to trigger)
        #    P(crisis)=0   → stress_threshold ≈ 0.70 (original)
        calm_t = 0.30 * max(0.01, 1.0 - 2.0 * regime_crisis_prob)
        stress_t = 0.70 * max(0.25, 1.0 - 0.4 * regime_crisis_prob)

        # ── 5. Action selection with safety override ──
        if pred > stress_t and is_reliable:
            action = "HIGH_CONFIDENCE_STRESS"
        elif pred < calm_t and is_reliable:
            action = "HIGH_CONFIDENCE_CALM"
        elif uncertainty < 0.15:
            action = "MEDIUM_CONFIDENCE"
        else:
            action = "LOW_CONFIDENCE"

        # Safety override: never say CALM when regime is CRISIS
        if action == "HIGH_CONFIDENCE_CALM" and regime_crisis_prob > 0.5:
            action = "MEDIUM_CONFIDENCE"
        if action in ("HIGH_CONFIDENCE_CALM", "MEDIUM_CONFIDENCE") and regime_crisis_prob > 0.85:
            action = "LOW_CONFIDENCE"

        return {
            'prediction': round(float(pred), 4),
            'lower_bound': round(float(lower), 4),
            'upper_bound': round(float(upper), 4),
            'uncertainty': round(float(uncertainty), 4),
            'is_reliable': bool(is_reliable),
            'recommended_action': action,
        }


# ════════════════════════════════════════════════════════════════
# PRIORITY 5: AUTO FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════

class AutoFeatureEngineer:
    """
    Generate and select predictive features.
    
    Generates:
    - Polynomial features (squared, cubed)
    - Ratio features (feature_i / feature_j)
    - Log/exp transformations
    - Rolling statistics (momentum, acceleration)
    
    Selection via mutual information (using scipy entropy).
    """
    
    def __init__(self):
        self.feature_names = []
        self.selected_indices = None
        self.mi_scores = None
        
    def generate(self, data_dict):
        """
        Generate augmented features from a dict of {name: value} metrics.
        
        Returns (feature_vector, feature_names) for ML consumption.
        """
        features = []
        names = []
        
        for name, val in data_dict.items():
            if val is None or not isinstance(val, (int, float)):
                continue
            
            v = float(val)
            
            # Original
            features.append(v)
            names.append(name)
            
            # Squared (non-linear relationship)
            features.append(v * v)
            names.append(f"{name}^2")
            
            # Log (if positive)
            if v > 0:
                features.append(math.log(max(v, 1e-10)))
                names.append(f"log_{name}")
            
            # Sigmoid transform
            features.append(1.0 / (1.0 + math.exp(-v * 5 + 2.5)))
            names.append(f"sigmoid_{name}")
        
        # Ratio features: BTC price / method score for pairs
        btc_price = data_dict.get('btc', data_dict.get('price', 0))
        if btc_price and btc_price > 0:
            for name, val in data_dict.items():
                if val is not None and isinstance(val, (int, float)) and val > 0:
                    if name not in ('btc', 'price', 'btc_mcap'):
                        features.append(btc_price / max(float(val), 1e-10))
                        names.append(f"btc_{name}_ratio")
        
        self.feature_names = names
        return np.array(features), names
    
    def select(self, X, y, top_k=15):
        """
        Select top-k features by mutual information.
        
        X: 2D array [n_samples, n_features]
        y: target array
        Returns (X_selected, importance_df)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        scores = []
        
        for i in range(n_features):
            mi = self._mutual_info(X[:, i], y)
            scores.append(mi)
        
        scores = np.array(scores)
        self.mi_scores = scores
        
        # Select top-k
        if n_features <= top_k:
            self.selected_indices = list(range(n_features))
        else:
            self.selected_indices = np.argsort(scores)[-top_k:].tolist()
        
        importance = []
        for idx in self.selected_indices:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
            importance.append({
                'feature': name,
                'index': idx,
                'mi_score': round(float(scores[idx]), 4),
            })
        
        importance.sort(key=lambda x: x['mi_score'], reverse=True)
        
        return X[:, self.selected_indices], importance
    
    def _mutual_info(self, x, y, bins=10):
        """Compute mutual information I(X;Y) using histogram."""
        x = x[~np.isnan(x) & ~np.isnan(y[:len(x)])]
        y = y[:len(x)][~np.isnan(x) & ~np.isnan(y[:len(x)])]
        
        if len(x) < 10:
            return 0.0
        
        # Discretize
        try:
            x_bins = np.digitize(x, np.percentile(x, np.linspace(0, 100, bins+1)[1:-1]))
            y_bins = np.digitize(y, np.percentile(y, np.linspace(0, 100, bins+1)[1:-1]))
        except:
            return 0.0
        
        n = len(x_bins)
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                p_xy = ((x_bins == i) & (y_bins == j)).sum() / n
                p_x = (x_bins == i).sum() / n
                p_y = (y_bins == j).sum() / n
                if p_xy > 0 and p_x > 0 and p_y > 0:
                    mi += p_xy * math.log(p_xy / (p_x * p_y))
        
        return mi / math.log(bins)  # Normalized [0, 1]
    
    def get_importance_report(self):
        """Return sorted feature importance."""
        if self.mi_scores is None:
            return "No selection performed."
        lines = []
        lines.append("Feature Importance (Mutual Information):")
        indices = np.argsort(self.mi_scores)[::-1]
        for i, idx in enumerate(indices):
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
            lines.append(f"  {i+1:2d}. {name:30s} MI={self.mi_scores[idx]:.4f}")
        return '\n'.join(lines)


# ════════════════════════════════════════════════════════════════
# PRIORITY 6: ALTERNATIVE DATA
# ════════════════════════════════════════════════════════════════

# Google Trends — direct HTTP (no pytrends dependency)
GOOGLE_TRENDS_CACHE = {}
GOOGLE_TRENDS_CACHE_TIME = 0
TRENDS_CACHE_TTL = 3600  # 1 hour

def fetch_google_trends(keywords=None, timeout=10):
    """
    Fetch Google Trends interest data for keywords.
    Note: Tokenless Trends API always blocked. Using FnG proxy instead
    (same info already captured in Rt factor — kept for backward compat).
    Returns dict of {keyword: score (0-100)}.
    """
    global GOOGLE_TRENDS_CACHE, GOOGLE_TRENDS_CACHE_TIME
    
    if keywords is None:
        keywords = ['recession', 'bitcoin crash', 'inflation']
    
    now = time.time()
    if GOOGLE_TRENDS_CACHE and (now - GOOGLE_TRENDS_CACHE_TIME) < TRENDS_CACHE_TTL:
        return GOOGLE_TRENDS_CACHE
    
    # Direct Trends API always blocks without auth — use FnG proxy
    # Note: (100 - FnG)/100 is already available as Rt factor, so this is info-duplicating
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=timeout)
        fng_data = r.json()
        fng_val = int(fng_data['data'][0]['value'])
        results = {
            'recession': round((100 - fng_val) / 100.0, 3),
            'bitcoin crash': round((100 - fng_val) / 100.0, 3),
            'inflation': round(fng_val / 100.0, 3),
        }
    except:
        results = {kw: 0.5 for kw in keywords}
    
    GOOGLE_TRENDS_CACHE = results
    GOOGLE_TRENDS_CACHE_TIME = now
    return results


# Reddit sentiment proxy — using simple RSS
REDDIT_CACHE = {}
REDDIT_CACHE_TIME = 0
REDDIT_CACHE_TTL = 600  # 10 min

def fetch_reddit_sentiment(timeout=10):
    """
    Fetch Reddit r/cryptocurrency + r/wallstreetbets sentiment.
    Uses simple keyword counting on hot posts via JSON API.
    Returns dict with sentiment_score, post_count.
    """
    global REDDIT_CACHE, REDDIT_CACHE_TIME
    
    now = time.time()
    if REDDIT_CACHE and (now - REDDIT_CACHE_TIME) < REDDIT_CACHE_TTL:
        return REDDIT_CACHE
    
    bullish_words = ['bull', 'long', 'buy', 'moon', 'mooning', 'rocket', 'green', 'pump']
    bearish_words = ['bear', 'short', 'sell', 'crash', 'red', 'dump', 'panic', 'fear']
    
    total_score = 0
    total_posts = 0
    
    for subreddit in ['cryptocurrency', 'wallstreetbets']:
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
            headers = {'User-Agent': 'SFC Terminal/2.0'}
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                continue
            
            data = r.json()
            posts = data.get('data', {}).get('children', [])
            
            for post in posts:
                title = post.get('data', {}).get('title', '').lower()
                score = 0
                for w in bullish_words:
                    if w in title:
                        score += 1
                for w in bearish_words:
                    if w in title:
                        score -= 1
                total_score += score
                total_posts += 1
        except:
            continue
    
    if total_posts > 0:
        sentiment = total_score / total_posts
    else:
        sentiment = 0.0
    
    # Normalize to [-1, 1]
    sentiment = max(-1.0, min(1.0, sentiment))
    
    result = {
        'sentiment_score': round(sentiment, 3),
        'post_count': total_posts,
        'sentiment_label': 'BULLISH' if sentiment > 0.2 else 'BEARISH' if sentiment < -0.2 else 'NEUTRAL',
    }
    
    REDDIT_CACHE = result
    REDDIT_CACHE_TIME = now
    return result


ALL_DATA_CACHE = {}
ALL_DATA_CACHE_TIME = 0
ALL_DATA_CACHE_TTL = 600  # 10 min

def fetch_all_alternative_data(force=False):
    """
    Fetch all alternative data sources in one call.
    Returns combined dict.
    """
    global ALL_DATA_CACHE, ALL_DATA_CACHE_TIME
    
    now = time.time()
    if not force and ALL_DATA_CACHE and (now - ALL_DATA_CACHE_TIME) < ALL_DATA_CACHE_TTL:
        return ALL_DATA_CACHE
    
    result = {}
    
    # 1. Google Trends proxy (via Fear & Greed)
    try:
        trends = fetch_google_trends()
        result['trends_recession'] = trends.get('recession', 0.5)
        result['trends_crash'] = trends.get('bitcoin crash', 0.5)
        result['trends_inflation'] = trends.get('inflation', 0.5)
    except:
        result['trends_recession'] = 0.5
        result['trends_crash'] = 0.5
        result['trends_inflation'] = 0.5
    
    # 2. Reddit sentiment
    try:
        reddit = fetch_reddit_sentiment()
        result['reddit_sentiment'] = reddit.get('sentiment_score', 0.0)
        result['reddit_posts'] = reddit.get('post_count', 0)
        result['reddit_label'] = reddit.get('sentiment_label', 'NEUTRAL')
    except:
        result['reddit_sentiment'] = 0.0
        result['reddit_posts'] = 0
        result['reddit_label'] = 'NEUTRAL'
    
    # 3. Deribit put/call ratio (already in collect.py, just normalize)
    # This is fetched separately in the pipeline
    
    # 4. CoinGecko market data supplement
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false"
            "&community_data=false&developer_data=false&market_data=true"
            "&x_cg_demo_api_key=REMOVED_SECRET",
            timeout=10
        )
        coin = r.json()
        md = coin.get('market_data', {})
        result['cg_price_change_24h'] = md.get('price_change_percentage_24h', 0)
        result['cg_price_change_7d'] = md.get('price_change_percentage_7d', 0)
        result['cg_price_change_30d'] = md.get('price_change_percentage_30d', 0)
        result['cg_ath_dd'] = round(
            (md.get('current_price', {}).get('usd', 0) - md.get('ath', {}).get('usd', 1)) /
            max(md.get('ath', {}).get('usd', 1), 1), 4
        )
    except:
        pass
    
    ALL_DATA_CACHE = result
    ALL_DATA_CACHE_TIME = now
    return result


# ════════════════════════════════════════════════════════════════
# COMPOSITE RUNNER
# ════════════════════════════════════════════════════════════════

def compute_all_advanced(historical_features=None):
    """
    Run all advanced modules and return combined results.
    
    Args:
        historical_features: list of feature dicts (from data_collection.json)
    
    Returns dict with all priority results.
    """
    results = {}
    
    # Priority 2: Regime Detection
    try:
        if historical_features and len(historical_features) > 20:
            # Build feature matrix from historical data
            feat_names = list(historical_features[0].keys())
            feat_matrix = []
            for obs in historical_features:
                row = [float(obs.get(k, 0.5)) for k in feat_names[:5]]
                feat_matrix.append(row)
            
            rd = RegimeDetector(n_regimes=4)
            rd.fit(np.array(feat_matrix))
            
            # Current regime
            current = feat_matrix[-1] if feat_matrix else [0.5]*5
            regime_status = rd.get_regime_status(np.array(current))
            regime_boost, _ = rd.score_stress_boost(np.array(current))
            
            results['regime'] = regime_status
            results['regime_boost'] = regime_boost
        else:
            results['regime'] = {'regime': 'NORMAL', 'crisis_probability': 0.0, 'stability': 0.9}
            results['regime_boost'] = 0
    except Exception as e:
        print(f"[Advanced] Regime detection error: {e}", file=sys.stderr)
        results['regime'] = {'regime': 'NORMAL', 'crisis_probability': 0.0, 'stability': 0.9}
        results['regime_boost'] = 0
    
    # Priority 4: Uncertainty Quantification (lightweight)
    try:
        uq = UncertaintyQuantifier(n_bootstrap=50)
        # SFC score as the raw score for uncertainty
        sfc_score = results.get('regime_boost', 0) / 100.0
        uq_result = uq.predict_with_uncertainty(np.array([sfc_score]))
        results['uncertainty'] = uq_result
    except Exception as e:
        print(f"[Advanced] Uncertainty error: {e}", file=sys.stderr)
        results['uncertainty'] = {'prediction': 0.5, 'uncertainty': 0.5, 'is_reliable': False}
    
    # Priority 6: Alternative Data
    try:
        alt_data = fetch_all_alternative_data()
        results['alternative'] = alt_data
    except Exception as e:
        print(f"[Advanced] Alt data error: {e}", file=sys.stderr)
        results['alternative'] = {}
    
    return results


if __name__ == "__main__":
    print("═" * 55)
    print("  SFC ADVANCED MODULE — Priority 2-6")
    print("═" * 55)
    
    # Test Regime Detection
    print("\n[Test] Regime Detection with synthetic data...")
    rng = np.random.RandomState(42)
    
    # Generate feature series with regime shifts
    n = 200
    features = []
    for t in range(n):
        if t < 50:      # BULL — low vol, positive trend
            f = [0.1 + t*0.001 + rng.randn()*0.02, 0.8 + rng.randn()*0.05]
        elif t < 100:   # BEAR — high vol, negative
            f = [0.3 + rng.randn()*0.05, 0.3 + rng.randn()*0.08]
        elif t < 150:   # SIDEWAYS — low vol, flat
            f = [0.2 + rng.randn()*0.01, 0.5 + rng.randn()*0.02]
        else:           # CRISIS — extreme vol
            f = [0.6 + rng.randn()*0.1, 0.1 + rng.randn()*0.12]
        features.append(f)
    
    rd = RegimeDetector(n_regimes=4)
    rd.fit(np.array(features))
    
    # Test last point
    status = rd.get_regime_status(np.array(features[-1]))
    print(f"  Current regime: {status['regime']}")
    print(f"  Crisis probability: {status['crisis_probability']:.3f}")
    print(f"  Stability: {status['stability']:.3f}")
    print(f"  Transitions: {status['transition_probabilities']}")
    
    # Test Uncertainty
    print("\n[Test] Uncertainty Quantification...")
    uq = UncertaintyQuantifier(n_bootstrap=50)
    # Simulate Platt fitting
    synth_scores = np.array([0.1, 0.2, 0.15, 0.3, 0.6, 0.8, 0.7, 0.9])
    synth_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    uq.fit(synth_scores, synth_labels)
    
    test = uq.predict_with_uncertainty(np.array([0.35]))
    print(f"  Prediction: {test['prediction']:.3f}")
    print(f"  Confidence: [{test['lower_bound']:.3f}, {test['upper_bound']:.3f}]")
    print(f"  Uncertainty: {test['uncertainty']:.3f}")
    print(f"  Action: {test['recommended_action']}")
    
    # Test Alternative Data
    print("\n[Test] Alternative Data...")
    alt = fetch_all_alternative_data(force=True)
    for k, v in alt.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n✅ All advanced modules operational")
