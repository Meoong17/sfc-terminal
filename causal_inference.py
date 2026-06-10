#!/usr/bin/env python3
"""
Causal Inference Module for SFC Terminal
=========================================
Granger Causality-based feature selection for SFC methods.
- Identifies which methods have statistically significant causal relationship with market stress
- Excludes constant/noise methods that provide zero predictive information
- Boosts genuinely causal methods
- Produces dynamic ensemble weights

Usage:
    from causal_inference import CausalFilter
    cf = CausalFilter()
    cf.load_history()
    cf.analyze_all()
    weights = cf.get_weights()
    
    # Apply to current scores
    filtered, active = cf.apply_filter(current_scores_dict)
"""

import json, os, math, sys
from datetime import datetime, timezone
import numpy as np
from scipy import stats as sp_stats

COLLECTION_FILE = os.path.join(os.path.dirname(__file__), "data_collection.json")

METHOD_NAMES = [
    "m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime",
    "m7_fisher", "m8_yield", "m9_liquidity", "m10_garch", "m11_var",
    "m12_jump", "m13_funding", "m14_skew", "m15_concentration",
    "m16_regime_ml", "m17_granger", "m18_entropy", "m19_mutual_info",
    "m20_obi", "m21_trade_flow", "m22_spread", "m23_liquidity",
    "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
    "m28_summers", "m29_debt", "m30_rajan", "m31_altman",
]

# M1-M6 are the CORE — always preserved
CORE_METHODS = set(METHOD_NAMES[:6])


def _compute_variance_metrics(series):
    """Compute variance metrics for a time series."""
    arr = np.array(series, dtype=float)
    unique = len(set(np.round(arr, 6)))  # rounded to avoid floating point noise
    return {
        'std': float(np.std(arr)),
        'unique_ratio': unique / max(len(arr), 1),
        'is_constant': unique <= 1,
        'is_low_var': unique <= 2 and np.std(arr) < 0.01,
        'cv': float(np.std(arr) / max(abs(np.mean(arr)), 1e-6)),  # coefficient of variation
    }


def _granger_test(y, x, max_lag=3):
    """
    Manual Granger Causality Test (OLS-based, no pandas/statsmodels).
    Tests if x Granger-causes y.
    
    H0: x does NOT Granger-cause y (p ≥ 0.05)
    H1: x DOES Granger-cause y (p < 0.05)
    """
    n = min(len(y), len(x))
    y = np.array(y[-n:], dtype=float)
    x = np.array(x[-n:], dtype=float)
    
    # Differencing for stationarity
    dy = np.diff(y)
    dx = np.diff(x)
    
    if len(dy) < max_lag + 3:
        return None
    
    results = []
    for lag in range(1, max_lag + 1):
        T = len(dy) - lag
        if T < lag + 2:
            continue
        
        Y = dy[lag:]
        
        # Restricted: Y ~ Y_lags
        X_r = np.column_stack([np.ones(T)] + [dy[lag-1-i:lag-1-i+T] for i in range(lag)])
        
        # Unrestricted: Y ~ Y_lags + X_lags
        X_u = np.column_stack([np.ones(T)] + 
                              [dy[lag-1-i:lag-1-i+T] for i in range(lag)] +
                              [dx[lag-1-i:lag-1-i+T] for i in range(lag)])
        
        try:
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            resid_r = Y - X_r @ beta_r
            rss_r = resid_r @ resid_r
            df_r = T - X_r.shape[1]
            
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
            resid_u = Y - X_u @ beta_u
            rss_u = resid_u @ resid_u
            df_u = T - X_u.shape[1]
            
            f_stat = ((rss_r - rss_u) / lag) / (rss_u / df_u)
            p_value = 1.0 - sp_stats.f.cdf(f_stat, lag, df_u)
            
            results.append({
                'lag': lag,
                'f_stat': float(f_stat),
                'p_value': float(p_value),
            })
        except np.linalg.LinAlgError:
            continue
    
    if not results:
        return None
    
    best = min(results, key=lambda r: r['p_value'])
    
    return {
        'causes': best['p_value'] < 0.05,
        'p_value': best['p_value'],
        'f_stat': best['f_stat'],
        'best_lag': best['lag'],
        'confidence': 1.0 - best['p_value'],
        'strong_causal': best['p_value'] < 0.01,
    }


CAUSAL_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".causal_cache.json")


class CausalFilter:
    """
    Causal filter for SFC methods.
    
    Loads historical data, runs Granger tests between each method and
    the independent SFC stress target, produces dynamic weights.
    """
    
    def __init__(self, max_lag=3):
        self.max_lag = max_lag
        self.causal_results = {}
        self.variance_metrics = {}
        self.causal_weights = {}
        self.method_timeseries = {}
        self.target_timeseries = None
        self.history_loaded = False
        
    def load_history(self):
        """Load from data_collection.json"""
        if not os.path.exists(COLLECTION_FILE):
            print("[Causal] No data_collection.json", file=sys.stderr)
            return False
        
        try:
            with open(COLLECTION_FILE) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Causal] Failed: {e}", file=sys.stderr)
            return False
        
        features = data.get("features", [])
        labels = data.get("labels", [])
        
        if len(features) < 10:
            print(f"[Causal] Only {len(features)} obs, need ≥10", file=sys.stderr)
            return False
        
        n_methods = len(features[0])
        
        # Build timeseries per method
        for i, name in enumerate(METHOD_NAMES[:n_methods]):
            series = []
            for obs in features:
                val = obs[i] if i < len(obs) and obs[i] is not None else 0.5
                series.append(float(val))
            self.method_timeseries[name] = series
            self.variance_metrics[name] = _compute_variance_metrics(series)
        
        # Build target: composite stress score (independent signal)
        # Use M1-M6 base + M20-M31 institutional + M7-M19 enrichment
        self.target_timeseries = []
        for obs in features:
            vals = [float(v) if v is not None else 0.5 for v in obs]
            if len(vals) >= 6:
                base_avg = sum(vals[:6]) / 6
                new_avg = sum(vals[6:19]) / 13 if len(vals) >= 19 else 0.5
                inst_avg = sum(vals[19:]) / max(len(vals[19:]), 1) if len(vals) > 19 else 0.5
            else:
                base_avg = new_avg = inst_avg = 0.5
            
            composite = 0.85 * base_avg + 0.10 * new_avg + 0.05 * inst_avg
            self.target_timeseries.append(composite)
        
        self.history_loaded = True
        print(f"[Causal] Loaded {len(features)} obs, {n_methods} methods", file=sys.stderr)
        return True
    
    def _save_cache(self):
        """Save causal weights to cache file for reuse."""
        try:
            cache = {
                'causal_weights': self.causal_weights,
                'causal_results': {k: {kk: vv for kk, vv in v.items() if kk != 'reason'} 
                                   for k, v in self.causal_results.items()},
                'ts': datetime.now(timezone.utc).isoformat(),
            }
            with open(CAUSAL_CACHE_FILE, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            print(f"[Causal] Cache save error: {e}", file=sys.stderr)
    
    def _load_cache(self):
        """Load cached causal weights if available and recent (< 24h)."""
        if not os.path.exists(CAUSAL_CACHE_FILE):
            return False
        try:
            with open(CAUSAL_CACHE_FILE) as f:
                cache = json.load(f)
            # Check age
            ts = cache.get('ts', '')
            age_hours = 999
            if ts:
                try:
                    saved = datetime.fromisoformat(ts)
                    age_hours = (datetime.now(timezone.utc) - saved).total_seconds() / 3600
                except:
                    pass
            if age_hours < 24:
                self.causal_weights = {k: float(v) for k, v in cache.get('causal_weights', {}).items()}
                self.causal_results = cache.get('causal_results', {})
                print(f"[Causal] Loaded cached weights ({age_hours:.1f}h old)", file=sys.stderr)
                return True
        except Exception as e:
            print(f"[Causal] Cache load error: {e}", file=sys.stderr)
        return False
    
    def analyze_all(self):
        """
        Run Granger tests on all non-constant methods.
        Returns results dict.
        """
        if not self.history_loaded:
            # Try cache first
            if self._load_cache():
                return self.causal_results
            if not self.load_history():
                return {}
        
        target = self.target_timeseries
        results = {}
        
        for name, series in self.method_timeseries.items():
            vm = self.variance_metrics.get(name, {})
            
            # Skip constant methods (std=0 or unique=1)
            if vm.get('is_constant', False):
                results[name] = {
                    'causes': False,
                    'p_value': 1.0,
                    'best_lag': 0,
                    'confidence': 0.0,
                    'reason': 'CONSTANT — no variance',
                    'variance_ok': False,
                }
                continue
            
            if len(series) < self.max_lag + 5:
                results[name] = {
                    'causes': False,
                    'p_value': 1.0,
                    'best_lag': 0,
                    'confidence': 0.0,
                    'reason': 'INSUFFICIENT_LENGTH',
                    'variance_ok': False,
                }
                continue
            
            granger = _granger_test(target, series, max_lag=self.max_lag)
            
            if granger is None:
                results[name] = {
                    'causes': False,
                    'p_value': 1.0,
                    'best_lag': 0,
                    'confidence': 0.0,
                    'reason': 'TEST_FAILED',
                    'variance_ok': False,
                }
                continue
            
            granger['reason'] = 'TEST_PASS'
            granger['variance_ok'] = True
            results[name] = granger
            
            label = "✓CAUSAL" if granger['causes'] else "✗NO-CAUSE"
            print(f"  {name:20s} → SFC: {label}  (p={granger['p_value']:.4f}, lag={granger['best_lag']})", file=sys.stderr)
        
        # Mark methods that weren't tested
        for name in METHOD_NAMES:
            if name not in results:
                results[name] = {
                    'causes': False, 'p_value': 1.0,
                    'best_lag': 0, 'confidence': 0.0,
                    'reason': 'NO_DATA', 'variance_ok': False,
                }
        
        self.causal_results = results
        self._compute_weights()
        self._save_cache()
        return results
    
    def _compute_weights(self):
        """
        Compute final dynamic weights using:
        1. Variance filter — constant methods get weight=0
        2. Granger causality — causal methods get boosted
        3. Core protection — M1-M6 protected from exclusion
        
        Rules:
        - CONSTANT (std=0, unique=1):          weight = 0.0  → EXCLUDE
        - LOW-VAR (unique≤2, std<0.01):        weight = 0.2  → near-exclude
        - CORE (M1-M6):                         weight = 1.0  → protected
        - CORE + significant:                   weight = 1.5
        - STRONG CAUSAL (p<0.01):               weight = 2.0
        - CAUSAL (p<0.05):                      weight = 1.5
        - MARGINAL (p<0.15, core methods only): weight = 1.0
        - MARGINAL (p<0.15, non-core):          weight = 0.8
        - NON-CAUSAL with variance:             weight = 0.5
        - NON-CAUSAL low-var:                   weight = 0.2
        """
        weights = {}
        
        for name in METHOD_NAMES:
            r = self.causal_results.get(name, {})
            vm = self.variance_metrics.get(name, {})
            p_val = r.get('p_value', 1.0)
            is_core = name in CORE_METHODS
            is_constant = vm.get('is_constant', False)
            is_low_var = vm.get('is_low_var', False)
            
            # 1. EXCLUDE constant methods entirely
            if is_constant:
                weights[name] = 0.0
                continue
            
            # 2. Near-exclude low-variance noise
            if is_low_var:
                weights[name] = 0.2 if not is_core else 0.8
                continue
            
            # 3. CORE methods — protected
            if is_core:
                if r.get('strong_causal', False):
                    weights[name] = 1.5
                elif r.get('causes', False):
                    weights[name] = 1.3
                elif p_val < 0.15:
                    weights[name] = 1.0  # marginal but acceptable for core
                else:
                    weights[name] = 0.8  # non-significant but keep as core
                continue
            
            # 4. Non-core methods based on Granger
            if r.get('strong_causal', False):
                weights[name] = 2.0
            elif r.get('causes', False):
                weights[name] = 1.5
            elif p_val < 0.10:
                weights[name] = 1.0  # marginal
            elif p_val < 0.20:
                weights[name] = 0.7  # weak
            else:
                weights[name] = 0.3  # non-causal
        
        self.causal_weights = weights
        return weights
    
    def get_weights(self):
        """Return dict of method_name → weight multiplier"""
        if not self.causal_weights:
            self._compute_weights()
        return self.causal_weights
    
    def apply_filter(self, method_scores_dict, min_weight=0.2):
        """
        Apply causal filter to current method scores.
        
        Args:
            method_scores_dict: dict of {method_key: score} 
            min_weight: minimum weight to include method in active ensemble
        
        Returns:
            filtered_scores: {method_key: adjusted_score}
            active: list of active method keys
        """
        weights = self.get_weights()
        
        filtered = {}
        active = []
        excluded = []
        
        for name, score in method_scores_dict.items():
            if score is None:
                continue
            
            w = weights.get(name, 0.5)
            
            if w >= min_weight:
                adjusted = min(1.0, max(0.0, score * w))
                filtered[name] = adjusted
                active.append(name)
            else:
                excluded.append((name, w))
        
        for name, w in excluded:
            print(f"  [Causal] EXCLUDED {name}: weight={w:.2f}", file=sys.stderr)
        
        return filtered, active
    
    def get_excluded_methods(self):
        """Get list of methods that should be excluded (weight < 0.2)."""
        weights = self.get_weights()
        excluded = [n for n, w in weights.items() if w < 0.2]
        low_weight = [(n, w) for n, w in weights.items() if 0.2 <= w < 0.5]
        return excluded, low_weight
    
    def get_blend_adjustment(self):
        """
        Get dynamic ensemble blend percentages.
        Adjust the 85/10/5 blend based on how many methods in each group are active.
        """
        weights = self.get_weights()
        
        def active_ratio(names):
            w = [weights.get(n, 0) for n in names]
            active = sum(1 for x in w if x >= 0.2)
            return active / max(len(names), 1)
        
        r1 = active_ratio(METHOD_NAMES[:6])     # M1-M6
        r2 = active_ratio(METHOD_NAMES[6:19])   # M7-M19
        r3 = active_ratio(METHOD_NAMES[19:])    # M20-M31
        
        # Base: 85/10/5, adjusted by active ratios
        total = 85*r1 + 10*r2 + 5*r3
        if total > 0:
            p1 = 85*r1 / total * 100
            p2 = 10*r2 / total * 100
            p3 = 5*r3 / total * 100
        else:
            p1, p2, p3 = 100, 0, 0
        
        return {
            'm1_m6_pct': round(p1, 1),
            'm7_m19_pct': round(p2, 1),
            'm20_m31_pct': round(p3, 1),
            'm7_m19_active_ratio': round(r2, 3),
            'm20_m31_active_ratio': round(r3, 3),
        }
    
    def get_report(self):
        """Generate human-readable causal report."""
        if not self.causal_results:
            return "No causal analysis performed."
        
        lines = []
        lines.append("╔" + "═"*58 + "╗")
        lines.append("║  CAUSAL INFERENCE REPORT — Granger Causality Filter       ║")
        lines.append("╠" + "═"*58 + "╣")
        lines.append(f"║ {'Method':20s} {'Status':14s} {'p-value':8s} {'Lag':4s} {'Weight':6s} ║")
        lines.append("╠" + "═"*58 + "╣")
        
        sig_count = 0
        const_count = 0
        excluded_count = 0
        
        for name in METHOD_NAMES:
            r = self.causal_results.get(name, {})
            w = self.causal_weights.get(name, 0.5)
            
            if r.get('reason') == 'CONSTANT':
                status = "🗑 CONSTANT"
                const_count += 1
                excluded_count += 1
            elif w < 0.2:
                status = "✗ EXCLUDED"
                excluded_count += 1
            elif r.get('causes', False):
                status = "✓✓ CAUSAL" if r.get('strong_causal', False) else "✓ CAUSAL"
                sig_count += 1
            elif r.get('variance_ok', False):
                status = "✗ NO-CAUSE"
            else:
                status = "✗ NO-DATA"
            
            p_str = f"{r.get('p_value', 1.0):.4f}" if r.get('reason') not in ('CONSTANT', 'NO_DATA') else "—"
            lag_str = f"{r.get('best_lag', 0)}" if r.get('best_lag', 0) > 0 else "—"
            
            lines.append(f"║ {name:20s} {status:14s} {p_str:>8s} {lag_str:>4s} {w:6.2f} ║")
        
        lines.append("╠" + "═"*58 + "╣")
        lines.append(f"║ Total: {len(METHOD_NAMES)} methods                                     ║")
        lines.append(f"║ Causal (p<0.05): {sig_count} | Constant/Excluded: {excluded_count}              ║")
        
        # List excluded methods
        excluded, _ = self.get_excluded_methods()
        if excluded:
            lines.append(f"║ Excluded from ensemble: {', '.join(excluded):48s} ║")
        
        # Blend adjustment
        adj = self.get_blend_adjustment()
        lines.append("╠" + "═"*58 + "╣")
        lines.append(f"║ Dynamic blend: M1-M6={adj['m1_m6_pct']:.0f}% | "
                     f"M7-M19={adj['m7_m19_pct']:.0f}% | "
                     f"M20-M31={adj['m20_m31_pct']:.0f}%     ║")
        lines.append("╚" + "═"*58 + "╝")
        
        return '\n'.join(lines)


def compute_causal_weights():
    """Convenience: load, analyze, return (cf, weights, report, adj)."""
    cf = CausalFilter(max_lag=3)
    if cf.load_history():
        cf.analyze_all()
        weights = cf.get_weights()
        report = cf.get_report()
        adj = cf.get_blend_adjustment()
        return cf, weights, report, adj
    return cf, {}, "No data available.", {}


if __name__ == "__main__":
    print("[Causal] Running Granger Causality analysis...", file=sys.stderr)
    cf, weights, report, adj = compute_causal_weights()
    print("\n" + report)
    
    excluded, low = cf.get_excluded_methods()
    if excluded:
        print(f"\n→ EXCLUDED (no signal): {', '.join(excluded)}")
    if low:
        print(f"→ REDUCED WEIGHT: {', '.join(f'{n}({w:.2f})' for n,w in low)}")
    
    print(f"\n→ Blend: M1-M6={adj['m1_m6_pct']:.0f}% / M7-M19={adj['m7_m19_pct']:.0f}% / M20-M31={adj['m20_m31_pct']:.0f}%")
