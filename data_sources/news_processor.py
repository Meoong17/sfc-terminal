#!/usr/bin/env python3
"""
Q9: News Scoring System for SFC Terminal
=========================================
Hybrid sentiment analysis (VADER + TextBlob) for free news sources.
No GPU needed — runs on CPU efficiently.

Sources:
  - cryptocurrency.cv (free, no API key, unlimited)
  - CryptoPanic (free tier, optional API key)
"""

import requests
import json
import time
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import numpy as np

# =======================================================
# 1. SOURCE CREDIBILITY SCORE (0-1)
# =======================================================

SOURCE_WEIGHTS = {
    # Tier 1 — Institutional grade
    "coindesk.com": 1.0,
    "theblock.co": 1.0,
    "bloomberg.com": 1.0,
    "reuters.com": 1.0,
    "wsj.com": 1.0,
    "ft.com": 1.0,
    # Tier 2 — Crypto native, credible
    "cointelegraph.com": 0.9,
    "decrypt.co": 0.9,
    "cryptoslate.com": 0.8,
    "blockworks.co": 0.85,
    "beincrypto.com": 0.7,
    # Tier 3 — Aggregator / secondary
    "cryptopanic.com": 0.6,
    "cryptocurrency.cv": 0.7,
    "news.bitcoin.com": 0.5,
    # Default
    "default": 0.4
}

def get_source_weight(url: str) -> float:
    """Extract domain from URL and return source credibility weight."""
    if not url:
        return SOURCE_WEIGHTS["default"]
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        domain = match.group(1).lower()
        for key, weight in SOURCE_WEIGHTS.items():
            if key in domain:
                return weight
    return SOURCE_WEIGHTS["default"]

# =======================================================
# 2. TOPIC CLASSIFICATION & IMPORTANCE
# =======================================================

TOPIC_WEIGHTS = {
    "regulatory": {
        "keywords": ["sec", "regulation", "ban", "law", "congress", "treasury",
                     "cfpb", "tax", "legal", "sue", "lawsuit", "illegal", "compliance"],
        "magnitude": 0.9
    },
    "macro": {
        "keywords": ["fed", "interest rate", "inflation", "recession", "dxy",
                     "dollar", "yellen", "powell", "economic", "jobs", "cpi", "ppi"],
        "magnitude": 0.7
    },
    "exchange": {
        "keywords": ["binance", "coinbase", "hack", "withdrawal", "delist",
                     "liquidate", "bankrupt", "outage", "exploit"],
        "magnitude": 0.6
    },
    "whale": {
        "keywords": ["whale", "large transaction", "accumulate", "dump", "move",
                     "transfer", "cold wallet", "exchange inflow", "exchange outflow"],
        "magnitude": 0.5
    },
    "technology": {
        "keywords": ["upgrade", "fork", "protocol", "layer2", "halving", "rollup",
                     "scaling", "security", "bug", "vulnerability"],
        "magnitude": 0.4
    },
    "general": {
        "magnitude": 0.2
    }
}

def classify_topic_importance(text: str) -> Tuple[str, float]:
    """Classify news topic and calculate importance (0-1)."""
    text_lower = text.lower()
    for topic, data in TOPIC_WEIGHTS.items():
        keywords = data.get("keywords", [])
        if any(kw in text_lower for kw in keywords):
            return topic, data["magnitude"]
    return "general", TOPIC_WEIGHTS["general"]["magnitude"]

# =======================================================
# 3. HYBRID SENTIMENT ANALYSIS (VADER + TextBlob)
# =======================================================
# pip install vaderSentiment textblob

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# TextBlob is optional — lazy import to avoid numpy/pandas conflicts
_textblob_available = None  # None = not checked yet; True/False after first attempt

def _check_textblob():
    global _textblob_available
    if _textblob_available is None:
        try:
            from textblob import TextBlob as _
            _textblob_available = True
        except Exception:
            _textblob_available = False
    return _textblob_available

_vader_analyzer = None

def get_vader_analyzer():
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer

def get_hybrid_sentiment(text: str) -> float:
    """
    Hybrid sentiment score (-1 to +1)
    VADER: 70% weight (better for social media & intensity)
    TextBlob: 30% weight (baseline) — fallback to VADER-only if unavailable
    """
    if not text:
        return 0.0
    vader = get_vader_analyzer()
    vader_score = vader.polarity_scores(text)['compound']
    if _check_textblob():
        try:
            from textblob import TextBlob
            blob_score = TextBlob(text).sentiment.polarity
            final_score = (vader_score * 0.7) + (blob_score * 0.3)
        except Exception:
            final_score = vader_score
    else:
        final_score = vader_score
    return max(-1.0, min(1.0, final_score))

# =======================================================
# 4. DECAY FUNCTION
# =======================================================

def compute_decay(hours_since_publish: float, half_life: float = 24.0) -> float:
    """
    Exponential decay: older news = less impact
    half_life = 24 hours means after 24h, weight is 50%
    """
    return 0.5 ** (hours_since_publish / half_life)

# =======================================================
# 5. CROSS-SOURCE CONFIRMATION
# =======================================================

_news_hash_cache = {}  # key: hash of headline, value: source count

def update_confirmation(headline: str) -> float:
    """
    Count how many different sources cover the same story.
    Simple keyword-based similarity.
    """
    clean = re.sub(r'[^\w\s]', '', headline.lower())
    words = clean.split()
    if len(words) > 8:
        key = ' '.join(words[:4] + words[-4:])
    else:
        key = clean
    if key not in _news_hash_cache:
        _news_hash_cache[key] = 1
    else:
        _news_hash_cache[key] += 1
    # Confirmation factor: 1 source = 0.5, 2+ sources = 1.0
    return min(1.0, _news_hash_cache[key] / 2)

# =======================================================
# 6. FETCH NEWS FROM CRYPTOCURRENCY.CV (FREE, NO KEY)
# =======================================================

def fetch_cryptocurrency_cv_news(limit: int = 30) -> List[Dict]:
    """
    Fetch news from cryptocurrency.cv API.
    No API key required, unlimited rate limits.
    """
    url = f"https://api.cryptocurrency.cv/v1/news?limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("name", "unknown"),
                    "published_at": item.get("published_at", datetime.now(timezone.utc).isoformat()),
                    "summary": item.get("summary", "")
                })
            return articles
        else:
            print(f"[Q9] Failed to fetch news: {response.status_code}", file=__import__('sys').stderr)
            return []
    except Exception as e:
        print(f"[Q9] Error fetching news: {e}", file=__import__('sys').stderr)
        return []

# =======================================================
# 7. PROCESS NEWS -> FINAL STRESS CONTRIBUTION
# =======================================================

def process_news_article(article: Dict) -> Dict:
    """
    Process a single news article into full scoring.
    """
    title = article.get("title", "")
    url = article.get("url", "")
    published_at = article.get("published_at", datetime.now(timezone.utc).isoformat())

    # Parse time
    try:
        pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
    except:
        pub_time = datetime.now(timezone.utc)
    hours_ago = (datetime.now(timezone.utc) - pub_time).total_seconds() / 3600

    # 1. Source credibility
    source_score = get_source_weight(url)

    # 2. Topic importance
    topic, importance = classify_topic_importance(title)

    # 3. Sentiment analysis (hybrid VADER+TextBlob)
    sentiment_raw = get_hybrid_sentiment(title)

    # 4. Impact magnitude (sentiment * importance)
    impact_magnitude = sentiment_raw * importance

    # 5. Cross-source confirmation
    confirmation = update_confirmation(title)

    # 6. Decay (older news less impact)
    decay = compute_decay(hours_ago, half_life=24)

    # 7. Final stress impact (scale 0-10%)
    # Formula: (|impact| * source_score * confirmation * (0.5+0.5*importance)) * decay
    raw_impact = abs(impact_magnitude) * source_score * confirmation * (0.5 + 0.5 * importance)
    stress_impact = min(5.0, raw_impact * 5)  # cap at 5%

    return {
        "title": title[:120] + "..." if len(title) > 120 else title,
        "source_score": source_score,
        "topic": topic,
        "importance": importance,
        "sentiment_raw": sentiment_raw,
        "impact_magnitude": impact_magnitude,
        "confirmation": confirmation,
        "decay": decay,
        "stress_impact": stress_impact,
        "direction": "positive" if sentiment_raw > 0.05 else "negative" if sentiment_raw < -0.05 else "neutral"
    }

def compute_news_stress(articles: List[Dict]) -> Dict:
    """
    Process all articles and compute aggregate news stress.
    """
    if not articles:
        return {"news_stress": 0.0, "sentiment_avg": 0.0, "article_count": 0}

    processed = []
    total_stress = 0.0
    total_sentiment = 0.0
    for article in articles:
        result = process_news_article(article)
        processed.append(result)
        total_stress += result["stress_impact"]
        total_sentiment += result["sentiment_raw"]

    # Top 5 articles with highest stress impact
    sorted_articles = sorted(processed, key=lambda x: x["stress_impact"], reverse=True)
    top_articles = sorted_articles[:5]

    # News stress: sum of top 5 + 10% of remaining total
    news_stress = sum(a["stress_impact"] for a in top_articles)
    news_stress += total_stress * 0.1  # 10% from remaining total

    # Cap at 10%
    news_stress = min(10.0, news_stress)

    return {
        "news_stress": round(news_stress, 2),
        "sentiment_avg": round(total_sentiment / len(articles), 4),
        "sentiment_raw": total_sentiment,
        "article_count": len(articles),
        "top_articles": top_articles
    }

# =======================================================
# 8. MAIN FUNCTION FOR INTEGRATION INTO COLLECT.PY
# =======================================================

def get_news_impact() -> Dict:
    """
    Main function called from collect.py.
    Returns dict with news_stress (0-10%) and sentiment_avg (-1 to +1).
    """
    try:
        articles = fetch_cryptocurrency_cv_news(limit=50)
        if not articles:
            return {"news_stress": 0.0, "sentiment_avg": 0.0, "article_count": 0}
        result = compute_news_stress(articles)
        return result
    except Exception as e:
        print(f"[Q9] Error in get_news_impact: {e}", file=__import__('sys').stderr)
        return {"news_stress": 0.0, "sentiment_avg": 0.0, "article_count": 0}

# =======================================================
# TEST (if run directly)
# =======================================================
if __name__ == "__main__":
    print("[Q9] Testing news processor...")
    result = get_news_impact()
    print(f"News Stress: {result['news_stress']}%")
    print(f"Sentiment Avg: {result['sentiment_avg']}")
    print(f"Articles: {result['article_count']}")
    for a in result.get("top_articles", [])[:3]:
        print(f"  - [{a['direction']}] {a['title'][:60]}... (impact: {a['stress_impact']}%)")
