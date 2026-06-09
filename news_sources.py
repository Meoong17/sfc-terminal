"""
news_sources.py — Multi-source free news aggregator
20+ free sources including RSS feeds, Reddit, Google News
"""

import os, re, html, time, hashlib
from datetime import datetime, timezone, timedelta

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Sentiment dictionaries
BEARISH = {
    "crash","liquidation","panic","crisis","collapse","bankrupt","freeze",
    "default","downgrade","recession","inflation","rate hike","tighten",
    "warning","risk","contagion","selloff","plunge","tumble","slump","loss",
    "debt","deficit","instability","volatile","drain","halt","emergency",
    "bailout","stress","toxic","overvalued","bubble","bear","dump","lawsuit",
    "hack","exploit","rug","scam","ban","regulate","crackdown","suspend",
    "insolvent","margin call","deleveraging","write-off","withdrawal halt",
    "circuit breaker","flash crash","cascading","contagion","systemic",
}
BULLISH = {
    "recovery","surge","bullish","rally","breakthrough","liquidity","growth",
    "expansion","profit","gain","uptick","momentum","inflow","stable",
    "confidence","positive","upgrade","all-time high","ath","boom",
    "outperform","dividend","buyback","oversubscribed","allocation","approve",
    "etf approval","institutional","adoption","partnership","launch","listing",
    "accumulate","hodl","spot etf","spot bitcoin","treasury","reserve",
}

# RSS Feed Definitions
RSS_FEEDS = [
    {"url": "https://cointelegraph.com/rss", "name": "CoinTelegraph", "weight": 2.5, "factor": "Rt"},
    {"url": "https://coindesk.com/arc/outboundfeeds/rss/", "name": "CoinDesk", "weight": 2.5, "factor": "Rt"},
    {"url": "https://decrypt.co/feed", "name": "Decrypt", "weight": 2.0, "factor": "Rt"},
    {"url": "https://theblock.co/rss.xml", "name": "TheBlock", "weight": 2.0, "factor": "Ft"},
    {"url": "https://beincrypto.com/feed/", "name": "BeInCrypto", "weight": 1.5, "factor": "Rt"},
    {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "name": "WSJ Markets", "weight": 3.0, "factor": "Ft"},
    {"url": "https://feeds.reuters.com/reuters/businessNews", "name": "Reuters Biz", "weight": 3.0, "factor": "Ft"},
    {"url": "https://feeds.reuters.com/reuters/topNews", "name": "Reuters Top", "weight": 2.5, "factor": "Sc"},
    {"url": "https://www.investing.com/rss/news_14.rss", "name": "Investing.com", "weight": 2.0, "factor": "Lt"},
    {"url": "https://www.zerohedge.com/fullrss2.xml", "name": "ZeroHedge", "weight": 2.5, "factor": "Ft"},
    {"url": "https://www.ft.com/rss/home/uk", "name": "FT", "weight": 3.0, "factor": "Ft"},
    {"url": "https://www.reddit.com/r/Bitcoin/hot.rss?limit=10", "name": "r/Bitcoin", "weight": 1.5, "factor": "Rt"},
    {"url": "https://www.reddit.com/r/economics/hot.rss?limit=10", "name": "r/Economics", "weight": 2.0, "factor": "Lt"},
    {"url": "https://www.reddit.com/r/wallstreetbets/hot.rss?limit=5", "name": "r/WSB", "weight": 1.5, "factor": "Rt"},
    {"url": "https://www.reddit.com/r/CryptoCurrency/hot.rss?limit=10", "name": "r/Crypto", "weight": 1.5, "factor": "Rt"},
    {"url": "https://news.google.com/rss/search?q=bitcoin+cryptocurrency&hl=en-US&gl=US&ceid=US:en", "name": "Google:BTC", "weight": 2.0, "factor": "Rt"},
    {"url": "https://news.google.com/rss/search?q=federal+reserve+interest+rate&hl=en-US&gl=US&ceid=US:en", "name": "Google:Fed", "weight": 2.5, "factor": "Lt"},
    {"url": "https://news.google.com/rss/search?q=global+financial+crisis+bank+stress&hl=en-US&gl=US&ceid=US:en", "name": "Google:Crisis", "weight": 3.0, "factor": "Ft"},
    {"url": "https://www.binance.com/en/support/announcement/c-48?rss=1", "name": "Binance", "weight": 2.0, "factor": "St"},
]

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 SFC-Terminal/7"}

def get_cryptopanic(key=None):
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={key}&public=true&kind=news&filter=hot" if key else "https://cryptopanic.com/api/v1/posts/?public=true&kind=news&filter=hot"
        r = requests.get(url, timeout=10, headers=HEADERS)
        if r.status_code != 200:
            return []
        items = r.json().get("results", [])[:15]
        out = []
        for item in items:
            title = item.get("title", "")
            votes = item.get("votes", {})
            positive = votes.get("positive", 0)
            negative = votes.get("negative", 0)
            sent_bias = (positive - negative) / (positive + negative) if positive + negative > 0 else 0.0
            currencies = [c.get("code","") for c in item.get("currencies", [])]
            btc_related = "BTC" in currencies or "ETH" in currencies
            out.append({"title": title, "source": "CryptoPanic", "factor": "Rt", "weight": 2.5 if btc_related else 1.5, "community_sentiment": round(sent_bias, 2), "age_hours": 0})
        return out
    except:
        return []

def fetch_rss_feed(feed_def, max_age_hours=6, max_items=8):
    if not HAS_FEEDPARSER:
        return []
    url = feed_def["url"]
    try:
        feed = feedparser.parse(url, request_headers=HEADERS)
        items = []
        now = datetime.now(timezone.utc)
        for entry in feed.entries[:max_items]:
            title = html.unescape(getattr(entry, "title", "") or "")
            title = re.sub(r"<[^>]+>", "", title).strip()
            if len(title) < 10:
                continue
            published = None
            for attr in ["published_parsed", "updated_parsed", "created_parsed"]:
                val = getattr(entry, attr, None)
                if val:
                    try:
                        import calendar
                        published = datetime.fromtimestamp(calendar.timegm(val), tz=timezone.utc)
                    except:
                        pass
                    break
            age_hours = 999
            if published:
                age_hours = (now - published).total_seconds() / 3600
            if age_hours > max_age_hours:
                continue
            items.append({"title": title, "source": feed_def["name"], "factor": feed_def["factor"], "weight": feed_def["weight"], "age_hours": round(age_hours, 1)})
        return items
    except:
        return []

def score_article(article):
    title_lower = article["title"].lower()
    words = title_lower.split()
    bear = sum(1 for w in words if w in BEARISH)
    bull = sum(1 for w in words if w in BULLISH)
    for phrase in ["rate hike","margin call","bank run","flash crash","circuit breaker","all-time high","spot etf","etf approval","emergency cut"]:
        if phrase in title_lower:
            if phrase in {"all-time high","spot etf","etf approval","emergency cut"}:
                bull += 2
            else:
                bear += 2
    net_sentiment = min(max(bull - bear, -3), 3)
    stress_contrib = 0.0
    if bear > 0:
        stress_contrib = min(bear * article["weight"] * 0.5, article["weight"] * 2)
    if bull > 0:
        stress_contrib -= min(bull * article["weight"] * 0.3, article["weight"])
    return {**article, "sentiment": net_sentiment, "stress_contrib": round(stress_contrib, 2), "bear_words": bear, "bull_words": bull}

def get_news_stress_v2(cryptopanic_key=None, max_workers=8):
    all_articles = []
    
    import concurrent.futures
    def fetch_one(feed_def):
        try:
            return fetch_rss_feed(feed_def, max_age_hours=6, max_items=6)
        except:
            return []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(fetch_one, RSS_FEEDS))
    for r in results:
        all_articles.extend(r)
    
    cp = get_cryptopanic(cryptopanic_key)
    all_articles.extend(cp)
    
    seen = set()
    unique = []
    for a in all_articles:
        h = hashlib.md5(a["title"][:60].lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(a)
    
    scored = [score_article(a) for a in unique]
    scored.sort(key=lambda x: x["stress_contrib"], reverse=True)
    
    total_stress = 0.0
    sentiments = []
    for a in scored:
        total_stress += a["stress_contrib"]
        sentiments.append(a["sentiment"])
    
    total_stress = round(min(total_stress, 30.0), 1)
    avg_sentiment = round(sum(sentiments) / len(sentiments), 3) if sentiments else 0.0
    
    headlines = []
    for a in scored[:8]:
        icon = "🔴" if a["sentiment"] < -0.5 else "🟢" if a["sentiment"] > 0.5 else "⚪"
        age_str = f"{a['age_hours']:.1f}h" if a["age_hours"] < 99 else ""
        src = a["source"][:12]
        title_short = a["title"][:75]
        tag = f"[{a['factor']}]"
        headlines.append(f"{icon}{tag}[{src}] {title_short}")
    
    stats = {"total_articles": len(scored), "sources_hit": len(set(a["source"] for a in scored)), "stress_raw": round(sum(a["stress_contrib"] for a in scored), 2), "stress_capped": total_stress}
    
    return total_stress, headlines[:8], avg_sentiment, scored[:20], stats

SHOCK_MATRIX = {
    "emergency rate cut": 0.40, "fed emergency": 0.35, "liquidity facility": 0.30,
    "bailout": 0.25, "ceasefire": 0.20, "truce": 0.15, "spot etf approved": 0.20,
    "nuclear": -0.50, "war declared": -0.45, "military strike": -0.35,
    "margin call": -0.25, "exchange halt": -0.30, "bank run": -0.35,
    "default": -0.25, "insolvency": -0.30, "circuit breaker": -0.25,
    "flash crash": -0.20, "cascading liquidations": -0.30, "tether insolvency": -0.45,
    "binance hack": -0.35, "exchange insolvent": -0.40, "sec charges": -0.15,
}

def detect_black_swan_v2(articles):
    worst_shock = 0.0
    worst_event = None
    worst_severity = "NONE"
    for a in articles:
        text = a["title"].lower()
        for keyword, shock in SHOCK_MATRIX.items():
            if keyword in text:
                if abs(shock) > abs(worst_shock):
                    worst_shock = shock
                    worst_event = f"{a['source']}: {a['title'][:60]}"
                    worst_severity = "CRITICAL" if abs(shock) >= 0.35 else "HIGH" if abs(shock) >= 0.20 else "MEDIUM"
    return worst_shock, worst_event, worst_severity
