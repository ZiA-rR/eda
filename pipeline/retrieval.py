"""
retrieval.py
------------
Price data and news retrieval.

NOTE: none of this can run in a restricted network. Run it on Colab or a
local machine with normal internet access.

Price data      : yfinance (free, no API key)
News discovery  : GDELT DOC 2.0 API (free, no API key, covers 2017 onward)
Article text    : trafilatura (handles boilerplate removal reasonably well)

The news source is deliberately kept behind one function so it can be
swapped. AER used serper.dev (Google News), which is paid but gives better
coverage. GDELT is the free option and is good enough to start.
"""

import time
import datetime as dt
from typing import List, Dict, Optional

import pandas as pd
import requests


# ---------------------------------------------------------------- tickers
# yfinance symbols for the four asset classes.
TICKERS = {
    "crypto": "BTC-USD",     # Bitcoin
    "gold":   "GC=F",        # COMEX gold futures
    "petrol": "BZ=F",        # Brent crude futures
    "forex":  "GBPUSD=X",    # sterling/dollar
}

# Extra symbols worth having, e.g. to widen the forex or crypto coverage later
EXTRA_TICKERS = {
    "crypto_eth": "ETH-USD",
    "gold_spot":  "XAUUSD=X",
    "petrol_wti": "CL=F",
    "forex_eur":  "EURUSD=X",
    "forex_jpy":  "JPY=X",
}


def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    Daily close prices. Requires internet.

        pip install yfinance
    """
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    close = df["Close"]
    # yfinance sometimes returns a one-column frame instead of a series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


# ---------------------------------------------------------------- news
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def search_news(query: str,
                date: str,
                days_before: int = 2,
                days_after: int = 1,
                max_records: int = 40,
                timeout: int = 30) -> List[Dict]:
    """
    Find articles around a date using GDELT.

    date        : "YYYY-MM-DD", the big move day
    days_before : how far back to look. Causes usually precede the move, so
                  this is wider than days_after.
    days_after  : a small window after, to catch same-story reporting that
                  landed late. Keep this SMALL, otherwise you pull in
                  consequences of the move and mislabel them as causes.
    """
    d = dt.datetime.strptime(date, "%Y-%m-%d")
    start = (d - dt.timedelta(days=days_before)).strftime("%Y%m%d%H%M%S")
    end = (d + dt.timedelta(days=days_after)).strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "startdatetime": start,
        "enddatetime": end,
        "format": "json",
        "sort": "hybridrel",
    }

    try:
        resp = requests.get(GDELT_URL, params=params, timeout=timeout,
                            headers={"User-Agent": "research-dataset-builder"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  GDELT request failed for '{query}' on {date}: {e}")
        return []

    articles = data.get("articles", [])
    return [{
        "title": a.get("title", ""),
        "url": a.get("url", ""),
        "source": a.get("domain", ""),
        "date": a.get("seendate", ""),
        "language": a.get("language", ""),
    } for a in articles]


def extract_text(url: str, timeout: int = 20) -> Optional[str]:
    """
    Pull the main body text from an article URL.

        pip install trafilatura

    Returns None on paywalls, blocks, or extraction failure, which is
    common. Expect to lose a meaningful share of URLs.
    """
    import trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False,
                                   include_tables=False)
        return text
    except Exception:
        return None


# ------------------------------------------------------- query building
# Queries for the on-story documents, per asset.
BASE_QUERIES = {
    "gold":   "gold price",
    "crypto": "bitcoin price",
    "petrol": "oil price crude",
    "forex":  "pound sterling dollar",
}

# Distractor queries: topically adjacent, but likely to return articles
# OUTSIDE the causal chain of any specific day's move. This mirrors what AER
# did, and it is what stops retrieval from being trivial.
DISTRACTOR_QUERIES = {
    "gold":   ["gold mining company", "gold jewellery demand", "central bank gold reserves"],
    "crypto": ["blockchain technology adoption", "crypto regulation proposal", "NFT market"],
    "petrol": ["refinery maintenance", "electric vehicle sales", "renewable energy investment"],
    "forex":  ["UK retail sales", "tourism spending", "trade balance figures"],
}


def build_topic(asset: str,
                date: str,
                n_relevant: int = 15,
                n_distractor: int = 5,
                fetch_text: bool = True,
                polite_delay: float = 1.0) -> Dict:
    """
    Assemble the document set for one big move day.

    Targets roughly 20 documents per topic, matching AER's 19.7 average,
    with a deliberate share of distractor documents mixed in.
    """
    docs = []

    # on-story documents
    hits = search_news(BASE_QUERIES[asset], date, max_records=n_relevant * 2)
    for h in hits[:n_relevant]:
        entry = {**h, "role": "relevant"}
        if fetch_text:
            entry["content"] = extract_text(h["url"])
            time.sleep(polite_delay)
        docs.append(entry)

    # distractor documents
    per_query = max(1, n_distractor // len(DISTRACTOR_QUERIES[asset]))
    for q in DISTRACTOR_QUERIES[asset]:
        hits = search_news(q, date, max_records=per_query * 2)
        for h in hits[:per_query]:
            entry = {**h, "role": "distractor"}
            if fetch_text:
                entry["content"] = extract_text(h["url"])
                time.sleep(polite_delay)
            docs.append(entry)

    kept = [d for d in docs if not fetch_text or d.get("content")]

    return {
        "asset": asset,
        "event_date": date,
        "docs": kept,
        "n_requested": len(docs),
        "n_with_text": len(kept),
    }
