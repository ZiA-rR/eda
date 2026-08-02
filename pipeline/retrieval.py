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

# GDELT allows roughly one request every 5 seconds. Going faster returns
# 429 Too Many Requests. build_topic makes four searches per topic (one for
# the story, three for distractors), so without a gap between them the
# second and third reliably fail.
#
# This is enforced inside search_news rather than passed in as an option,
# because every caller needs it and forgetting is the whole problem.
GDELT_MIN_INTERVAL = 6.0
_last_gdelt_call = 0.0


def _gdelt_wait():
    """Sleep just long enough since the previous GDELT call."""
    global _last_gdelt_call
    elapsed = time.time() - _last_gdelt_call
    if elapsed < GDELT_MIN_INTERVAL:
        time.sleep(GDELT_MIN_INTERVAL - elapsed)
    _last_gdelt_call = time.time()


def search_news(query: str,
                date: str,
                days_before: int = 2,
                days_after: int = 1,
                max_records: int = 40,
                timeout: int = 30,
                english_only: bool = True,
                retries: int = 4,
                verbose: bool = True) -> List[Dict]:
    """
    Find articles around a date using GDELT.

    date        : "YYYY-MM-DD", the big move day
    days_before : how far back to look. Causes usually precede the move, so
                  this is wider than days_after.
    days_after  : a small window after, to catch same-story reporting that
                  landed late. Keep this SMALL, otherwise you pull in
                  consequences of the move and mislabel them as causes.
    english_only: GDELT indexes many languages. Without this you get
                  Romanian and Spanish articles mixed in, which the
                  extraction stage cannot use.

    Waits between calls and retries on 429 with escalating backoff.
    """
    d = dt.datetime.strptime(date, "%Y-%m-%d")
    start = (d - dt.timedelta(days=days_before)).strftime("%Y%m%d%H%M%S")
    end = (d + dt.timedelta(days=days_after)).strftime("%Y%m%d%H%M%S")

    q = f"{query} sourcelang:english" if english_only else query

    params = {
        "query": q,
        "mode": "artlist",
        "maxrecords": max_records,
        "startdatetime": start,
        "enddatetime": end,
        "format": "json",
        "sort": "hybridrel",
    }

    for attempt in range(retries):
        _gdelt_wait()
        try:
            resp = requests.get(GDELT_URL, params=params, timeout=timeout,
                                headers={"User-Agent": "research-dataset-builder"})

            if resp.status_code == 429:
                wait = GDELT_MIN_INTERVAL * (2 ** attempt)
                if verbose:
                    print(f"  rate limited, waiting {wait:.0f}s "
                          f"(attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            if attempt == retries - 1:
                print(f"  GDELT failed for '{query}' on {date}: {e}")
                return []
            time.sleep(GDELT_MIN_INTERVAL * (2 ** attempt))
            continue

        articles = data.get("articles", [])
        out = [{
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("domain", ""),
            "date": a.get("seendate", ""),
            "language": a.get("language", ""),
        } for a in articles]

        if english_only:
            out = [a for a in out
                   if not a["language"] or a["language"].lower().startswith("eng")]
        return out

    print(f"  GDELT gave up on '{query}' after {retries} attempts")
    return []


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
                polite_delay: float = 1.0,
                min_docs: int = 5,
                verbose: bool = True) -> Dict:
    """
    Assemble the document set for one big move day.

    Targets roughly 20 documents per topic, matching AER's 19.7 average,
    with a deliberate share of distractor documents mixed in.

    polite_delay applies between ARTICLE downloads. The gap between GDELT
    searches is handled inside search_news, since that is where the rate
    limit actually bites.

    min_docs : warn loudly if the topic comes back this thin. A topic with
               three documents cannot produce a good question, and it is
               better to know now than to find out at the assembly stage.
    """
    docs = []
    search_failures = 0

    # on-story documents
    hits = search_news(BASE_QUERIES[asset], date, max_records=n_relevant * 2)
    if not hits:
        search_failures += 1
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
        if not hits:
            search_failures += 1
        for h in hits[:per_query]:
            entry = {**h, "role": "distractor"}
            if fetch_text:
                entry["content"] = extract_text(h["url"])
                time.sleep(polite_delay)
            docs.append(entry)

    kept = [d for d in docs if not fetch_text or d.get("content")]
    n_rel = sum(1 for d in kept if d["role"] == "relevant")

    if verbose:
        if search_failures:
            print(f"  warning: {search_failures} of 4 searches returned nothing")
        if len(kept) < min_docs:
            print(f"  warning: only {len(kept)} documents for {asset} {date}, "
                  f"too thin to build a good question")
        elif n_rel < 5:
            print(f"  warning: only {n_rel} on-story documents "
                  f"(distractors do not carry the answer)")

    return {
        "asset": asset,
        "event_date": date,
        "docs": kept,
        "n_requested": len(docs),
        "n_with_text": len(kept),
        "n_relevant": n_rel,
        "n_distractor": len(kept) - n_rel,
        "search_failures": search_failures,
        "thin": len(kept) < min_docs,
    }
