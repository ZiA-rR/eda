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

import logging
import os
import re
import time
import datetime as dt
from typing import List, Dict, Optional

import pandas as pd
import requests

# trafilatura logs an ERROR for every blocked or paywalled URL. Those
# failures are expected and routine, and the noise buries everything else.
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)


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
GDELT_MIN_INTERVAL = 12.0
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
                quality_only: bool = True,
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
    quality_only: keep only recognised financial and news outlets. Without
                  this the results fill up with price-listing pages from
                  local papers, which contain no explanation of anything.

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

        if quality_only:
            filtered = [a for a in out if is_quality_source(a["source"])]
            if verbose and out and not filtered:
                print(f"  none of {len(out)} results were from quality outlets")
            out = filtered

        return out

    print(f"  GDELT gave up on '{query}' after {retries} attempts")
    return []


BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")


def extract_text(url: str, timeout: int = 20, min_words: int = 120) -> Optional[str]:
    """
    Pull the main body text from an article URL.

        pip install trafilatura

    Fetches with requests using a browser user agent rather than letting
    trafilatura use its default, because many news sites return 403 to
    anything that looks automated.

    min_words drops stubs: cookie notices, "subscribe to continue" pages and
    paywall teasers extract fine but contain no article. A real news article
    is well over 120 words.

    Returns None on paywalls, blocks or extraction failure, which is common.
    Expect to lose a meaningful share of URLs whatever you do.
    """
    import trafilatura
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": BROWSER_UA,
                                     "Accept-Language": "en-US,en;q=0.9"})
        if resp.status_code != 200:
            return None

        text = trafilatura.extract(resp.text, include_comments=False,
                                   include_tables=False)
        if not text or len(text.split()) < min_words:
            return None
        return text
    except Exception:
        return None


# ------------------------------------------------------- query building
# Queries for the on-story documents, per asset.
# Queries have to be specific enough to return financial ANALYSIS rather
# than daily price-listing pages. A bare "gold price" query returns local
# retail listings ("Gold Rate in Pakistan Today") which carry no causal
# content and are useless for building questions.
# Query length matters and the right length depends on the backend.
# GDELT matches keywords literally, so it needs several terms to avoid
# returning price-listing pages. Google News ranks by relevance and a long
# query plus a narrow date window returns almost nothing, so short is
# better there.
BASE_QUERIES = {
    "gold":   "gold price",
    "crypto": "bitcoin price",
    "petrol": "oil prices",
    "forex":  "pound sterling dollar",
}

# Longer versions, used when falling back to GDELT.
BASE_QUERIES_GDELT = {
    "gold":   "gold prices rally investors safe haven Federal Reserve",
    "crypto": "bitcoin falls investors selloff market",
    "petrol": "oil prices crude supply OPEC market",
    "forex":  "sterling pound falls dollar markets Bank of England",
}

# Only keep articles from outlets that actually do financial journalism.
# GDELT indexes an enormous long tail of aggregators and local papers that
# republish price tables. Those pass a keyword filter but carry no
# explanation of why anything moved.
QUALITY_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "cnbc.com", "wsj.com",
    "marketwatch.com", "investing.com", "barrons.com", "economist.com",
    "apnews.com", "theguardian.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "forbes.com", "businessinsider.com", "axios.com",
    "cnn.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com", "npr.org",
    "telegraph.co.uk", "thetimes.co.uk", "independent.co.uk", "standard.co.uk",
    "aljazeera.com", "dw.com", "france24.com", "scmp.com", "japantimes.co.jp",
    "kitco.com", "mining.com", "oilprice.com", "rigzone.com", "spglobal.com",
    "coindesk.com", "cointelegraph.com", "theblock.co", "decrypt.co",
    "fxstreet.com", "dailyfx.com", "forexlive.com", "seekingalpha.com",
    "morningstar.com", "yahoo.com", "fortune.com", "time.com", "newsweek.com",
}


def is_quality_source(domain: str) -> bool:
    """Is this outlet one we want text from."""
    if not domain:
        return False
    d = domain.lower().replace("www.", "")
    return any(d == q or d.endswith("." + q) for q in QUALITY_DOMAINS)

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
    queries = BASE_QUERIES if serper_available() else BASE_QUERIES_GDELT
    hits = search(queries[asset], date, max_records=n_relevant * 4)
    if not hits:
        search_failures += 1
    for h in hits[:n_relevant]:
        entry = {**h, "role": "relevant"}
        if fetch_text:
            txt = extract_text(h["url"])
            if not txt and h.get("snippet"):
                # Reuters, Bloomberg, FT and others block scrapers. The
                # search snippet is only a sentence or two, but a headline
                # plus a sentence from Reuters still carries a real event,
                # and losing those outlets entirely is worse.
                txt = f"{h.get('title','')}. {h['snippet']}"
                entry["content_is_snippet"] = True
            entry["content"] = txt
            time.sleep(polite_delay)
        docs.append(entry)

    # distractor documents
    per_query = max(1, n_distractor // len(DISTRACTOR_QUERIES[asset]))
    for q in DISTRACTOR_QUERIES[asset]:
        hits = search(q, date, max_records=per_query * 4)
        if not hits:
            search_failures += 1
        for h in hits[:per_query]:
            entry = {**h, "role": "distractor"}
            if fetch_text:
                txt = extract_text(h["url"])
                if not txt and h.get("snippet"):
                    txt = f"{h.get('title','')}. {h['snippet']}"
                    entry["content_is_snippet"] = True
                entry["content"] = txt
                time.sleep(polite_delay)
            docs.append(entry)

    kept = [d for d in docs if not fetch_text or d.get("content")]
    n_rel = sum(1 for d in kept if d["role"] == "relevant")
    n_snip = sum(1 for d in kept if d.get("content_is_snippet"))

    if verbose:
        if search_failures:
            print(f"  warning: {search_failures} of 4 searches returned nothing")
        if len(kept) < min_docs:
            print(f"  warning: only {len(kept)} documents for {asset} {date}, "
                  f"too thin to build a good question")
        elif n_rel < 5:
            print(f"  warning: only {n_rel} on-story documents "
                  f"(distractors do not carry the answer)")
        if n_snip:
            print(f"  note: {n_snip} of {len(kept)} are snippet-only "
                  f"(site blocked full text)")

    return {
        "asset": asset,
        "event_date": date,
        "docs": kept,
        "n_requested": len(docs),
        "n_with_text": len(kept),
        "n_relevant": n_rel,
        "n_distractor": len(kept) - n_rel,
        "search_failures": search_failures,
        "n_snippet_only": n_snip,
        "thin": len(kept) < min_docs,
    }


# ------------------------------------------------------------ diagnostics
def check_gdelt(verbose: bool = True) -> bool:
    """
    Is GDELT actually reachable and not rate limiting us right now.

    Colab hands out shared IP addresses, so other people's requests count
    against the same limit. Some sessions are simply unusable, and it is
    better to find out in ten seconds than forty minutes into a run.
    """
    try:
        _gdelt_wait()
        resp = requests.get(GDELT_URL,
                            params={"query": "markets sourcelang:english",
                                    "mode": "artlist", "maxrecords": 5,
                                    "format": "json"},
                            timeout=20,
                            headers={"User-Agent": "research-dataset-builder"})
        if resp.status_code == 429:
            if verbose:
                print("GDELT is rate limiting this IP.")
                print("Colab shares IPs, so this is often not your fault.")
                print("Options: wait and retry, or Runtime > Disconnect and")
                print("delete runtime to get a new IP, then reconnect.")
            return False
        resp.raise_for_status()
        n = len(resp.json().get("articles", []))
        if verbose:
            print(f"GDELT is responding, {n} articles for a test query")
        return True
    except Exception as e:
        if verbose:
            print(f"GDELT check failed: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------- serper
# GDELT rate-limits per IP, and Colab hands out shared IPs, so on a bad day
# every request comes back 429 no matter how slowly you go. Serper is the
# fallback: it wraps Google News, gives 2,500 free queries without a card,
# and is what the AER team used.
#
#   1. sign up at serper.dev
#   2. copy the API key
#   3. os.environ["SERPER_API_KEY"] = "..."
#
# At four searches per topic, the free allowance covers about 600 topics.

SERPER_URL = "https://google.serper.dev/news"

VERSION = "2026-08-03-d"   # bump when editing, check with retrieval.VERSION



def _parse_serper_date(s: str, reference: dt.date) -> Optional[dt.date]:
    """
    Serper reports article dates inconsistently: sometimes "2 days ago",
    sometimes "Mar 13, 2023", sometimes nothing. Parse what we can.

    reference is kept for the signature but relative strings resolve
    against today, since that is what "2 hours ago" actually means.
    """
    if not s:
        return None
    s = str(s).strip().lower()

    m = re.match(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = {"minute": 0, "hour": 0, "day": 1,
                "week": 7, "month": 30, "year": 365}[unit] * n
        # Relative dates are relative to NOW, never to the date being
        # searched for. Resolving "2 hours ago" against the search anchor
        # makes every fresh article look like it was published on the day
        # we asked about, which is exactly backwards.
        return dt.date.today() - dt.timedelta(days=days)

    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%d %b %Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s.title(), fmt).date()
        except ValueError:
            continue
    return None


def serper_available() -> bool:
    return bool(os.environ.get("SERPER_API_KEY"))


def search_news_serper(query: str,
                       date: str,
                       days_before: int = 2,
                       days_after: int = 1,
                       max_records: int = 20,
                       english_only: bool = True,
                       quality_only: bool = True,
                       timeout: int = 30,
                       verbose: bool = True) -> List[Dict]:
    """
    Same contract as search_news, but via Serper's Google News endpoint.

    Date filtering uses Google's tbs parameter, which wants US-style
    m/d/yyyy dates.
    """
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY is not set")

    d = dt.datetime.strptime(date, "%Y-%m-%d")
    lo = d - dt.timedelta(days=days_before)
    hi = d + dt.timedelta(days=days_after)
    tbs = (f"cdr:1,cd_min:{lo.month}/{lo.day}/{lo.year},"
           f"cd_max:{hi.month}/{hi.day}/{hi.year}")

    payload = {"q": query, "num": min(max_records, 100), "tbs": tbs}
    if english_only:
        payload["gl"] = "us"
        payload["hl"] = "en"

    try:
        resp = requests.post(SERPER_URL, json=payload, timeout=timeout,
                             headers={"X-API-KEY": key,
                                      "Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if verbose:
            print(f"  serper failed for '{query}': {e}")
        return []

    # The news endpoint accepts tbs but does not reliably honour it, so
    # results outside the window get filtered out here as well.
    lo_d, hi_d = lo.date(), hi.date()

    out = []
    dropped_by_date = 0
    for a in data.get("news", []):
        link = a.get("link", "")

        art_date = _parse_serper_date(a.get("date", ""), d.date())
        if art_date and not (lo_d - dt.timedelta(days=1)
                             <= art_date
                             <= hi_d + dt.timedelta(days=1)):
            dropped_by_date += 1
            continue
        # serper reports the outlet name, not the domain, so take the
        # domain from the URL to keep the whitelist check working
        domain = ""
        try:
            from urllib.parse import urlparse
            domain = urlparse(link).netloc.replace("www.", "")
        except Exception:
            pass
        out.append({
            "title": a.get("title", ""),
            "url": link,
            "source": domain or a.get("source", ""),
            "date": date,          # serper gives relative dates, so use the
                                   # window we asked for
            "language": "eng",
            "snippet": a.get("snippet", ""),
            "reported_date": a.get("date", ""),
        })

    if verbose and dropped_by_date:
        print(f"  dropped {dropped_by_date} results outside the date window")

    if quality_only and out:
        filtered = [a for a in out if is_quality_source(a["source"])]
        if not filtered:
            # Better to keep unknown outlets than return nothing. The
            # extraction stage will drop anything without real content
            # anyway, and a thin topic is worse than a slightly noisy one.
            if verbose:
                doms = sorted({a["source"] for a in out})[:6]
                print(f"  no whitelisted outlets among {len(out)} results, "
                      f"keeping all. saw: {', '.join(doms)}")
        else:
            out = filtered

    return out


def search(query: str, date: str, **kw) -> List[Dict]:
    """
    Search using whichever backend is available.

    Serper first when a key is set, since it is more reliable and gives
    better results. GDELT otherwise.
    """
    if serper_available():
        return search_news_serper(query, date, **kw)
    return search_news(query, date, **kw)


def check_serper(verbose: bool = True) -> bool:
    """Is the serper key set and working."""
    if not serper_available():
        if verbose:
            print("SERPER_API_KEY is not set.")
            print("Sign up free at serper.dev, then:")
            print('  os.environ["SERPER_API_KEY"] = "your-key"')
        return False
    try:
        hits = search_news_serper("markets", "2023-03-13", quality_only=False,
                                  max_records=5, verbose=False)
        if verbose:
            print(f"serper is working, {len(hits)} results for a test query")
        return bool(hits)
    except Exception as e:
        if verbose:
            print(f"serper check failed: {e}")
        return False


def debug_search(asset: str, date: str, n: int = 20) -> None:
    """
    Show what the search actually returns, before and after filtering.

    Use this when a topic comes back thin: it tells you whether the problem
    is the query returning nothing, or the whitelist throwing everything
    away, or article text extraction failing.
    """
    backend = "serper" if serper_available() else "gdelt"
    queries = BASE_QUERIES if serper_available() else BASE_QUERIES_GDELT
    q = queries[asset]

    print(f"backend : {backend}")
    print(f"query   : {q!r}")
    print(f"date    : {date}\n")

    if backend == "serper":
        raw = search_news_serper(q, date, max_records=n, quality_only=False,
                                 verbose=False)
    else:
        raw = search_news(q, date, max_records=n, quality_only=False,
                          verbose=False)

    print(f"raw results: {len(raw)}")
    for a in raw[:12]:
        mark = "keep" if is_quality_source(a["source"]) else "drop"
        print(f"  [{mark}] {a['source']:28s} {a['title'][:52]}")

    kept = [a for a in raw if is_quality_source(a["source"])]
    print(f"\nafter whitelist: {len(kept)} of {len(raw)}")

    if kept:
        print("\ntesting text extraction on the first 3:")
        for a in kept[:3]:
            txt = extract_text(a["url"])
            if txt:
                status = f"{len(txt.split())} words (full)"
            elif a.get("snippet"):
                n = len(f"{a.get('title','')}. {a['snippet']}".split())
                status = f"{n} words (snippet fallback)"
            else:
                status = "FAILED, no snippet either"
            print(f"  {a['source']:28s} {status}")
