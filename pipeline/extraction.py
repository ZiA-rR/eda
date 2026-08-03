"""
extraction.py
-------------
Stage 3: pull candidate cause events out of the retrieved documents.

Approach follows CRAB, which prompts a model to extract the main events from
each article rather than using a syntactic event extractor. CRAB found the
generative approach gave better precision and, more importantly, produced
events at the right level of abstraction: a whole newsworthy happening
rather than a single verb phrase.

Each extracted event carries its source document and a date, because
temporal ordering is what lets us later tell causes from consequences.
VERSION = "2026-08-03-b"
"""

import json
import re
import datetime as dt
from typing import List, Dict, Optional

from llm import call_model     # provider-agnostic wrapper, see llm.py


EXTRACTION_PROMPT = """You are finding possible CAUSES of a specific market move.

THE MOVE TO BE EXPLAINED:
{target}

Read the article below and list events that could help explain why that move
happened. An event is something that occurred: a decision, an announcement, a
policy change, a failure, a disruption, a regulatory action, a large trade, a
statement by someone influential.

CRITICAL: do NOT list the price move itself, or any description of it. Those
are the thing being explained, not an explanation. Specifically exclude:
- any statement of what the price was, or how much it rose or fell
  ("bitcoin traded at $4,190", "the price gained 14 percent")
- trading volume, order book or liquidity statistics
- technical analysis (moving averages, support levels, chart patterns)
- descriptions of the move happening ("prices surged", "the rally continued")
- other assets moving in the same direction at the same time
- analyst opinion, forecasts, or predictions

DO list things like:
- a central bank decision or signal
- a bank failure, bankruptcy or default
- a regulatory or legal action
- a geopolitical event, conflict or sanction
- a supply disruption or production decision
- a large identifiable transaction or liquidation
- an economic data release
- a public statement by a government, company or major investor

Each event must be one self-contained sentence with the specific names,
figures and dates the article gives.

If the article contains no such events, return an empty list. An empty list is
the right answer more often than you might expect: many articles only report
the move itself.

Article published: {date}
Article source: {source}

ARTICLE:
{text}

Return ONLY a JSON list of strings, no other text. Example:
["California regulators closed Silicon Valley Bank on 10 March."]
"""


def _parse_json_list(raw: str) -> List[str]:
    """
    Models wrap JSON in prose or code fences more often than they should.
    Pull out the first JSON array we can find.
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass

    # fall back to the first [...] block
    m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    return []


def extract_events_from_doc(doc: Dict,
                            target_event: str = "",
                            model: str = "claude",
                            max_chars: int = 6000) -> List[Dict]:
    """
    Extract candidate events from one document.

    Returns a list of dicts, each carrying the event text plus provenance:
    which document it came from, which source, and the publication date.
    """
    text = (doc.get("content") or "").strip()
    if len(text) < 200:
        return []

    prompt = EXTRACTION_PROMPT.format(
        target=target_event or "an unusually large move in this asset's price",
        date=doc.get("date", "unknown"),
        source=doc.get("source", "unknown"),
        text=text[:max_chars],
    )

    raw = call_model(prompt, model=model, max_tokens=1000)
    events = _parse_json_list(raw)

    return [{
        "text": e,
        "doc_id": doc.get("url", doc.get("id", "")),
        "source": doc.get("source", ""),
        "doc_date": doc.get("date", ""),
        "doc_role": doc.get("role", "relevant"),
    } for e in events]


# --------------------------------------------------------------- filtering
def filter_events(events: List[Dict],
                  min_tokens: int = 4,
                  max_tokens: int = 60) -> List[Dict]:
    """
    Drop events that are too short to be meaningful or too long to be a
    single event. CRAB dropped anything under 3 tokens and had two experts
    check the rest.

    Also drops anything that looks like reported speech or opinion, which
    the prompt asks the model to avoid but which slips through anyway.
    """
    OPINION_MARKERS = [
        "analysts said", "analysts expect", "could see", "may rise",
        "is expected to", "predicts", "forecasts that", "warned that",
        "believes", "according to analysts",
    ]

    # Restatements of the move itself. The prompt asks the model to skip
    # these but some always get through, and they are worse than useless as
    # candidate causes because they are guaranteed to look related.
    PRICE_DESCRIPTION = [
        "moving average", "support level", "resistance", "chart",
        "trading volume", "was priced at", "traded at", "price rose",
        "price fell", "price spiked", "price surged", "price climbed",
        "price dropped", "gain of", "loss of", "percent in the last",
        "highest since", "lowest since", "market cap", "order book",
        "btc/usd", "trading session", "leveled out", "rallied to",
        "24 hours", "intraday",
        # movement verbs applied to the asset itself
        "spiked", "surged", "soared", "plunged", "tumbled", "slumped",
        "the surge", "the rally", "the selloff", "the plunge",
        "sudden rise", "sudden fall", "sharp rise", "sharp drop",
        "big revival", "enjoying the", "experienced a", "climbed to",
        "jumped to", "fell to", "rose to", "hit a high", "hit a low",
        "record high", "record low", "week high", "week low",
        "month high", "month low", "all-time high", "all-time low",
    ]

    kept = []
    for e in events:
        t = e["text"].strip()
        n = len(t.split())
        if n < min_tokens or n > max_tokens:
            continue
        low = t.lower()
        if any(m in low for m in OPINION_MARKERS):
            continue
        if any(m in low for m in PRICE_DESCRIPTION):
            continue
        kept.append(e)
    return kept


def deduplicate_events(events: List[Dict], threshold: float = 0.75) -> List[Dict]:
    """
    The same event gets reported by many outlets in different words. Merge
    near-duplicates, keeping the first occurrence and recording how many
    documents mentioned it.

    Uses token Jaccard, which is crude but works well enough for headline
    events that share proper nouns and figures. A sentence embedding model
    would do better if one is available.

    The mention count is useful later: an event reported by many outlets is
    usually a more central part of the story.
    """
    def toks(s):
        return set(re.findall(r"\w+", s.lower()))

    merged: List[Dict] = []
    for e in events:
        te = toks(e["text"])
        hit = None
        for m in merged:
            tm = toks(m["text"])
            union = te | tm
            if union and len(te & tm) / len(union) >= threshold:
                hit = m
                break
        if hit:
            hit["n_mentions"] += 1
            hit["sources"].add(e["source"])
        else:
            e = dict(e)
            e["n_mentions"] = 1
            e["sources"] = {e["source"]}
            merged.append(e)

    for m in merged:
        m["sources"] = sorted(x for x in m["sources"] if x)
    return merged


# --------------------------------------------------------------- timeline
def _parse_date(s: str) -> Optional[dt.date]:
    """GDELT dates look like 20230313T120000Z. Also accept YYYY-MM-DD."""
    if not s:
        return None
    s = str(s)
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s[:len(fmt) + 2].rstrip("Z"), fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def build_timeline(events: List[Dict], event_date: str) -> List[Dict]:
    """
    Order events in time and mark each as before or after the target event.

    This is what makes temporal distractors possible. Anything dated after
    the target move cannot be a cause of it, so those become distractors
    rather than candidates.

    CRAB constructed timelines the same way and noted that this is harder
    across documents than within one, because articles publish on different
    dates and their timelines interleave.
    """
    target = _parse_date(event_date)
    out = []
    for e in events:
        d = _parse_date(e.get("doc_date", ""))
        e = dict(e)
        e["parsed_date"] = d.isoformat() if d else None
        if d and target:
            e["position"] = "before" if d <= target else "after"
        else:
            e["position"] = "unknown"
        out.append(e)

    out.sort(key=lambda x: (x["parsed_date"] is None, x["parsed_date"] or ""))
    return out


def extract_candidates_for_topic(topic: Dict,
                                 model: str = "claude",
                                 verbose: bool = True) -> Dict:
    """
    Run the whole of stage 3 for one topic (one big move day).
    """
    all_events = []
    for i, doc in enumerate(topic["docs"], 1):
        evs = extract_events_from_doc(
            doc, target_event=topic.get("target_event", ""), model=model)
        all_events.extend(evs)
        if verbose:
            print(f"    doc {i}/{len(topic['docs'])}: {len(evs)} events")

    filtered = filter_events(all_events)
    deduped = deduplicate_events(filtered)
    timeline = build_timeline(deduped, topic["event_date"])

    if verbose:
        n_before = sum(1 for e in timeline if e["position"] == "before")
        n_after = sum(1 for e in timeline if e["position"] == "after")
        print(f"  {len(all_events)} raw -> {len(filtered)} filtered -> "
              f"{len(deduped)} unique  ({n_before} before, {n_after} after)")

    return {**topic, "candidates": timeline}
