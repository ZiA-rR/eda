"""
assembly.py
-----------
Stage 5: turn scored candidates into four-option questions.

Two things drive the design here.

First, the distractors are stratified, following AER. Rather than picking
wrong answers at random, each one is a specific type:

    temporal   - happened after the target event, so it is a consequence
    semantic   - shares entities or subject matter but is not a cause
    background - a genuine long-run condition, scored too low to count

Second, the assembly actively resists the two shortcuts found in AER.

    Style leakage. In the EDA, a classifier reading only the option text,
    with no question and no documents, still scored 89.4% against a 60.6%
    baseline. So correct and incorrect options must not differ in length or
    phrasing. This module balances lengths and reports the gap.

    Structural leakage. The winning AER system gained 5.6 points from
    post-hoc rules alone, because the "none of the others" option is
    correct every time it appears there, and duplicate options always share
    a truth value. Here the none option is sometimes wrong, and duplicates
    are merged rather than repeated.
"""

import datetime as dt
import random
import re
import statistics
from typing import List, Dict, Optional

NONE_TEXT = "None of the other options are correct causes."


# --------------------------------------------------------- classification
def classify_distractor(cand: Dict, target_entities: set) -> str:
    """
    Work out which of the three distractor types a candidate is.
    """
    if cand.get("position") == "after":
        return "temporal"

    score = cand.get("consensus")
    if score == 1:
        return "background"

    toks = set(re.findall(r"\w+", cand["text"].lower()))
    if target_entities and len(toks & target_entities) >= 2:
        return "semantic"

    return "unrelated"


def _content_words(text: str) -> set:
    STOP = {"the","a","an","of","to","in","on","and","for","by","at","as",
            "its","it","was","were","is","are","that","this","with","from",
            "after","before","has","had","have","been","be","which","their"}
    return {w for w in re.findall(r"\w+", text.lower())
            if w not in STOP and len(w) > 2}


# ------------------------------------------------------------- balancing
def length_gap(options: List[str], correct_idx: List[int]) -> float:
    """
    Mean word-count gap between correct and incorrect options. This is the
    single most exploitable style signal, so it gets measured on every item.
    """
    correct = [len(options[i].split()) for i in correct_idx]
    wrong = [len(o.split()) for i, o in enumerate(options)
             if i not in correct_idx and NONE_TEXT not in o]
    if not correct or not wrong:
        return 0.0
    return abs(statistics.mean(correct) - statistics.mean(wrong))


def _pick_balanced(correct: List[Dict], distractors: List[Dict],
                   n_options: int, max_gap: float) -> Optional[List[Dict]]:
    """
    Choose distractors whose lengths sit close to the correct options.

    Tries the closest-matching combination rather than a random one. If the
    best available gap is still too wide, gives up and returns None so the
    caller can drop the item rather than ship a leaky one.
    """
    n_needed = n_options - len(correct)
    if n_needed <= 0 or len(distractors) < n_needed:
        return None

    target_len = statistics.mean(len(c["text"].split()) for c in correct)

    # sort distractors by how close they are to the correct-option length
    ranked = sorted(distractors,
                    key=lambda d: abs(len(d["text"].split()) - target_len))

    # keep type variety: take the closest from each type first, then fill
    chosen, seen_types = [], set()
    for d in ranked:
        t = d.get("distractor_type", "unrelated")
        if t not in seen_types and len(chosen) < n_needed:
            chosen.append(d)
            seen_types.add(t)
    for d in ranked:
        if len(chosen) >= n_needed:
            break
        if d not in chosen:
            chosen.append(d)

    opts = [c["text"] for c in correct] + [d["text"] for d in chosen]
    gap = length_gap(opts, list(range(len(correct))))
    if gap > max_gap:
        return None

    return chosen



# --------------------------------------------------- cross-topic pool
def build_distractor_pool(topics: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Collect candidates per asset so a question can draw distractors from
    other events of the same asset.

    A real cause of a different gold move is topically convincing but is
    definitively not a cause of this one. That is AER's semantic distractor
    category, and it costs nothing.

    Without this, extraction has to supply all four options for every
    question, and roughly half of topics do not yield enough.
    """
    pool: Dict[str, List[Dict]] = {}
    for t in topics:
        asset = t.get("asset", "unknown")
        for c in t.get("candidates", []):
            pool.setdefault(asset, []).append({
                **c,
                "from_topic": f"{asset}_{t.get('event_date')}",
                "from_date": t.get("event_date"),
            })
    return pool


def _date_leaks(text: str, event_date: str, months: int = 3) -> bool:
    """
    Does this candidate name a date far from the event.

    A distractor dated September for a July question is rejectable without
    any reasoning at all, which is exactly the kind of shortcut the whole
    design is meant to avoid.
    """
    if not event_date:
        return False
    try:
        ev = dt.datetime.strptime(event_date, "%Y-%m-%d").date()
    except ValueError:
        return False

    # explicit years that are not the event's year
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    full_years = re.findall(r"\b((?:19|20)\d{2})\b", text)
    for y in full_years:
        if abs(int(y) - ev.year) >= 1:
            return True

    # ISO dates
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if abs((d - ev).days) > months * 31:
            return True

    # "March 7", "September 3" style, checked against the event month
    MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
              "july":7,"august":8,"september":9,"october":10,"november":11,
              "december":12}
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", text, re.I):
            gap = min(abs(num - ev.month), 12 - abs(num - ev.month))
            if gap > months:
                return True
    return False


def _cross_topic_distractors(topic: Dict,
                             pool: Dict[str, List[Dict]],
                             n_needed: int,
                             target_len: float,
                             rng: random.Random) -> List[Dict]:
    """
    Pick candidates from other events of the same asset, preferring ones
    whose length is close to the correct options so the length gap stays
    small.
    """
    asset = topic.get("asset", "unknown")
    this_topic = f"{asset}_{topic.get('event_date')}"

    others = [c for c in pool.get(asset, [])
              if c.get("from_topic") != this_topic]
    if not others:
        return []

    # never reuse text that already appears in this topic
    own = {c["text"].strip().lower() for c in topic.get("candidates", [])}
    others = [c for c in others if c["text"].strip().lower() not in own]

    event_date = topic.get("event_date", "")

    # prefer distractors from events close in time, so the borrowed option
    # is plausible for this date rather than obviously from another year
    def _months_apart(c):
        try:
            a_d = dt.datetime.strptime(event_date, "%Y-%m-%d").date()
            b_d = dt.datetime.strptime(c.get("from_date", ""), "%Y-%m-%d").date()
            return abs((a_d - b_d).days) / 30.0
        except (ValueError, TypeError):
            return 999.0

    # drop any whose own text gives the date away
    others = [c for c in others if not _date_leaks(c["text"], event_date)]

    near = [c for c in others if _months_apart(c) <= 6]
    if len(near) >= n_needed:
        others = near

    # closest length first, then a little randomness so questions differ
    others.sort(key=lambda d: abs(len(d["text"].split()) - target_len))
    shortlist = others[:max(n_needed * 4, 12)]
    rng.shuffle(shortlist)

    out = []
    seen = set()
    for d in shortlist:
        key = d["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({**d, "consensus": 0, "distractor_type": "semantic_crosstopic"})
        if len(out) >= n_needed:
            break
    return out


# ------------------------------------------------------------- assembly
def build_question(topic: Dict,
                   n_options: int = 4,
                   max_length_gap: float = 4.0,
                   none_option_prob: float = 0.15,
                   pool: Dict[str, List[Dict]] = None,
                   rng: random.Random = None) -> Optional[Dict]:
    """
    Build one question from a scored topic.

    Returns None if a sound question cannot be made, which is the right
    outcome when there are not enough good candidates or the length gap
    cannot be closed. Dropping an item is much cheaper than shipping one
    that leaks.
    """
    rng = rng or random.Random()
    cands = topic.get("candidates", [])

    # A candidate can have consensus None when every model failed to score
    # it. Treat unscored as "not usable as a correct answer" but still fine
    # as a distractor, rather than letting None reach a comparison.
    def _score(c):
        v = c.get("consensus")
        return v if isinstance(v, int) else None

    correct = [c for c in cands if _score(c) is not None and _score(c) >= 2]
    weak = [c for c in cands if _score(c) == 1]
    none_cands = [c for c in cands if _score(c) == 0 or _score(c) is None]

    if not correct:
        return None

    # AER runs about 56% single, 30% double, 14% triple. Cap at 3 so at
    # least one distractor always remains.
    if len(correct) > 3:
        correct = sorted(correct, key=lambda c: -(_score(c) or 0))[:3]

    target_entities = _content_words(topic.get("target_event", ""))
    local_pool = []
    for c in weak + none_cands:
        c = dict(c)
        c["distractor_type"] = classify_distractor(c, target_entities)
        local_pool.append(c)

    # top up from other events of the same asset when this topic did not
    # yield enough of its own
    n_needed = n_options - len(correct)
    if pool and len(local_pool) < n_needed:
        target_len = statistics.mean(len(c["text"].split()) for c in correct)
        local_pool += _cross_topic_distractors(
            topic, pool, n_needed - len(local_pool), target_len, rng)

    # a none-option question: all four options wrong except the none itself
    use_none_as_answer = rng.random() < none_option_prob and len(local_pool) >= 3

    if use_none_as_answer:
        chosen = local_pool[:3] if len(local_pool) >= 3 else None
        if not chosen:
            return None
        entries = [{"text": d["text"], "correct": False,
                    "score": d.get("consensus", 0),
                    "type": d.get("distractor_type")} for d in chosen]
        entries.append({"text": NONE_TEXT, "correct": True, "score": 3,
                        "type": "none_option"})
    else:
        chosen = _pick_balanced(correct, local_pool, n_options, max_length_gap)
        if chosen is None:
            return None
        entries = [{"text": c["text"], "correct": True,
                    "score": c.get("consensus"), "type": "correct"}
                   for c in correct]
        entries += [{"text": d["text"], "correct": False,
                     "score": d.get("consensus", 0),
                     "type": d.get("distractor_type")} for d in chosen]

        # sometimes include the none option as a genuine distractor, which
        # is what stops the "none means correct" shortcut from working
        if rng.random() < none_option_prob and len(entries) == n_options:
            replace_at = next((i for i, e in enumerate(entries)
                               if not e["correct"]), None)
            if replace_at is not None:
                entries[replace_at] = {"text": NONE_TEXT, "correct": False,
                                       "score": 0, "type": "none_option"}

    # drop exact duplicates rather than repeating text the way AER does
    seen, deduped = set(), []
    for e in entries:
        key = e["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    if len(deduped) < n_options:
        return None
    entries = deduped[:n_options]

    rng.shuffle(entries)

    letters = "ABCD"[:n_options]
    q = {
        "topic_id": topic.get("topic_id"),
        "asset": topic.get("asset"),
        "event_date": topic.get("event_date"),
        "target_event": topic.get("target_event"),
    }
    strengths, golden, types = {}, [], {}
    for L, e in zip(letters, entries):
        q[f"option_{L}"] = e["text"]
        strengths[L] = e["score"]
        types[L] = e["type"]
        if e["correct"]:
            golden.append(L)

    q["causal_strength"] = strengths
    q["golden_answer"] = ",".join(sorted(golden))
    q["option_types"] = types
    q["length_gap"] = round(length_gap([e["text"] for e in entries],
                                       [i for i, e in enumerate(entries)
                                        if e["correct"]]), 2)
    return q


def rebalance_positions(questions: List[Dict], rng: random.Random = None) -> List[Dict]:
    """
    Make sure correct answers are spread evenly across A, B, C and D.

    AER balanced this and my co-occurrence check confirmed no letter bias
    there. In my hand-built sample option D was never correct, which is
    exactly the kind of pattern a model can learn instead of reasoning, so
    this pass exists to catch it.
    """
    rng = rng or random.Random()
    letters = ["A", "B", "C", "D"]

    for q in questions:
        perm = letters[:]
        rng.shuffle(perm)

        opts = {L: q[f"option_{L}"] for L in letters}
        strengths = q["causal_strength"]
        types = q["option_types"]
        golden = set(q["golden_answer"].split(","))

        # rebuild cleanly rather than mutating in place, otherwise a letter
        # that has already been overwritten gets read back later in the loop
        new_opts, new_str, new_types, new_gold = {}, {}, {}, []
        for old, new in zip(letters, perm):
            new_opts[new] = opts[old]
            new_str[new] = q["causal_strength"][old]
            new_types[new] = types[old]
            if old in golden:
                new_gold.append(new)

        for L in letters:
            q[f"option_{L}"] = new_opts[L]
        q["causal_strength"] = new_str
        q["option_types"] = new_types
        q["golden_answer"] = ",".join(sorted(new_gold))

    return questions


def verify_consistency(questions: List[Dict]) -> Dict:
    """
    The golden answer must always be exactly what the ratings imply. If
    these ever drift apart the dataset is silently wrong, so this runs on
    every batch.
    """
    bad = []
    for q in questions:
        derived = {L for L, s in q["causal_strength"].items()
                   if isinstance(s, int) and s >= 2}   # None is excluded
        # the none option is correct despite carrying a score of 3 by design
        none_letters = {L for L, t in q.get("option_types", {}).items()
                        if t == "none_option"}
        stated = set(q["golden_answer"].split(",")) - {""}

        if stated - none_letters != derived - none_letters:
            bad.append({"id": q.get("id", q.get("event_date")),
                        "derived": sorted(derived), "stated": sorted(stated)})

    return {"n_checked": len(questions), "n_bad": len(bad), "problems": bad}
