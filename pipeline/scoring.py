"""
scoring.py
----------
Stage 4: rate how strongly each candidate caused the target event.

Every candidate is scored by three different models. Where they agree the
label is taken as settled. Where they disagree it goes to a human. This is
the approach the AER task paper used, and CRAB used a version of it too:
CRAB's crowd annotators only reached 0.28 agreement, but sending the
borderline cases to expert reviewers lifted agreement on those to 0.70.

The point is that we do not have to hand-label everything, only the part
that is genuinely uncertain, which is usually a quarter or so.

Scale is 0 to 3:
    3  strong    a main driver of the move
    2  moderate  a real contributing cause
    1  weak      only a small part of the reason
    0  none      not a cause

Anything 2 or above becomes a correct option. The reason for four coarse
levels rather than a 0-100 scale is that people agree far better on a few
buckets than on an exact number, which is what sank CRAB's agreement.
"""

import json
import re
import statistics
from typing import List, Dict, Optional

from llm import call_model, SCORING_MODELS


SCORING_PROMPT = """You are assessing how strongly one event caused another.

TARGET EVENT (the thing to be explained):
{target}

CANDIDATE CAUSE:
{candidate}

CONTEXT (news reporting from around that time):
{context}

Rate how strongly the candidate caused the target event:

3 = STRONG. A main driver. The target event would probably not have
    happened as it did without this.
2 = MODERATE. A real contributing cause, but not the main one. It pushed in
    the same direction alongside other factors.
1 = WEAK. Related background or a minor influence. Too far removed to call
    it a cause of this specific move.
0 = NONE. Not a cause. This includes events that happened AFTER the target
    event (those are consequences), and events that are merely topically
    related or moved in parallel for the same underlying reason.

Think about it carefully, then give your answer.

Return ONLY JSON in this form, no other text:
{{"reasoning": "one or two sentences", "score": 0}}
"""


def _parse_score(raw: str) -> Optional[Dict]:
    """Pull the JSON object out of a model response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "score" in obj:
            return {"score": int(obj["score"]),
                    "reasoning": str(obj.get("reasoning", ""))}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "score" in obj:
                return {"score": int(obj["score"]),
                        "reasoning": str(obj.get("reasoning", ""))}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # last resort: a bare digit 0-3
    m = re.search(r'"?score"?\s*[:=]\s*([0-3])', raw)
    if m:
        return {"score": int(m.group(1)), "reasoning": ""}

    return None


def build_context(topic: Dict, max_chars: int = 8000) -> str:
    """
    Assemble the evidence text the scorer sees.

    Only relevant documents go in. Distractor documents belong in the
    finished question so the model being evaluated has to sort through
    them, but they should not confuse our own labelling.
    """
    parts = []
    for d in topic["docs"]:
        if d.get("role") == "distractor":
            continue
        content = (d.get("content") or "")[:1500]
        if content:
            parts.append(f"[{d.get('source','')} {d.get('date','')}]\n{content}")
    return "\n\n".join(parts)[:max_chars]


def score_candidate(target: str,
                    candidate: str,
                    context: str,
                    models: List[str] = None) -> Dict:
    """
    Score one candidate with each model in turn.
    """
    models = models or SCORING_MODELS
    scores, reasoning, failures = {}, {}, []

    prompt = SCORING_PROMPT.format(target=target, candidate=candidate,
                                   context=context)

    for m in models:
        try:
            raw = call_model(prompt, model=m, max_tokens=300)
            parsed = _parse_score(raw)
            if parsed and 0 <= parsed["score"] <= 3:
                scores[m] = parsed["score"]
                reasoning[m] = parsed["reasoning"]
            else:
                failures.append(f"{m}: unparseable")
        except Exception as e:
            failures.append(f"{m}: {type(e).__name__}")

    return _summarise(scores, reasoning, failures)


def _summarise(scores: Dict[str, int], reasoning: Dict, failures: List) -> Dict:
    """
    Turn three model scores into a label plus a decision about whether a
    human needs to see it.

    Routing rules:
      - fewer than 2 usable scores      -> needs review (not enough signal)
      - all models agree exactly        -> settled
      - spread of 1 (e.g. 2,2,3)        -> settled, take the median
      - spread of 2 or more             -> needs review
      - median sits on the 1/2 boundary -> needs review, because that
        boundary is what decides whether the option is correct at all
    """
    vals = list(scores.values())

    if len(vals) < 2:
        return {"model_scores": scores, "model_reasoning": reasoning,
                "failures": failures, "consensus": None, "spread": None,
                "needs_review": True, "review_reason": "too few model scores"}

    spread = max(vals) - min(vals)
    median = statistics.median(vals)
    consensus = int(round(median))

    needs_review, reason = False, ""
    if spread >= 2:
        needs_review, reason = True, f"models disagree by {spread}"
    elif median in (1.5,):
        needs_review, reason = True, "sits on the include/exclude boundary"

    return {
        "model_scores": scores,
        "model_reasoning": reasoning,
        "failures": failures,
        "consensus": consensus,
        "spread": spread,
        "mean": round(statistics.mean(vals), 2),
        "needs_review": needs_review,
        "review_reason": reason,
    }


def score_topic(topic: Dict,
                models: List[str] = None,
                max_candidates: int = 40,
                verbose: bool = True) -> Dict:
    """
    Score every candidate for one topic.

    Candidates dated after the target event are given 0 without spending an
    API call: an event that happened afterwards cannot have caused it. They
    are kept, because they make good temporal distractors.
    """
    context = build_context(topic)
    target = topic.get("target_event") or topic.get("event_date")

    scored = []
    for i, cand in enumerate(topic["candidates"][:max_candidates], 1):

        if cand.get("position") == "after":
            scored.append({**cand, "consensus": 0, "spread": 0,
                           "model_scores": {}, "needs_review": False,
                           "review_reason": "",
                           "auto_zero": "dated after the target event"})
            continue

        result = score_candidate(target, cand["text"], context, models=models)
        scored.append({**cand, **result})

        if verbose and i % 5 == 0:
            print(f"    scored {i}/{min(len(topic['candidates']), max_candidates)}")

    if verbose:
        n_review = sum(1 for c in scored if c.get("needs_review"))
        auto = sum(1 for c in scored if c.get("auto_zero"))
        print(f"  {len(scored)} candidates, {auto} auto-zeroed as post-dated, "
              f"{n_review} need human review "
              f"({100*n_review/max(1,len(scored)):.0f}%)")

    return {**topic, "candidates": scored}


# ------------------------------------------------------------- agreement
def krippendorff_alpha(data: List[List[Optional[int]]],
                       level: str = "ordinal") -> float:
    """
    Krippendorff's alpha for annotator agreement.

    data : one list per annotator, same length, None where that annotator
           did not label that item.

    Ordinal is the right level for our 0-3 scale, since the categories are
    ordered and being one apart is a smaller disagreement than being three
    apart. Nominal would treat 0 vs 1 as just as wrong as 0 vs 3.

    For reference, AER reported 0.51 with three annotators, CRAB's experts
    reached 0.70, and UNcommonsense reported 0.40 to 0.60. Anything in the
    0.5 to 0.7 band is normal for this kind of judgement.
    """
    n_items = len(data[0])
    if any(len(d) != n_items for d in data):
        raise ValueError("every annotator needs the same number of slots")

    units = []
    for i in range(n_items):
        vals = [d[i] for d in data if d[i] is not None]
        if len(vals) >= 2:
            units.append(vals)

    if not units:
        return float("nan")

    all_vals = [v for u in units for v in u]
    if len(set(all_vals)) == 1:
        return 1.0    # everyone agreed on everything

    def delta(a, b):
        if level == "nominal":
            return 0.0 if a == b else 1.0
        return (a - b) ** 2      # ordinal / interval

    # observed disagreement
    num = 0.0
    den = 0.0
    for u in units:
        m = len(u)
        pair_sum = sum(delta(a, b) for i, a in enumerate(u)
                       for j, b in enumerate(u) if i != j)
        num += pair_sum / (m - 1)
        den += m
    Do = num / den

    # expected disagreement
    N = len(all_vals)
    De = sum(delta(a, b) for i, a in enumerate(all_vals)
             for j, b in enumerate(all_vals) if i != j) / (N * (N - 1))

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def review_queue(scored_topics: List[Dict]) -> List[Dict]:
    """
    Flatten everything that needs a human into one list, worst disagreement
    first so the most useful items get looked at even if time runs short.
    """
    queue = []
    for t in scored_topics:
        for c in t["candidates"]:
            if c.get("needs_review"):
                queue.append({
                    "topic_id": t.get("topic_id"),
                    "asset": t.get("asset"),
                    "event_date": t.get("event_date"),
                    "target_event": t.get("target_event"),
                    "candidate": c["text"],
                    "model_scores": c.get("model_scores", {}),
                    "spread": c.get("spread"),
                    "reason": c.get("review_reason", ""),
                })
    queue.sort(key=lambda x: -(x["spread"] or 0))
    return queue
