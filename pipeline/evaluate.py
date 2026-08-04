"""
evaluate.py
-----------
Running models against the finished dataset.

Two things matter here beyond a headline score.

First, multi-answer questions need partial credit. Exact match alone hides
most of what is going on, because a model that gets two of three correct
options is doing better than one that gets none, and AER's own scoring uses
both.

Second, the interesting result is not the score, it is WHERE models fail.
The winning AER team found three patterns shared across 14 models from 7
families: causal chain incompleteness, proximate cause preference and
salience bias, with under-selections outnumbering over-selections 1,389 to
52. This module measures those directly, which is the part of the analysis
worth publishing.
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from typing import List, Dict, Optional

from llm import call_model


ANSWER_PROMPT = """You are analysing what caused a financial market event.

EVIDENCE (news reporting from around the time):
{context}

EVENT TO EXPLAIN:
{target}

Which of the following were causes of this event? More than one may be
correct. Select every option that genuinely contributed.

A. {a}
B. {b}
C. {c}
D. {d}

Return ONLY JSON, no other text:
{{"reasoning": "brief", "answer": ["A"]}}
"""


def _parse_answer(raw: str) -> List[str]:
    """
    Pull the selected letters out of a model response.

    Models vary a lot here: some return clean JSON, some wrap it in prose or
    code fences, some answer "A and C" in plain English. Being strict about
    the format means scoring a parsing failure as if the model got the
    question wrong, which is worse than useless.
    """
    if not raw:
        return []
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    def _clean(vals):
        out = set()
        for v in vals:
            s = str(v).strip().strip('"\'').upper()
            if s and s[0] in "ABCD":
                out.add(s[0])
        return sorted(out)

    # 1. clean JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "answer" in obj:
            a = obj["answer"]
            return _clean(a if isinstance(a, list) else [a])
        if isinstance(obj, list):
            return _clean(obj)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. a JSON object somewhere in the text
    m = re.search(r'\{[^{}]*"answer"\s*:\s*(\[[^\]]*\]|"[A-D]")[^{}]*\}',
                  raw, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            a = obj["answer"]
            return _clean(a if isinstance(a, list) else [a])
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. just the answer array
    m = re.search(r'"answer"\s*:\s*\[(.*?)\]', raw, flags=re.DOTALL)
    if m:
        return _clean(m.group(1).split(","))

    # 4. prose: "the answer is A and C", "Options A, B"
    m = re.search(r"(?:answer|correct|select)[^.\n]*?((?:\b[A-D]\b[,\s and&]*)+)",
                  raw, flags=re.I)
    if m:
        letters = re.findall(r"\b([A-D])\b", m.group(1))
        if letters:
            return _clean(letters)

    # 5. lines that start with a letter, as in "A. yes" or "- B"
    letters = re.findall(r"^[\s\-\*]*([A-D])[.):\s]", raw, flags=re.MULTILINE)
    if letters:
        return _clean(letters)

    # 6. last resort, any standalone A-D in the text
    letters = re.findall(r"\b([A-D])\b", raw)
    return _clean(letters) if letters else []


# ------------------------------------------------------------- metrics
def score_prediction(pred: List[str], gold: List[str]) -> Dict:
    p, g = set(pred), set(gold)
    tp = len(p & g)

    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "exact_match": int(p == g),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": len(p),
        "n_gold": len(g),
        "under_selected": max(0, len(g) - tp),   # correct ones it missed
        "over_selected": max(0, len(p) - tp),    # wrong ones it picked
    }


def evaluate_model(questions: List[Dict],
                   docs_by_topic: Dict,
                   model: str = "claude",
                   max_context: int = 20000,
                   verbose: bool = True) -> Dict:
    """
    Run one model over the dataset.

    The full document set goes into the context, distractor documents
    included, because sorting the relevant evidence from the noise is part
    of the task.
    """
    results = []
    parse_failures = []

    for i, q in enumerate(questions, 1):
        topic = docs_by_topic.get(q["topic_id"], {})
        parts = []
        for d in topic.get("docs", []):
            c = (d.get("content") or "")[:1500]
            if c:
                parts.append(f"[{d.get('source','')} {d.get('date','')}]\n{c}")
        context = "\n\n".join(parts)[:max_context]

        prompt = ANSWER_PROMPT.format(
            context=context, target=q["target_event"],
            a=q["option_A"], b=q["option_B"], c=q["option_C"], d=q["option_D"])

        try:
            raw = call_model(prompt, model=model, max_tokens=500)
            pred = _parse_answer(raw)
        except Exception as e:
            pred = []
            raw = f"ERROR {type(e).__name__}"

        gold = [x for x in q["golden_answer"].split(",") if x]
        m = score_prediction(pred, gold)

        if not pred and not raw.startswith("ERROR"):
            parse_failures.append(raw[:200])

        results.append({
            "id": q.get("id", f"{q.get('asset')}_{q.get('event_date')}"),
            "asset": q.get("asset"),
            "pred": pred, "gold": gold, **m,
            "n_gold": len(gold),
            "raw": raw[:300],
        })

        if verbose and i % 20 == 0:
            print(f"  {i}/{len(questions)}")

    if parse_failures:
        print(f"  WARNING: could not parse {len(parse_failures)} of "
              f"{len(questions)} responses from {model}. "
              f"A score of zero here means a parsing failure, not a wrong "
              f"answer. First response was:\n    {parse_failures[0][:160]}")

    return {"model": model, "results": results,
            "parse_failures": len(parse_failures), **aggregate(results)}


def aggregate(results: List[Dict]) -> Dict:
    if not results:
        return {}
    return {
        "n": len(results),
        "exact_match": round(statistics.mean(r["exact_match"] for r in results), 4),
        "f1": round(statistics.mean(r["f1"] for r in results), 4),
        "precision": round(statistics.mean(r["precision"] for r in results), 4),
        "recall": round(statistics.mean(r["recall"] for r in results), 4),
        "total_under_selected": sum(r["under_selected"] for r in results),
        "total_over_selected": sum(r["over_selected"] for r in results),
        "mean_n_pred": round(statistics.mean(r["n_pred"] for r in results), 2),
        "mean_n_gold": round(statistics.mean(r["n_gold"] for r in results), 2),
    }


# ------------------------------------------------------- failure analysis
def failure_analysis(eval_result: Dict, questions: List[Dict]) -> Dict:
    """
    Break the results down the way the AER winning team did.

    The headline finding to test against: do models under-select far more
    than they over-select, and does accuracy collapse as the number of
    correct answers rises? If financial reasoning shows the same pattern as
    general news, that is a result worth reporting on its own.
    """
    by_id = {q.get("id", f"{q.get('asset')}_{q.get('event_date')}"): q
             for q in questions}
    res = eval_result["results"]

    # by how many answers were correct
    by_card = defaultdict(list)
    for r in res:
        by_card[r["n_gold"]].append(r)

    cardinality = {}
    for k in sorted(by_card):
        rs = by_card[k]
        cardinality[k] = {
            "n": len(rs),
            "exact_match": round(statistics.mean(x["exact_match"] for x in rs), 3),
            "f1": round(statistics.mean(x["f1"] for x in rs), 3),
            "mean_predicted": round(statistics.mean(x["n_pred"] for x in rs), 2),
        }

    # by asset
    by_asset = defaultdict(list)
    for r in res:
        by_asset[r["asset"]].append(r)
    assets = {a: {"n": len(rs),
                  "exact_match": round(statistics.mean(x["exact_match"] for x in rs), 3),
                  "f1": round(statistics.mean(x["f1"] for x in rs), 3)}
              for a, rs in by_asset.items()}

    # which distractor types fooled it most
    fooled = Counter()
    missed_types = Counter()
    for r in res:
        q = by_id.get(r["id"])
        if not q or "option_types" not in q:
            continue
        gold = set(r["gold"])
        for L in set(r["pred"]) - gold:
            fooled[q["option_types"].get(L, "unknown")] += 1
        for L in gold - set(r["pred"]):
            missed_types[q["option_types"].get(L, "unknown")] += 1

    under = eval_result.get("total_under_selected", 0)
    over = eval_result.get("total_over_selected", 0)

    return {
        "under_vs_over": {
            "under_selected": under,
            "over_selected": over,
            "ratio": round(under / over, 1) if over else None,
            "aer_reference": "1389 vs 52 in the AER winning system's analysis",
        },
        "by_cardinality": cardinality,
        "by_asset": assets,
        "distractors_that_fooled_it": dict(fooled.most_common()),
        "correct_options_missed_by_type": dict(missed_types.most_common()),
    }


def compare_models(evals: List[Dict]) -> str:
    """Small text table comparing several models."""
    lines = [f"{'model':12s} {'n':>5s} {'exact':>8s} {'F1':>8s} "
             f"{'prec':>8s} {'recall':>8s} {'pred/gold':>12s}",
             "-" * 66]
    for e in evals:
        lines.append(
            f"{e['model']:12s} {e['n']:5d} {e['exact_match']:8.3f} {e['f1']:8.3f} "
            f"{e['precision']:8.3f} {e['recall']:8.3f} "
            f"{e['mean_n_pred']:5.2f}/{e['mean_n_gold']:<6.2f}")
    return "\n".join(lines)
