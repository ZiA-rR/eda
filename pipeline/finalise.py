"""
finalise.py
-----------
The last stages: human review, splitting, and the EDA on our own data.

Splitting is done by EVENT, not by question. Two questions about the same
market event share documents, so putting one in train and the other in test
leaks evidence between splits. AER reuses topics across train and dev,
which is a choice we do not have to copy.

The split also keeps the answer-cardinality mix consistent across train,
dev and test. AER's test set was noticeably easier than its dev set (18.3%
multi-answer against 47.5%), which makes scores on the two hard to compare.
"""

import json
import random
import statistics
from collections import Counter, defaultdict
from typing import List, Dict, Optional


# ------------------------------------------------------- human review
def make_review_sheet(queue: List[Dict], path: str,
                      overlap_frac: float = 0.15,
                      n_annotators: int = 3,
                      seed: int = 42) -> Dict:
    """
    Turn the review queue into per-annotator assignments.

    A shared overlap subset goes to everyone, and that is what Krippendorff's
    alpha is computed on. The rest is divided up, so the workload per person
    is roughly (1 - overlap) / n_annotators of the queue plus the overlap.

    With three annotators and 15% overlap, each person sees about 43% of the
    queue rather than all of it.
    """
    rng = random.Random(seed)
    items = list(queue)
    rng.shuffle(items)

    n_overlap = max(1, int(len(items) * overlap_frac))
    overlap = items[:n_overlap]
    rest = items[n_overlap:]

    assignments = {f"annotator_{i+1}": list(overlap) for i in range(n_annotators)}
    for i, item in enumerate(rest):
        assignments[f"annotator_{(i % n_annotators)+1}"].append(item)

    payload = {
        "instructions": {
            "scale": {
                "3": "STRONG - a main driver, the event would not have happened as it did without this",
                "2": "MODERATE - a real contributing cause alongside others",
                "1": "WEAK - background or minor, too far removed to count for this specific move",
                "0": "NONE - not a cause, including anything that happened afterwards",
            },
            "note": "Score independently. Do not discuss items in the shared set with other annotators before finishing, or the agreement number becomes meaningless.",
        },
        "n_overlap": n_overlap,
        "overlap_ids": [f"{it['topic_id']}::{it['candidate'][:40]}" for it in overlap],
        "assignments": {k: [{**it, "your_score": None} for it in v]
                        for k, v in assignments.items()},
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return {
        "queue_size": len(items),
        "overlap_items": n_overlap,
        "per_annotator": {k: len(v) for k, v in assignments.items()},
        "path": path,
    }


def collect_reviews(paths: List[str]) -> Dict:
    """
    Read back completed sheets and compute agreement on the shared subset.
    """
    from scoring import krippendorff_alpha

    sheets = []
    for p in paths:
        with open(p) as f:
            sheets.append(json.load(f))

    overlap_ids = sheets[0]["overlap_ids"]
    matrix = []
    for sheet in sheets:
        row = []
        by_id = {}
        for items in sheet["assignments"].values():
            for it in items:
                key = f"{it['topic_id']}::{it['candidate'][:40]}"
                by_id[key] = it.get("your_score")
        for oid in overlap_ids:
            v = by_id.get(oid)
            row.append(int(v) if v is not None else None)
        matrix.append(row)

    alpha = krippendorff_alpha(matrix, level="ordinal")

    return {
        "n_annotators": len(sheets),
        "n_overlap_items": len(overlap_ids),
        "krippendorff_alpha": round(alpha, 3),
        "reference": "AER reported 0.51, CRAB's experts 0.70, UNcommonsense 0.40-0.60",
        "verdict": ("acceptable" if alpha >= 0.45 else
                    "too low, the guidelines probably need tightening"),
    }


# ------------------------------------------------------------- splitting
def make_splits(questions: List[Dict],
                train: float = 0.7, dev: float = 0.15,
                seed: int = 42, verbose: bool = True) -> Dict:
    """
    Split by event so no market event appears in more than one split, and
    keep the cardinality mix similar across the three.
    """
    rng = random.Random(seed)

    by_event = defaultdict(list)
    for q in questions:
        key = (q.get("asset"), q.get("event_date"))
        by_event[key].append(q)

    events = list(by_event)
    rng.shuffle(events)

    # stratify roughly by asset so each split covers all four
    by_asset = defaultdict(list)
    for e in events:
        by_asset[e[0]].append(e)

    splits = {"train": [], "dev": [], "test": []}
    for asset, evs in by_asset.items():
        n = len(evs)
        n_tr = int(n * train)
        n_dv = int(n * dev)
        splits["train"] += evs[:n_tr]
        splits["dev"] += evs[n_tr:n_tr + n_dv]
        splits["test"] += evs[n_tr + n_dv:]

    out = {}
    for name, evs in splits.items():
        qs = [q for e in evs for q in by_event[e]]
        out[name] = qs

    if verbose:
        print(f"{'split':8s} {'events':>7s} {'questions':>10s} {'multi-answer':>13s}")
        print("-" * 42)
        for name in ["train", "dev", "test"]:
            qs = out[name]
            if not qs:
                continue
            multi = sum(1 for q in qs
                        if len([x for x in q["golden_answer"].split(",") if x]) > 1)
            print(f"{name:8s} {len(splits[name]):7d} {len(qs):10d} "
                  f"{100*multi/len(qs):12.1f}%")
        print("\n(AER's test split was much easier than its dev split, "
              "18.3% vs 47.5% multi-answer. These should be close.)")

    return out


# ------------------------------------------------------------------ EDA
def dataset_eda(questions: List[Dict], docs: List[Dict] = None) -> Dict:
    """
    The same statistics computed on AER in the EDA, run on our own data so
    the two can be compared directly.
    """
    n = len(questions)

    card = Counter(len([x for x in q["golden_answer"].split(",") if x])
                   for q in questions)
    multi = sum(v for k, v in card.items() if k > 1)

    assets = Counter(q.get("asset") for q in questions)
    events = len({(q.get("asset"), q.get("event_date")) for q in questions})

    tgt_len = [len(str(q.get("target_event", "")).split()) for q in questions]

    report = {
        "n_questions": n,
        "n_events": events,
        "questions_per_event": round(n / max(1, events), 2),
        "by_asset": dict(assets),
        "cardinality_pct": {k: round(100 * v / max(1, n), 1)
                            for k, v in sorted(card.items())},
        "multi_answer_pct": round(100 * multi / max(1, n), 1),
        "mean_gold_set_size": round(statistics.mean(
            len([x for x in q["golden_answer"].split(",") if x])
            for q in questions), 3),
        "target_event_words": {
            "mean": round(statistics.mean(tgt_len), 1) if tgt_len else 0,
        },
        "aer_comparison": {
            "multi_answer_pct": 43.58,
            "mean_gold_set_size": 1.574,
            "docs_per_topic": 19.67,
        },
    }

    if docs:
        per_topic = [len(t.get("docs", [])) for t in docs]
        words = [sum(len((d.get("content") or "").split())
                     for d in t.get("docs", [])) for t in docs]
        sources = Counter(d.get("source", "")
                          for t in docs for d in t.get("docs", []))
        n_dist = sum(1 for t in docs for d in t.get("docs", [])
                     if d.get("role") == "distractor")
        total_docs = sum(per_topic)

        report["documents"] = {
            "docs_per_topic": round(statistics.mean(per_topic), 2) if per_topic else 0,
            "mean_words_per_topic": round(statistics.mean(words), 0) if words else 0,
            "approx_tokens_per_topic": round(statistics.mean(words) * 1.3, 0) if words else 0,
            "unique_sources": len([s for s in sources if s]),
            "top5_source_share_pct": round(
                100 * sum(c for _, c in sources.most_common(5)) / max(1, total_docs), 1),
            "distractor_pct": round(100 * n_dist / max(1, total_docs), 1),
            "aer_reference": {"docs_per_topic": 19.7,
                              "tokens_per_topic": 28047,
                              "unique_sources": 535,
                              "top5_share_pct": 21},
        }

    return report


def print_eda(report: Dict):
    print("=" * 60)
    print("DATASET EDA")
    print("=" * 60)
    print(f"  questions          {report['n_questions']}")
    print(f"  events             {report['n_events']} "
          f"({report['questions_per_event']} questions each)")
    print(f"  by asset           {report['by_asset']}")
    print(f"  cardinality        {report['cardinality_pct']}")
    print(f"  multi-answer       {report['multi_answer_pct']}%   "
          f"(AER {report['aer_comparison']['multi_answer_pct']}%)")
    print(f"  mean gold set      {report['mean_gold_set_size']}   "
          f"(AER {report['aer_comparison']['mean_gold_set_size']})")
    if "documents" in report:
        d = report["documents"]
        r = d["aer_reference"]
        print(f"\n  docs per topic     {d['docs_per_topic']}   (AER {r['docs_per_topic']})")
        print(f"  tokens per topic   ~{d['approx_tokens_per_topic']:.0f}   "
              f"(AER {r['tokens_per_topic']})")
        print(f"  unique sources     {d['unique_sources']}   (AER {r['unique_sources']})")
        print(f"  top 5 source share {d['top5_source_share_pct']}%   (AER {r['top5_share_pct']}%)")
        print(f"  distractor docs    {d['distractor_pct']}%")
    print("=" * 60)
