"""
quality.py
----------
Stage 6: the gates a batch has to pass before it counts as finished.

Two kinds of shortcut, both found in AER, both checked here.

STYLE LEAKAGE. In the EDA, a classifier reading only the option text, with
no question and no documents at all, scored 89.4% against a 60.6% baseline.
That means most of the answer sat in how the options were phrased rather
than in the reasoning. ART was cleaned until the equivalent number fell to
about 51%, which is chance.

STRUCTURAL LEAKAGE. The winning AER system gained 5.6 points from rules
applied after the model answered, with no reasoning involved, because the
"none of the others" option was correct every time it appeared and
duplicate options always shared a truth value.

Both gates run on every batch. The leakage classifier needs a few hundred
questions before its number means anything, so it reports low confidence
below that.
"""

import re
import statistics
from collections import Counter
from typing import List, Dict

import numpy as np


NONE_MARKER = "None of the other"


# --------------------------------------------------------- style leakage
def leakage_check(questions: List[Dict], seed: int = 42) -> Dict:
    """
    Train a classifier on option text alone and see how well it does.

    Each (question, option) pair is one example. Features are TF-IDF of the
    option text and nothing else. If this beats the majority-class baseline
    by much, the options are giving themselves away.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    rows = []
    for q in questions:
        golden = set(q["golden_answer"].split(",")) - {""}
        for L in "ABCD":
            key = f"option_{L}"
            if key not in q:
                continue
            rows.append({"text": str(q[key]), "label": 1 if L in golden else 0})

    if len(rows) < 40:
        return {"error": "not enough options to test", "n": len(rows)}

    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    if len(set(labels)) < 2:
        return {"error": "only one class present"}

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels)

    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_tr)
    acc = accuracy_score(y_te, clf.predict(Xte))
    baseline = max(np.mean(y_te), 1 - np.mean(y_te))

    n_q = len(questions)
    return {
        "accuracy": round(float(acc), 4),
        "baseline": round(float(baseline), 4),
        "gap_pp": round(float((acc - baseline) * 100), 1),
        "n_questions": n_q,
        "n_options": len(rows),
        "reliable": n_q >= 200,
        "note": ("" if n_q >= 200 else
                 f"only {n_q} questions, needs ~200+ before this number means much"),
    }


def length_balance(questions: List[Dict]) -> Dict:
    """
    Word-count gap between correct and incorrect options, the simplest and
    most exploitable style signal. The none option is excluded because it
    is a fixed short string by design.
    """
    correct, wrong = [], []
    for q in questions:
        golden = set(q["golden_answer"].split(",")) - {""}
        for L in "ABCD":
            key = f"option_{L}"
            if key not in q:
                continue
            txt = str(q[key])
            if NONE_MARKER in txt:
                continue
            (correct if L in golden else wrong).append(len(txt.split()))

    if not correct or not wrong:
        return {"error": "missing one class"}

    return {
        "correct_mean": round(statistics.mean(correct), 1),
        "wrong_mean": round(statistics.mean(wrong), 1),
        "gap": round(abs(statistics.mean(correct) - statistics.mean(wrong)), 2),
    }


def word_overlap(questions: List[Dict]) -> Dict:
    """
    Jaccard overlap between the target event and each option, split by
    whether the option is correct. In AER the gap was small (0.077 vs
    0.064), so this is a mild signal rather than a serious one, but it is
    cheap to keep watching.
    """
    def toks(s):
        return set(re.findall(r"\w+", str(s).lower()))

    c_vals, w_vals = [], []
    for q in questions:
        tgt = toks(q.get("target_event", ""))
        golden = set(q["golden_answer"].split(",")) - {""}
        for L in "ABCD":
            key = f"option_{L}"
            if key not in q:
                continue
            o = toks(q[key])
            union = tgt | o
            j = len(tgt & o) / len(union) if union else 0.0
            (c_vals if L in golden else w_vals).append(j)

    if not c_vals or not w_vals:
        return {"error": "missing one class"}
    return {
        "correct_mean": round(statistics.mean(c_vals), 4),
        "wrong_mean": round(statistics.mean(w_vals), 4),
        "gap": round(abs(statistics.mean(c_vals) - statistics.mean(w_vals)), 4),
    }


# ---------------------------------------------------- structural leakage
def structural_check(questions: List[Dict]) -> Dict:
    """
    Look for patterns a model could exploit without reading anything.

    Specifically the two the winning AER team turned into free points: the
    none option always being correct, and duplicate options always sharing
    a truth value.
    """
    n = len(questions)

    # none option
    none_present = 0
    none_correct = 0
    for q in questions:
        letters = [L for L in "ABCD"
                   if f"option_{L}" in q and NONE_MARKER in str(q[f"option_{L}"])]
        if not letters:
            continue
        none_present += 1
        golden = set(q["golden_answer"].split(",")) - {""}
        if any(L in golden for L in letters):
            none_correct += 1

    # duplicate option text
    dupes = 0
    dupes_same_label = 0
    for q in questions:
        texts = {}
        for L in "ABCD":
            key = f"option_{L}"
            if key not in q:
                continue
            texts.setdefault(str(q[key]).strip().lower(), []).append(L)
        rep = [ls for ls in texts.values() if len(ls) > 1]
        if rep:
            dupes += 1
            golden = set(q["golden_answer"].split(",")) - {""}
            if all(len({L in golden for L in ls}) == 1 for ls in rep):
                dupes_same_label += 1

    # position balance
    pos = Counter()
    for q in questions:
        for L in q["golden_answer"].split(","):
            if L:
                pos[L] += 1
    counts = [pos.get(L, 0) for L in "ABCD"]
    spread = (max(counts) - min(counts)) / max(1, sum(counts))

    # answer cardinality
    card = Counter(len([x for x in q["golden_answer"].split(",") if x])
                   for q in questions)

    return {
        "n_questions": n,
        "none_option": {
            "present_pct": round(100 * none_present / max(1, n), 1),
            "correct_when_present_pct": round(100 * none_correct / max(1, none_present), 1)
                if none_present else None,
            "n_present": none_present,
        },
        "duplicates": {
            "pct_with_duplicates": round(100 * dupes / max(1, n), 1),
            "n": dupes,
        },
        "position_balance": {
            "counts": {L: pos.get(L, 0) for L in "ABCD"},
            "spread_pct": round(100 * spread, 1),
        },
        "cardinality_pct": {k: round(100 * v / max(1, n), 1)
                            for k, v in sorted(card.items())},
    }


# ------------------------------------------------------------ the gate
def run_gate(questions: List[Dict],
             max_leakage_gap_pp: float = 15.0,
             max_length_gap: float = 4.0,
             max_none_correct_pct: float = 75.0,
             max_position_spread_pct: float = 12.0,
             verbose: bool = True) -> Dict:
    """
    Run every check and say whether the batch passes.

    Thresholds are deliberately not set at perfection. AER's leakage gap was
    28.7 points, so 15 is a real tightening while still being achievable.
    A batch that fails should go back for distractor rewriting, not be
    quietly shipped.
    """
    leak = leakage_check(questions)
    lens = length_balance(questions)
    overlap = word_overlap(questions)
    struct = structural_check(questions)

    failures = []

    if "gap_pp" in leak:
        if leak["gap_pp"] > max_leakage_gap_pp and leak.get("reliable"):
            failures.append(f"style leakage {leak['gap_pp']}pp above baseline "
                            f"(limit {max_leakage_gap_pp})")

    if "gap" in lens and lens["gap"] > max_length_gap:
        failures.append(f"option length gap {lens['gap']} words "
                        f"(limit {max_length_gap})")

    ncp = struct["none_option"]["correct_when_present_pct"]
    if ncp is not None and ncp > max_none_correct_pct:
        failures.append(f"'none' option correct {ncp}% of the time "
                        f"(limit {max_none_correct_pct}) - exploitable")

    if struct["position_balance"]["spread_pct"] > max_position_spread_pct:
        failures.append(f"answer positions uneven, spread "
                        f"{struct['position_balance']['spread_pct']}%")

    report = {
        "passed": not failures,
        "failures": failures,
        "leakage": leak,
        "length_balance": lens,
        "word_overlap": overlap,
        "structural": struct,
    }

    if verbose:
        print("=" * 62)
        print(f"QUALITY GATE  ({len(questions)} questions)")
        print("=" * 62)
        if "gap_pp" in leak:
            flag = "" if leak["gap_pp"] <= max_leakage_gap_pp else "  <-- FAIL"
            print(f"  style leakage    {leak['accuracy']*100:.1f}% vs "
                  f"{leak['baseline']*100:.1f}% baseline "
                  f"(+{leak['gap_pp']}pp){flag}")
            if leak.get("note"):
                print(f"                   {leak['note']}")
        if "gap" in lens:
            flag = "" if lens["gap"] <= max_length_gap else "  <-- FAIL"
            print(f"  option lengths   correct {lens['correct_mean']} vs "
                  f"wrong {lens['wrong_mean']} words (gap {lens['gap']}){flag}")
        if "gap" in overlap:
            print(f"  word overlap     correct {overlap['correct_mean']} vs "
                  f"wrong {overlap['wrong_mean']}")
        no = struct["none_option"]
        print(f"  'none' option    present {no['present_pct']}%, correct "
              f"{no['correct_when_present_pct']}% of those")
        print(f"  duplicates       {struct['duplicates']['pct_with_duplicates']}%")
        pb = struct["position_balance"]
        print(f"  positions        {pb['counts']} (spread {pb['spread_pct']}%)")
        print(f"  cardinality      {struct['cardinality_pct']}")
        print("-" * 62)
        print("  RESULT: PASSED" if not failures else "  RESULT: FAILED")
        for f in failures:
            print(f"    - {f}")
        print("=" * 62)

    return report
