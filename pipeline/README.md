# Dataset pipeline

Code for building a financial abductive reasoning benchmark: questions that
give a market event and ask which of four candidate causes actually explain
it.

Design decisions here come from four sources: the EDA on the SemEval AER
dataset, the AER task paper (how they built theirs), the AILS-NTUA winning
system paper (where models fail), and the CRAB and UNcommonsense papers.

## Files

| file | what it does |
|---|---|
| `big_moves.py` | find the days worth asking about |
| `retrieval.py` | price data and news retrieval |
| `extraction.py` | pull candidate cause events from articles |
| `llm.py` | one interface to three model providers |
| `scoring.py` | three-model scoring, variance routing, Krippendorff alpha |
| `assembly.py` | build the four-option questions |
| `quality.py` | leakage and structural shortcut gates |
| `finalise.py` | human review sheets, splits, EDA |
| `evaluate.py` | run models against the finished dataset |
| `checkpoint.py` | resumable loops so a dropped session costs one item |
| `pipeline_stages_1_2.ipynb` | events and documents |
| `pipeline_stages_3_8.ipynb` | candidates through to evaluation |

## Running it

Needs internet and API keys, so run on Colab or locally.

```
pip install yfinance trafilatura openai anthropic google-generativeai scikit-learn
```

Run `pipeline_stages_1_2.ipynb` first, it writes `topics.json`. Then
`pipeline_stages_3_8.ipynb` picks that up.

## Surviving Colab disconnects

Colab sessions drop: idle timeout at around 90 minutes, a hard cap at 12
hours, and whenever the laptop sleeps. Retrieval and scoring both run for
hours, so this matters.

The fix is not to keep the session alive, it is to make a disconnect cheap.
All three long loops go through `resumable_map`, which saves after every
item and skips anything already done when you re-run the cell. A drop at
item 90 of 100 costs one item, not ninety.

This matters most on the scoring stage, where redoing work means paying for
the same API calls twice.

```python
from checkpoint import checkpoint_status, show_failures

checkpoint_status("scored_progress.json")   # where did it get to
show_failures("scored_progress.json")       # what went wrong
```

Then just re-run the same cell. To retry items that errored, pass
`retry_failed=True`.

Work inside Google Drive so the checkpoint files survive the session:

```python
from google.colab import drive
drive.mount('/content/drive')
import os; os.chdir('/content/drive/MyDrive/dataset')
```

## The three design problems this is built around

**1. What counts as a big move day**

A fixed percentage does not work across four assets, since crypto moving 5%
is an ordinary day and a major currency pair moving 1% is not. Everything
here is volatility-relative.

Two methods. The simple one divides each day's return by a trailing
volatility estimate. The other is the Lee-Mykland jump test (2008, Review of
Financial Studies), which estimates volatility with bipower variation so a
jump cannot inflate its own yardstick, and takes its threshold from a
Gumbel distribution rather than a number chosen by hand.

Tested on synthetic series with known jumps injected, some hidden inside
high-volatility stretches:

| method | jumps caught | false positives |
|---|---|---|
| z-score (3.0) | 5 of 5 | 7 |
| Lee-Mykland (0.01) | 4 of 5 | 0 |

Lee-Mykland is the default. For dataset building precision beats recall: a
false positive means writing a question about a day with no story behind
it, while a missed jump only costs one question. It is also far less
sensitive to its parameter, flagging the same days at alpha 0.05 and 0.01
where the z-score count swings from 54 to 6.

**2. Labelling without hand-checking everything**

Every candidate is scored 0 to 3 by three models from different families.
Where they agree the label is settled, where they disagree a human looks at
it. Usually about a quarter needs review.

The four-level scale is deliberate. CRAB used 0 to 100 and their annotators
only reached 0.28 agreement. AER used three levels and reached 0.51. People
agree far better on a few buckets than on an exact number.

Krippendorff's alpha needs at least two annotators to exist, since it
measures agreement between people. The review sheets give everyone a shared
subset for the alpha calculation and split the rest, so with three people
each sees about 43% of the queue.

**3. Stopping models from cheating**

Two shortcuts, both found in AER, both gated here.

*Style leakage.* In the EDA a classifier reading only the option text, with
no question and no documents, scored 89.4% against a 60.6% baseline. ART
was cleaned until the same number fell to about 51%. `quality.py` runs this
check and fails the batch if the gap is too wide.

*Structural leakage.* The winning AER system gained 5.6 points from rules
applied after the model answered, because the "none of the others" option
was correct every time it appeared and duplicate options always shared a
truth value. Here the none option is sometimes wrong, and duplicates are
merged rather than repeated.

The gate was tested against a deliberately leaky batch (42.8pp above
baseline, failed) and a matched one (below baseline, passed).

## Making the questions hard

Rather than picking events at random and hoping, the assembly targets the
three failure patterns the AILS-NTUA team found across 14 models from 7
families:

- **causal chain incompleteness** — models pick one link and miss the rest
- **proximate cause preference** — they take the most recent trigger and
  miss the enabling condition
- **salience bias** — they take the dramatic cause and miss the quiet one

Under-selections outnumbered over-selections 1,389 to 52 in their analysis.
`evaluate.py` measures that ratio directly, so whether the same pattern
appears in financial reasoning is testable.

Distractors are stratified into the three AER types: temporal (happened
after, so a consequence), semantic (shares entities, not a cause), and
background (real long-run condition, scored too low).

## What is tested and what is not

Tested here, with synthetic data:

- jump detection against known injected jumps
- scoring routing and the Krippendorff implementation
- question assembly at 300 topics: cardinality came out 55/33/12 against
  AER's 56/30/14, positions balanced, zero duplicates, none option correct
  52% of the time instead of AER's 100%
- the quality gate distinguishing leaky from clean batches
- evaluation metrics and the failure-mode breakdown
- splitting with zero event leakage across train, dev and test

Not tested, because it needs network access:

- price download, news retrieval, article extraction
- every model API call

Expect the first real run to need small fixes, particularly around article
extraction, where paywalls and blocks will lose a share of URLs.

## Target volume

AER averages 19.7 documents and roughly 28,000 tokens of evidence per
topic. That volume is a large part of what makes the task hard, since the
model has to locate the relevant events inside it. Both notebooks check
against this.
