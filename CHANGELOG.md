# Change log

Changes after the October 6 handoff. The scientific protocol (`PROTOCOL.md`) is unchanged throughout.

## 2026-10-12: parser and annotation-completeness corrections (final source review)

Reviewed revision: active freeze `a295e47a72fb80ab87e6b9b83f34c25fedff52f2` (main at `3ec16c3351059b3d2ab4b561a0228cf328946f5f`).
Diff: `git diff 3ec16c3..ACTIVE_FREEZE_PENDING`

**Not changed:** `PROTOCOL.md`, `config/run.yaml`, model and dataset revisions, sample and filter rules, seeds, prompts, metric definitions, the Wilson interval, the minimum of 30, the decision thresholds, `src/runner.py`, `src/model.py`, `src/dataset.py`, `src/prompts.py`, `src/metrics.py`, `src/schema.py`. No outcome run has been performed.

1. **Answer extraction (`src/scoring.py`).** `extract_answer` now accepts only a complete explicit answer statement (`Answer:` followed by `(X)`, or by a bare capital letter ending its line). Ordinary words and option mentions in the reasoning can neither supply nor replace an answer; unsupported or truncated responses stay empty. The last explicit statement wins when there are several. The loose fallbacks (`the answer is X`, `option X`, a lone letter) are removed. Exact rule: `OPERATIONAL_CLARIFICATIONS.md` C5.
2. **Annotation completeness (`analysis.py`).** Labels are validated against the three rubric values. No hypothesis decision is issued while any switched row is unlabeled (`ANNOTATION_INCOMPLETE`) or invalid (`ANNOTATION_INVALID`); progress counts stay visible and are marked as progress only. With complete valid annotation the unchanged three-zone rule applies. Rule: `OPERATIONAL_CLARIFICATIONS.md` C6.
3. **M1 reporting clarification (documentation).** M1 is measured on the fresh greedy No-hint generations and is not guaranteed to be 100% by the sampled-answer filter. `OPERATIONAL_CLARIFICATIONS.md` C4.

**Synthetic tests:** `tests/test_scoring.py` (the reviewer's four regression strings, valid explicit answers, empty cases, multiple explicit answers) and `tests/test_analysis.py` (invalid labels, incomplete annotation including the reviewer's 30 + 70 case, completed three-zone cases, unchanged thresholds and interval). Receipt: `receipts/2026-10-12-pytest-synthetic.txt`.

**Limits:** these are synthetic tests. The stricter parser has not been run on real model outputs; if the model states answers in a form outside C5, those outputs become parse failures and are reported as such.

## 2026-10-09: correction to the operational clarification (documentation only)

Added entry C3 to `OPERATIONAL_CLARIFICATIONS.md` at the reviewer's request: after an unclean termination the last file-write time is only a lower bound on elapsed runtime; a recorded termination time is used where available and its evidence preserved; otherwise elapsed time is labelled uncertain and the unobserved interval is not treated as unused retry budget. The author's earlier bullet is kept struck through and marked superseded. No source, test, config or protocol file changed; the active freeze is unchanged.

## 2026-10-08: operational clarification (documentation only)

Added `OPERATIONAL_CLARIFICATIONS.md` recording the reviewer's answers: the six-hour cap is an aggregate ceiling across attempts on cumulative elapsed runner time, and a retry is possible only for an infrastructure failure after review. `REPRODUCE.md` and `CLOSEOUT.md` now point to it. No source, test, config or protocol file changed, and the active freeze (`a295e47a72fb80ab87e6b9b83f34c25fedff52f2`) is unchanged. The harness enforces the cap per attempt, not the aggregate; that gap is stated in the clarification.

## Harness corrections from the October 6 source review

Reviewed revision: `861244905eede2cc183a07a5215b3a133e4f26fa` (pointer commit `7d911f4ba0fa69030927ad87a22d533403af54cd`).
Diff: `git diff 7d911f4..a295e47` (active freeze `a295e47a72fb80ab87e6b9b83f34c25fedff52f2`)

**Not changed:** model and dataset revisions, sample and filter rules, seeds, prompts and hint definitions, metric definitions, decision thresholds, `PROTOCOL.md`, `config/run.yaml`, `src/prompts.py`, `src/dataset.py`, `src/scoring.py`, `src/metrics.py`, `src/schema.py`, `examples/`. No outcome run has been performed.

### 1. Preserve progress and failed attempts
- `src/runner.py::real_run` writes an attempt manifest (`<output>.attempt/attempt.json`) with a unique attempt ID and the commit SHA before the model is loaded.
- Probe results and every generation batch are appended and fsynced to per-stage files as they are produced. Each row carries attempt ID, stage, condition, seed and question ID.
- The manifest records the terminal state: `complete`, `failed` (with traceback), `interrupted`, or `deadline_expired`. An attempt killed without warning is left as `running` with its stage files.
- Incomplete attempts are never resumed or reused. `analysis.py` refuses a stage file, an output whose attempt is not `complete`, and a `run-*.jsonl` with no attempt record. No metric or decision logic in `analysis.py` changed.

### 2. Compute cap effective throughout inference
- One wall-clock deadline per attempt, checked around every probe call and every generation batch, and again before the final output is written.
- `src/model.py::generate_batches` (was `generate_all`) yields per batch and accepts a stop callback, wired to a `StoppingCriteria` so an in-flight generation stops at the next decoding step after the deadline. A batch that ends after the deadline is discarded.
- The manifest records measured elapsed wall-clock time, overshoot, and the terminal state. The cap is described as a coarse wall-clock cap, not a metered GPU-hour measurement. Sampling settings and the seed handling are unchanged.

### 3. Refuse output collisions
- The attempt directory is created with an atomic exclusive `mkdir` before any expensive work. The run refuses to start if the output file, its filter log, or an attempt directory already exists, including one left by an interrupted attempt.
- The final output and filter log are opened in exclusive-create mode. The runner has no overwrite option. Smoke-test scratch files are cleared by `scripts/run.py` for `_smoke` paths only.

### Synthetic test receipts
`tests/test_runner_durability.py` (fake model, fake clock, no network) covers the review's acceptance checks: exception after an early batch and before the final condition; deadline exceeded during probing, during stability sampling, during the final biasing condition, mid-batch, and after the last batch; pre-existing output refused before model initialisation; concurrent claims with exactly one winner; interrupted artifact byte-identical after a retry; primary-result path refusing incomplete attempts. The saved test output is in `receipts/`.

### Limits
These are synthetic tests of control flow. The real-model path was exercised only by a 1-question CPU smoke test (not analysed). The in-flight stop has not been timed on a GPU. Whether a retry after an incomplete attempt is permitted, and whether the cap applies per attempt or across attempts, is left to the reviewer (see `REPRODUCE.md`).
