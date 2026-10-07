# Change log

Changes after the October 6 handoff. The scientific protocol (`PROTOCOL.md`) is unchanged throughout.

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
