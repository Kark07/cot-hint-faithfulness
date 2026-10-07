# Reproduce

Two paths: a fixture dry-run that works today without GPU or network, and the gated outcome run that executes only after protocol review.

## Pre-outcome dry-run (today — no GPU, no network)

```
pip install -r requirements.txt
python scripts/run.py --dry-run --fixtures-only
python analysis.py --input results/_fixture_dryrun.jsonl
```

Expected: schema-valid rows written to `results/_fixture_dryrun.jsonl`, no HuggingFace downloads, no GPU required. `analysis.py` prints aggregated metrics using Wilson 95% CIs on the synthetic data. The dry-run file is gitignored.

## Engineering tests

```
pytest -q
```

Covers token-alignment of the three hint variants, dataset filter cascade behaviour on synthetic MMLU rows, and answer extraction on a battery of CoT shapes. `tests/test_runner_durability.py` runs the real-run code path with a fake model and a fake clock (no model, no network) and checks: collisions are refused before inference, progress is on disk while a run is in flight, failed and interrupted attempts leave a durable record, and the compute deadline is enforced inside the probe loop, inside generation batches, and after the final condition.

## Engineering smoke test (allowed before review; at most 5 questions)

```
python scripts/run.py --limit 3 --max-new-tokens 256
```

Loads the pinned model and dataset, runs the full filter cascade and all three conditions on the first 3 questions (in smoke mode, questions that fail a filter are still generated so every code path runs; their pass flags are recorded), and writes `results/_smoke.jsonl`, `results/_smoke.jsonl.filters.json` and the attempt record `results/_smoke.jsonl.attempt/`. Smoke outputs are scratch: a new smoke test replaces the previous one (the outcome run never replaces anything). These files are gitignored and are not analysed or reported. A GPU is strongly recommended; on CPU this takes several minutes.

## Outcome run (gated — DO NOT run before protocol review)

```
python scripts/run.py --config config/run.yaml --confirm-protocol-approved
python analysis.py --input results/run-<commit_sha>.jsonl --summary-out results/summary-<commit_sha>.json
```

- The runner refuses to start without `--confirm-protocol-approved`, and refuses if the working tree has uncommitted changes, so the output file name `run-<commit_sha>` always points at the exact code that produced it.
- **Attempt record and no-overwrite rule.** Before the model is loaded, the runner exclusively creates `results/run-<commit_sha>.jsonl.attempt/`. If that directory, the output file, or its `.filters.json` log already exists, the runner refuses to start (exit code 3). Nothing is ever overwritten, including the leftovers of an interrupted attempt.
- **Incremental progress.** Every probe result and every generation batch is appended and synced to a stage file in the attempt directory as it is produced (`probe.jsonl`, `stability-seed<N>.jsonl`, `no_hint.jsonl`, `neutral_hint.jsonl`, `biasing_hint.jsonl`). `attempt.json` records the status: `running`, `complete`, `failed`, `interrupted` or `deadline_expired`, with the stage reached and per-stage progress. A status of `running` with no live process means the attempt was killed (for example a lost Colab session).
- **Compute deadline.** The cap in `config/run.yaml` is measured from the start of the attempt and checked around every probe call and every generation batch, and again before the final output is written. On expiry the attempt is recorded as `deadline_expired` with `incomplete: true`, no final output is written (exit code 4), and work that finished after the deadline is not persisted.
- **Incomplete attempts are records, not inputs.** Partial stage files are never resumed or reused. An incomplete attempt is committed as-is. Because the runner refuses a second attempt at the same commit, any further attempt happens at a later commit that contains the earlier attempt record, with the frozen configuration unchanged, and only with the reviewer's agreement. All attempts are reported.
- Compute requirement (estimate, not yet measured on a T4): single Colab T4 free-tier session, ~2-3 hours wallclock for `n = 200` across three conditions per question.
- No paid credits, no API spend. If free-tier capacity is insufficient, sample size is reduced before capacity is purchased (see `PROTOCOL.md §2.3`).
- Manual annotation of the influence subset (`acknowledgment_label`) is performed after generation; see `PROTOCOL.md §7` for the rubric.

## Environment pins

Exact pins live in `requirements.txt`. Model and dataset revisions are recorded in `FREEZE.json` at the freeze commit.
