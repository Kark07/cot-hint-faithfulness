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

Covers token-alignment of the three hint variants, dataset filter cascade behaviour on synthetic MMLU rows, and answer extraction on a battery of CoT shapes.

## Engineering smoke test (allowed before review; at most 5 questions)

```
python scripts/run.py --limit 3 --max-new-tokens 256
```

Loads the pinned model and dataset, runs the full filter cascade and all three conditions on the first 3 questions (in smoke mode, questions that fail a filter are still generated so every code path runs; their pass flags are recorded), and writes `results/_smoke.jsonl` plus `results/_smoke.jsonl.filters.json`. These files are gitignored and are not analysed or reported. A GPU is strongly recommended; on CPU this takes several minutes.

## Outcome run (gated — DO NOT run before protocol review)

```
python scripts/run.py --config config/run.yaml --confirm-protocol-approved
python analysis.py --input results/run-<commit_sha>.jsonl
```

- The runner refuses to start without `--confirm-protocol-approved`, and refuses if the working tree has uncommitted changes, so the output file name `run-<commit_sha>` always points at the exact code that produced it.
- Compute requirement: single Colab T4 free-tier session, ~2-3 hours wallclock for `n = 200` across three conditions per question.
- No paid credits, no API spend. If free-tier capacity is insufficient, sample size is reduced before capacity is purchased (see `PROTOCOL.md §2.3`).
- Manual annotation of the influence subset (`acknowledgment_label`) is performed after generation; see `PROTOCOL.md §7` for the rubric.

## Environment pins

Exact pins live in `requirements.txt`. Model and dataset revisions are recorded in `FREEZE.json` at the freeze commit.
