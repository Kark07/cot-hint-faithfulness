# results/

Empty until the outcome run is approved by the reviewer. No result-bearing comparison may be committed here before `PROTOCOL.md` review completes.

Expected contents post-approval:

- `run-<commit_sha>.jsonl` — one JSON row per retained question, schema from `src/schema.py`.
- `run-<commit_sha>.jsonl.filters.json` — filter counts and provenance for a completed run.
- `run-<commit_sha>.jsonl.attempt/` — the attempt record, created before inference: `attempt.json` (status `running` / `complete` / `failed` / `interrupted` / `deadline_expired`) plus per-stage files written batch by batch. Kept for every attempt, including incomplete ones. Never overwritten or deleted.
- `summary-<commit_sha>.json` — aggregated metrics written by `analysis.py --summary-out results/summary-<commit_sha>.json`.
- `annotations-<commit_sha>.jsonl` — manual acknowledgment labels for the influence subset (merged into the run file for the final analysis).

Fixture dry-run outputs (`_fixture_dryrun.jsonl`) are produced by `scripts/run.py --dry-run --fixtures-only` and are gitignored.
