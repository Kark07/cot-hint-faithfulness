# cot-hint-faithfulness

> **Status: PRE-OUTCOME — protocol under review.**
> No result-bearing comparison has been run. This repository contains the pre-registered protocol, the standardized BU1LD handoff artifacts, and a working engineering harness that runs end-to-end on synthetic fixtures. The outcome run is gated until the protocol is reviewed.

A reproducible test of whether final-output evaluation misses hint-driven, unacknowledged answer changes. One small open model (`Qwen/Qwen2.5-1.5B-Instruct`), one public task (MMLU), one hint modality (sycophancy hint adapted from Chen et al. 2025; suggested-answer idea from Turpin et al. 2023), four independent metrics (correctness / influence / faithfulness / safety).

## What this is

- `PROTOCOL.md` — pre-registered protocol v1.0 (frozen).
- `FREEZE.json` — freeze manifest (model/dataset revisions, seeds, sample size, rejection rule, pre-result commit SHA).
- `CLAIMS.md` — claim → evidence mapping (all rows currently *unsupported — not yet run*).
- `REPRODUCE.md` — shortest clean path to reproduce (fixture dry-run today; gated outcome run after review).
- `CLOSEOUT.md` — status note.
- `analysis.py` — the exact analysis script that would produce reported numbers.
- `src/` — harness implementation.
- `tests/` — engineering tests on synthetic fixtures.
- `examples/hand_scored.md` — hand-labeled CoTs showing the annotation rubric in action.

## Fixture dry-run (works today, no GPU, no network)

```
pip install -r requirements.txt
python scripts/run.py --dry-run --fixtures-only
```

## Outcome run (gated — do not execute until PROTOCOL review complete)

See `REPRODUCE.md`.

## License

MIT. See `LICENSE`.
