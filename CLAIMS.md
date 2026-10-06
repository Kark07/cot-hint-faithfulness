# Claims → Evidence Mapping

**Status:** No outcome claims yet. The result-bearing run is gated until protocol review completes.

Every conclusion in any prose summary of this project MUST map to a specific artifact row below. An unsupported conclusion is marked `unsupported` here, not softened into prose. Negative or null results are preserved as-is, not reinterpreted.

## Primary claim

| # | Claim (template, instantiated post-outcome) | Supporting artifact | Status |
|---|---|---|---|
| C1 | At `n = <N>`, `verbalization_rate` among hint-switched instances is `<rate>` (Wilson 95% CI `[<lo>, <hi>]`). Under the pre-registered three-zone rule (PROTOCOL.md §6) the outcome is `<H1 rejected / H1 supported / inconclusive / insufficient evidence>`. | `results/run-<sha>.jsonl` → `analysis.py::verbalization_rate()` | unsupported — not yet run |

## Secondary claims

| # | Claim (template) | Supporting artifact | Status |
|---|---|---|---|
| C2 | At `n = <N>`, `switch_rate` under the biasing hint is `<rate>` (Wilson 95% CI). | `results/run-<sha>.jsonl` → `analysis.py::switch_rate()` | unsupported — not yet run |
| C3 | Per-subject verbalization rates: `<table>`. (Descriptive only; no per-subject rejection rule.) | `results/run-<sha>.jsonl` → `analysis.py::per_subject_verbalization()` | unsupported — not yet run |
| C4 | `neutral_shift_rate` is `<rate>`, indicating the persona surface form `<does / does not>` destabilize answers independent of hint direction. | `results/run-<sha>.jsonl` → `analysis.py::neutral_shift_rate()` | unsupported — not yet run |
| C5 | `safety_pass_rate` under the biasing-hint condition is 1.00 by construction (no safety classifier is run; see PROTOCOL.md §5 M4). Reported only to show that a final-output safety grade would pass these outputs while faithfulness can fail. | `results/run-<sha>.jsonl` → `analysis.py::safety_pass_rate()` | unsupported — not yet run |

## Rules this file enforces

- No prose claim in `README.md`, `CLOSEOUT.md`, or any report may be added without a corresponding row here that cites a specific artifact.
- A row marked `unsupported` remains `unsupported` until a run is performed under the frozen protocol and its artifact is cited.
- If the outcome run falsifies H1, C1 is still recorded with its negative result and the row's Status becomes `supported — H1 rejected`. The claim is not softened or restated.
- If the frozen protocol must change before a run, see `PROTOCOL.md §10`: the original protocol stays in place and a successor is recorded.
