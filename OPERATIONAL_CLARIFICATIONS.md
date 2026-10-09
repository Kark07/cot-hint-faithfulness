# Operational clarifications

Dated clarifications of how the frozen comparison is operated. They are recorded here so that the historical protocol is not edited: `PROTOCOL.md` and `config/run.yaml` are unchanged since the original October 6 freeze (`861244905eede2cc183a07a5215b3a133e4f26fa`). Nothing in this file changes a model or dataset revision, sample or filter rule, seed, prompt, metric, or decision threshold.

Where this file and an older operational description (for example in `REPRODUCE.md`) differ, this file governs.

---

## 2026-10-08 — Compute ceiling and retries

**Source:** reviewer's reply to the harness-correction package (active freeze `a295e47a72fb80ab87e6b9b83f34c25fedff52f2`, pointer commit `925018036a827db47823386487606ebc1fda91d2`), answering two questions the author had left open.

**Status of this entry:** documentation only. Recorded after the active freeze; no source, test, config or protocol file is changed by it, and no new freeze is made for it. No outcome-bearing run has been executed.

### C1. The six-hour cap is an aggregate ceiling

- The six-hour cap (`PROTOCOL.md` §2.3 and §6; `compute.gpu_hour_cap` in `config/run.yaml`) is a single ceiling for this frozen comparison **across all attempts**, including failed and interrupted attempts. It is not a fresh allowance for each attempt or for each commit.
- The operational limit is **cumulative elapsed runner time**: the sum, over every attempt, of the wall-clock time recorded in that attempt's `attempt.json` (`elapsed_seconds`), measured from the start of the attempt and including model and dataset loading. This is deliberately conservative.
- Cumulative elapsed runner time is **not** a metered GPU-hour measurement and is not reported as one. Any overshoot past the ceiling is recorded (`wall_clock_overshoot_seconds`) and reported.
- ~~For an attempt that was killed without a terminal record (status left as `running`), the last recorded `elapsed_seconds` understates the time used. Its elapsed time is then taken as the interval from `started_utc` to the latest modification time of any file in its attempt directory, and this is stated when the budget is reported.~~ **Superseded on 2026-10-09 by C3 below.** This bullet was the author's own addition, not the reviewer's wording, and the reviewer corrected it. The original text is kept here struck through as a record.
- No paid compute is approved.

### C2. Retries

- A retry may be considered **only for an infrastructure failure**, and only after the following have been recorded and reviewed by the reviewer: the previous attempt, the reason it failed, the outputs that were retained, and the remaining aggregate budget under C1.
- A new commit must not be used merely to get around the runner's collision refusal or to reset the budget.
- Any approved retry has its own exclusive attempt identity, leaves every earlier attempt artifact unchanged, and uses the frozen scientific settings without modification.
- Whether to retry is never decided on the basis of favorable or unfavorable scientific outcomes.
- All attempts, complete or not, are reported.

### How this relates to the harness as frozen at `a295e47`

- Already enforced by the harness: one wall-clock deadline per attempt, checked throughout inference; elapsed time and overshoot recorded per attempt; exclusive attempt directory; refusal to start when any output or attempt record exists for the commit; no overwrite option; `analysis.py` refuses incomplete attempts.
- **Not enforced by the harness:** the aggregate ceiling in C1. The runner applies the cap to each attempt separately and does not subtract time used by earlier attempts. This has no effect on a first attempt. Before any retry is approved under C2, the remaining aggregate budget must be computed from the earlier attempt records, and the way it is applied to the retry must be recorded and reviewed first.
- Not enforceable by code: the review requirement and the "not to bypass" condition in C2. These are procedural commitments recorded here.

### Limitation restated

The safety field in this comparison is not an empirical safety assessment. The task is benign and no safety classifier is run, so it does not establish safety performance (`PROTOCOL.md` §5, M4).

---

## 2026-10-09 — Elapsed time after an unclean termination (corrects one bullet of C1)

**Source:** reviewer's reply to the 2026-10-08 entry (commit `3fb899e30dcb3417a162f23b9819425b3bd81846`). The reviewer confirmed that the aggregate six-hour ceiling, the infrastructure-only retry rule and the no-paid-compute restriction are consistent with what was asked, and corrected one accounting detail that the author had added.

**Status of this entry:** documentation only. No source, test, config or protocol file is changed, the active freeze remains `a295e47a72fb80ab87e6b9b83f34c25fedff52f2`, and no code change was requested. This entry is not an outcome-run release.

### C3. Accounting for an attempt that ended without a terminal record

This replaces the struck-through bullet in C1.

- After an unclean termination (for example a lost session that leaves `attempt.json` at status `running`), the last file-write timestamp in the attempt directory is only a **lower bound** on elapsed runtime. It is not used as the elapsed time.
- Where a recorded process or session termination time is available (for example from the compute platform's session or runtime log), that time is used as the end of the attempt, and the evidence is preserved alongside the attempt record. The attempt's own files are not modified.
- Where the end time is unknown, the attempt's elapsed time is labelled **uncertain**. The unobserved interval after the last file write is not treated as unused retry budget.
- Any retry remains blocked until a conservative remaining-budget calculation, made on this basis, has been reviewed by the reviewer (see C2).

