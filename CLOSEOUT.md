# Closeout — October 6, 2026

**Status:** `blocked`

**Update, October 7, 2026:** the October 6 source review did not approve the outcome run and requested three harness corrections (durable attempt records, compute deadline throughout inference, exclusive attempt path). They are implemented with synthetic tests; see `CHANGELOG.md` and `receipts/`. The original freeze is preserved in `FREEZE.json` (`freeze_history`).

**Active freeze:** `a295e47a72fb80ab87e6b9b83f34c25fedff52f2` (recorded in `FREEZE.json` as `pre_result_commit_sha`). The scientific protocol is unchanged from the original freeze.

**Update, October 8, 2026:** the reviewer's answers on the compute ceiling (aggregate across attempts) and on retries are recorded in `OPERATIONAL_CLARIFICATIONS.md`. Documentation only; `PROTOCOL.md`, the code and the active freeze are unchanged.

**Update, October 9, 2026:** the reviewer's correction on how elapsed time is counted after an unclean termination is recorded as entry C3 in `OPERATIONAL_CLARIFICATIONS.md`. Documentation only.

**Reason:** awaiting the final harness/analysis review and an explicit release; the outcome run is not approved and has not been run. The pre-outcome deliverables (`PROTOCOL.md`, `FREEZE.json`, harness implementation, `analysis.py` on fixtures, hand-scored annotation examples) were handed off under freeze commit SHA `861244905eede2cc183a07a5215b3a133e4f26fa` (the original October 6 freeze). No outcome claims are made and `CLAIMS.md` currently lists every row as `unsupported — not yet run`.

On protocol approval, the outcome run is executed per `REPRODUCE.md`. The outcome run uses only compute that is already legitimately free to the author (Colab free-tier T4). No paid credits or API spend are used.

**Negative-result preservation:** acknowledged. Whatever the outcome under the three-zone rule in `PROTOCOL.md §6` (H1 rejected, H1 supported, inconclusive, or insufficient evidence), including a result that falsifies H1, it is reported as-is with no re-run, re-seeding, metric substitution, or exclusion-criteria change. The `CLAIMS.md` row for C1 is updated with that outcome and the result is preserved.

**Protocol change policy:** if any element of `PROTOCOL.md` must change before a run, the original is preserved and a successor `PROTOCOL_v2.md` is created per `PROTOCOL.md §10`.
