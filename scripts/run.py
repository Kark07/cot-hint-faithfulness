"""
Single reproduction command. See REPRODUCE.md.

Fixture dry-run (no GPU, no network):

    python scripts/run.py --dry-run --fixtures-only

Engineering smoke test (<= 5 questions; output gitignored, not analysed):

    python scripts/run.py --limit 3 --max-new-tokens 256

Outcome run (GATED — only after protocol review):

    python scripts/run.py --config config/run.yaml --confirm-protocol-approved
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runner import (  # noqa: E402
    AttemptCollision, ComputeCapExceeded, attempt_dir_for, dry_run, real_run,
)

SMOKE_MAX = 5


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "run.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use the stub generator; no model, no network.")
    ap.add_argument("--fixtures-only", action="store_true",
                    help="Load fixtures from tests/fixtures (requires --dry-run).")
    ap.add_argument("--fixtures-path", type=Path,
                    default=ROOT / "tests" / "fixtures" / "synthetic_examples.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help=f"Smoke test on at most {SMOKE_MAX} questions (real model).")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Smoke tests only: override max_new_tokens to save time.")
    ap.add_argument("--confirm-protocol-approved", action="store_true",
                    help="Required for the full outcome run (PROTOCOL.md §9).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dry_run:
        if not args.fixtures_only:
            print("ERROR: --dry-run requires --fixtures-only.", file=sys.stderr)
            return 2
        out = args.out or ROOT / "results" / "_fixture_dryrun.jsonl"
        n = dry_run(args.fixtures_path, out, limit=args.limit, seed=args.seed)
        print(f"[dry-run] wrote {n} rows to {out}")
        return 0

    import yaml
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    smoke = args.limit is not None
    if smoke:
        if not 1 <= args.limit <= SMOKE_MAX:
            print(f"ERROR: smoke tests are limited to {SMOKE_MAX} questions.", file=sys.stderr)
            return 2
        out = ROOT / "results" / "_smoke.jsonl"
    else:
        if not args.confirm_protocol_approved:
            print("ERROR: the outcome run is gated until protocol review (PROTOCOL.md §9).\n"
                  "Re-run with --confirm-protocol-approved only after approval.", file=sys.stderr)
            return 2
        if args.max_new_tokens is not None:
            print("ERROR: --max-new-tokens is for smoke tests only; the outcome run uses the "
                  "frozen config.", file=sys.stderr)
            return 2
        if _git("status", "--porcelain"):
            print("ERROR: working tree has uncommitted changes; the outcome run must be tied "
                  "to a clean commit.", file=sys.stderr)
            return 2
        out = ROOT / "results" / f"run-{_git('rev-parse', 'HEAD')}.jsonl"

    # Smoke outputs are gitignored scratch and may be replaced; outcome attempts never are.
    try:
        log = real_run(cfg, out, limit=args.limit, max_new_tokens=args.max_new_tokens,
                       keep_filtered=smoke, replace_existing=smoke,
                       commit=_git("rev-parse", "HEAD"))
    except AttemptCollision as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except ComputeCapExceeded as e:
        print(f"INCOMPLETE: {e}. No final output was written. The attempt record is kept at "
              f"{attempt_dir_for(out)} (status: deadline_expired).", file=sys.stderr)
        return 4
    except BaseException:
        print(f"ATTEMPT DID NOT COMPLETE. The attempt record is kept at {attempt_dir_for(out)}.",
              file=sys.stderr)
        raise
    print(f"[{'smoke' if smoke else 'outcome'}] wrote {log['n_retained']} rows to {out}")
    print(f"filter log: {out}.filters.json")
    print(f"attempt record: {attempt_dir_for(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
