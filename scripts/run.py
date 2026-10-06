"""
Single reproduction command. See REPRODUCE.md.

Fixture dry-run (today, no GPU, no network):

    python scripts/run.py --dry-run --fixtures-only

Outcome run (GATED — do not execute before protocol review):

    python scripts/run.py --config config/run.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runner import dry_run, real_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "run.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use the stub generator; no model, no network.")
    ap.add_argument("--fixtures-only", action="store_true",
                    help="Load fixtures from tests/fixtures (requires --dry-run).")
    ap.add_argument("--fixtures-path", type=Path,
                    default=ROOT / "tests" / "fixtures" / "synthetic_examples.jsonl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "_fixture_dryrun.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dry_run and args.fixtures_only:
        n = dry_run(args.fixtures_path, args.out, limit=args.limit, seed=args.seed)
        print(f"[dry-run] wrote {n} rows to {args.out}")
        return 0

    if args.dry_run and not args.fixtures_only:
        print("ERROR: --dry-run currently requires --fixtures-only (no non-fixture dry path).",
              file=sys.stderr)
        return 2

    # Non-dry-run path is the gated outcome run.
    import yaml
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    real_run(cfg, args.out)  # raises; see src/runner.py
    return 0


if __name__ == "__main__":
    sys.exit(main())
