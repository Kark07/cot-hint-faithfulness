"""
MMLU loader + filter cascade. See PROTOCOL.md §2.2 and §4.

Can be exercised without network via `load_fixtures()` which reads
tests/fixtures/synthetic_examples.jsonl. The runner's --fixtures-only path
uses this and never hits HuggingFace.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Iterable

from .prompts import Question, LETTERS


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_mmlu(
    subjects: list[str],
    per_subject: int,
    seed: int,
    revision: str | None = None,
) -> list[Question]:
    """Load MMLU questions. Imports datasets lazily so --fixtures-only runs
    without the dependency being resolvable at import time.
    """
    from datasets import load_dataset  # type: ignore

    rng = random.Random(seed)
    out: list[Question] = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test", revision=revision)
        rows = list(ds)
        rng.shuffle(rows)
        for i, r in enumerate(rows[:per_subject]):
            out.append(Question(
                qid=f"mmlu:{subj}:{i:04d}",
                subject=subj,
                stem=r["question"],
                options=list(r["choices"]),
                gold_idx=int(r["answer"]),
                meta={"source": "cais/mmlu", "subject": subj},
            ))
    return out


def load_fixtures(path: Path) -> list[Question]:
    """Load synthetic MMLU-shaped rows for engineering tests and the dry-run."""
    out: list[Question] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            out.append(Question(
                qid=r["qid"],
                subject=r["subject"],
                stem=r["stem"],
                options=r["options"],
                gold_idx=int(r["gold_idx"]),
                meta=r.get("meta", {}),
            ))
    return out


# ----------------------------------------------------------------------------
# Filter cascade (PROTOCOL.md §4)
# ----------------------------------------------------------------------------

def filter_token_alignment(
    questions: list[Question],
    x_selector: Callable[[Question], str],
    aligner: Callable[[Question, str], tuple[bool, list[int]]],
) -> list[tuple[Question, str, list[int]]]:
    """Keep questions whose neutral and biasing variants token-align.

    Returns [(question, x_letter, diff_positions), ...].
    """
    kept = []
    for q in questions:
        x = x_selector(q)
        ok, diffs = aligner(q, x)
        if ok:
            kept.append((q, x, diffs))
    return kept


def filter_stability(
    items: Iterable[tuple[Question, str, list[int]]],
    stability_sampler: Callable[[Question], list[str]],
) -> list[tuple[Question, str, list[int], list[str]]]:
    """Keep questions whose unhinted answers are identical, parseable letters across samples."""
    kept = []
    for q, x, diffs in items:
        samples = stability_sampler(q)
        if samples and len(set(samples)) == 1 and samples[0] in LETTERS:
            kept.append((q, x, diffs, samples))
    return kept


def filter_correctness(
    items: Iterable[tuple[Question, str, list[int], list[str]]],
) -> list[tuple[Question, str, list[int], list[str]]]:
    """Keep questions whose stable unhinted answer equals the ground truth."""
    kept = []
    for q, x, diffs, samples in items:
        stable_answer = samples[0]
        if stable_answer == q.gold_letter:
            kept.append((q, x, diffs, samples))
    return kept


def alphabetical_x_selector(q: Question) -> str:
    """Deterministic fallback X selector (PROTOCOL.md §4.3)."""
    return q.incorrect_letters()[0]
