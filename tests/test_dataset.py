"""Tests for the filter cascade. Synthetic MMLU rows only — no network."""

from pathlib import Path

import pytest

from src.dataset import (
    alphabetical_x_selector,
    filter_correctness,
    filter_stability,
    filter_token_alignment,
    load_fixtures,
)
from src.prompts import Question


FIXTURES = Path(__file__).parent / "fixtures" / "synthetic_examples.jsonl"


def test_load_fixtures_shape():
    qs = load_fixtures(FIXTURES)
    assert len(qs) == 10
    for q in qs:
        assert isinstance(q, Question)
        assert len(q.options) == 4
        assert q.gold_letter in {"A", "B", "C", "D"}


def test_x_selector_picks_an_incorrect_letter():
    qs = load_fixtures(FIXTURES)
    for q in qs:
        x = alphabetical_x_selector(q)
        assert x != q.gold_letter


def _always_aligned(q, x):
    return True, []


def _never_aligned(q, x):
    return False, []


def test_token_alignment_filter_keeps_all_when_aligner_ok():
    qs = load_fixtures(FIXTURES)
    kept = filter_token_alignment(qs, alphabetical_x_selector, _always_aligned)
    assert len(kept) == len(qs)


def test_token_alignment_filter_drops_all_when_aligner_fails():
    qs = load_fixtures(FIXTURES)
    kept = filter_token_alignment(qs, alphabetical_x_selector, _never_aligned)
    assert kept == []


def test_stability_filter_drops_unstable():
    qs = load_fixtures(FIXTURES)
    aligned = filter_token_alignment(qs, alphabetical_x_selector, _always_aligned)

    def _sampler_identical(q):
        return [q.gold_letter] * 3

    def _sampler_mixed(q):
        return [q.gold_letter, q.gold_letter, "A" if q.gold_letter != "A" else "B"]

    kept_stable = filter_stability(aligned, _sampler_identical)
    kept_unstable = filter_stability(aligned, _sampler_mixed)
    assert len(kept_stable) == len(qs)
    assert kept_unstable == []


def test_correctness_filter_keeps_only_matching_gold():
    qs = load_fixtures(FIXTURES)
    aligned = filter_token_alignment(qs, alphabetical_x_selector, _always_aligned)
    stable = filter_stability(aligned, lambda q: [q.gold_letter] * 3)
    kept = filter_correctness(stable)
    assert len(kept) == len(qs)

    # Now swap the sampler so the "stable" answer is NOT gold — all should drop.
    stable_wrong = filter_stability(aligned, lambda q: ["Z"] * 3 if False else ["A"] * 3)
    # For questions whose gold IS "A", some should pass; for the rest, drop.
    kept2 = filter_correctness(stable_wrong)
    expected = sum(1 for q in qs if q.gold_letter == "A")
    assert len(kept2) == expected


def test_question_rejects_bad_option_count():
    with pytest.raises(ValueError):
        Question(qid="bad", subject="x", stem="?", options=["a", "b", "c"], gold_idx=0)


def test_question_rejects_bad_gold_idx():
    with pytest.raises(ValueError):
        Question(qid="bad", subject="x", stem="?", options=["a", "b", "c", "d"], gold_idx=5)
