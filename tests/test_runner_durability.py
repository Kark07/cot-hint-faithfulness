"""Synthetic tests for the real-run durability contract (no model, no network).

A fake model handle and a fake clock stand in for Qwen and wall time, so these
tests exercise src.runner.real_run itself: exclusive attempt claiming,
incremental persistence, failure/interrupt records, and deadline enforcement.
"""

import json
from pathlib import Path

import pytest

from src.dataset import load_fixtures
from src.runner import (
    AttemptCollision,
    ComputeCapExceeded,
    attempt_dir_for,
    filters_path_for,
    real_run,
)
from src.schema import ExampleRecord

FIXTURES = Path(__file__).parent / "fixtures" / "synthetic_examples.jsonl"
N_Q = 10
BATCH = 4
N_BATCHES_PER_STAGE = 3          # ceil(10 / 4)

CFG = {
    "model": {"name": "fake-model", "revision": "fake-rev", "precision": "bf16"},
    "dataset": {"subjects": [], "per_subject_raw": 0, "revision": "fake-data", "target_n": 200},
    "generation": {"T_stability": 0.7, "T_answer": 0.0, "max_new_tokens": 64, "batch_size": BATCH},
    "seeds": {"stability": [42, 43, 44], "stratified_sampling": 42, "final_answer_generation": 42},
    "compute": {"gpu_hour_cap": 1},     # 3600 fake seconds
}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _ConstTokenizer:
    """Every prompt tokenizes to the same ids, so the alignment filter passes."""

    def __call__(self, text, add_special_tokens=False):
        class Enc:
            input_ids = [1, 2, 3]
        return Enc()


class FakeHandle:
    """Answers every question with its gold letter. Hooks let a test advance the
    fake clock or raise at a chosen probe call / generation batch."""

    precision = "fake"
    device_name = "fake-device"

    def __init__(self, gold_by_stem, on_probe=None, on_batch=None):
        self.tokenizer = _ConstTokenizer()
        self.gold_by_stem = gold_by_stem
        self.on_probe = on_probe
        self.on_batch = on_batch
        self.probe_calls = 0
        self.batch_calls = 0

    def chat(self, user_text):
        return "<chat>" + user_text

    def _gold(self, user_text):
        return next(g for stem, g in self.gold_by_stem.items() if stem in user_text)

    def answer_letter_logits(self, user_text):
        self.probe_calls += 1
        if self.on_probe:
            self.on_probe(self.probe_calls)
        gold = self._gold(user_text)
        return {L: (5.0 if L == gold else 1.0) for L in "ABCD"}

    def generate_batches(self, user_texts, temperature, seed, max_new_tokens, batch_size=16):
        for i in range(0, len(user_texts), batch_size):
            self.batch_calls += 1
            if self.on_batch:
                self.on_batch(self.batch_calls)
            yield [f"Reasoning. Answer: ({self._gold(t)})" for t in user_texts[i:i + batch_size]]


def _run(tmp_path, clock=None, on_probe=None, on_batch=None, out_name="run-fakesha.jsonl", **kw):
    questions = load_fixtures(FIXTURES)
    gold = {q.stem: q.gold_letter for q in questions}
    made = []

    def factory(cfg):
        h = FakeHandle(gold, on_probe=on_probe, on_batch=on_batch)
        made.append(h)
        return h

    out = tmp_path / out_name
    call = lambda: real_run(  # noqa: E731
        CFG, out, commit="fakesha", handle_factory=factory,
        question_loader=lambda cfg: list(questions), clock=clock or FakeClock(), **kw,
    )
    return out, made, call


def _attempt(out):
    return json.loads((attempt_dir_for(out) / "attempt.json").read_text(encoding="utf-8"))


def _lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Successful run
# ---------------------------------------------------------------------------

def test_complete_run_writes_outputs_and_complete_record(tmp_path):
    out, made, call = _run(tmp_path)
    log = call()

    rows = [ExampleRecord.model_validate_json(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == N_Q == log["n_retained"]
    assert filters_path_for(out).exists()

    a = _attempt(out)
    assert a["status"] == "complete" and a["incomplete"] is False
    assert a["completed_stages"] == ["probe", "stability", "no_hint", "neutral_hint",
                                     "biasing_hint", "finalize"]
    assert a["commit"] == "fakesha"
    adir = attempt_dir_for(out)
    for stage in ("probe", "stability-seed42", "stability-seed43", "stability-seed44",
                  "no_hint", "neutral_hint", "biasing_hint"):
        assert len(_lines(adir / f"{stage}.jsonl")) == N_Q


# ---------------------------------------------------------------------------
# Correction 3: exclusive attempt path, collisions refused before inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("leftover", ["final", "filters", "attempt_dir"])
def test_collision_refused_before_any_inference(tmp_path, leftover):
    out, made, call = _run(tmp_path)
    if leftover == "final":
        out.write_text("PRIOR RESULT\n", encoding="utf-8")
        sentinel = out
    elif leftover == "filters":
        filters_path_for(out).write_text("PRIOR LOG", encoding="utf-8")
        sentinel = filters_path_for(out)
    else:
        attempt_dir_for(out).mkdir()
        sentinel = attempt_dir_for(out) / "attempt.json"
        sentinel.write_text('{"status": "running"}', encoding="utf-8")
    before = sentinel.read_text(encoding="utf-8")

    with pytest.raises(AttemptCollision):
        call()

    assert made == []                                   # model never constructed
    assert sentinel.read_text(encoding="utf-8") == before   # nothing overwritten


def test_second_run_after_success_is_refused_and_output_untouched(tmp_path):
    out, _, call = _run(tmp_path)
    call()
    before = out.read_bytes()
    out2, made2, call2 = _run(tmp_path)
    with pytest.raises(AttemptCollision):
        call2()
    assert made2 == [] and out.read_bytes() == before


def test_second_run_after_interrupted_attempt_is_refused(tmp_path):
    def boom(n):
        if n == 2:
            raise KeyboardInterrupt
    out, _, call = _run(tmp_path, on_batch=boom)
    with pytest.raises(KeyboardInterrupt):
        call()
    record_before = (attempt_dir_for(out) / "attempt.json").read_text(encoding="utf-8")

    out2, made2, call2 = _run(tmp_path)
    with pytest.raises(AttemptCollision):
        call2()
    assert made2 == []
    assert (attempt_dir_for(out) / "attempt.json").read_text(encoding="utf-8") == record_before


def test_smoke_mode_may_replace_its_own_scratch_output(tmp_path):
    out, _, call = _run(tmp_path, out_name="_smoke.jsonl", replace_existing=True)
    call()
    out2, made2, call2 = _run(tmp_path, out_name="_smoke.jsonl", replace_existing=True)
    call2()
    assert _attempt(out2)["status"] == "complete" and len(made2) == 1


# ---------------------------------------------------------------------------
# Correction 1: incremental progress + durable failed / interrupted record
# ---------------------------------------------------------------------------

def test_failure_mid_generation_keeps_progress_and_failed_record(tmp_path):
    # Batches 1-3 are stability-seed42; batch 5 is the 2nd batch of stability-seed43.
    def boom(n):
        if n == 5:
            raise RuntimeError("simulated CUDA failure")
    out, _, call = _run(tmp_path, on_batch=boom)
    with pytest.raises(RuntimeError, match="simulated CUDA failure"):
        call()

    adir = attempt_dir_for(out)
    a = _attempt(out)
    assert a["status"] == "failed" and a["incomplete"] is True
    assert "simulated CUDA failure" in a["error"]
    assert a["stage"] == "stability-seed43"
    assert a["completed_stages"] == ["probe"]
    assert a["progress"]["stability-seed43"] == {"done": BATCH, "total": N_Q}
    # Everything produced before the failure is on disk.
    assert len(_lines(adir / "probe.jsonl")) == N_Q
    assert len(_lines(adir / "stability-seed42.jsonl")) == N_Q
    assert len(_lines(adir / "stability-seed43.jsonl")) == BATCH
    # No final output or success log exists for a failed attempt.
    assert not out.exists() and not filters_path_for(out).exists()


def test_keyboard_interrupt_is_recorded_as_interrupted(tmp_path):
    def boom(n):
        if n == 1:
            raise KeyboardInterrupt
    out, _, call = _run(tmp_path, on_batch=boom)
    with pytest.raises(KeyboardInterrupt):
        call()
    a = _attempt(out)
    assert a["status"] == "interrupted" and a["incomplete"] is True
    assert len(_lines(attempt_dir_for(out) / "probe.jsonl")) == N_Q
    assert not out.exists()


def test_progress_is_on_disk_before_the_run_ends(tmp_path):
    """A hard kill cannot run handlers, so progress must already be durable
    while the run is still in flight."""
    seen = {}

    def peek(n):
        if n == 3:      # during the 3rd batch of stability-seed42
            adir = attempt_dir_for(tmp_path / "run-fakesha.jsonl")
            seen["status"] = json.loads((adir / "attempt.json").read_text(encoding="utf-8"))["status"]
            seen["probe"] = len(_lines(adir / "probe.jsonl"))
            seen["stab"] = len(_lines(adir / "stability-seed42.jsonl"))
    _, _, call = _run(tmp_path, on_batch=peek)
    call()
    assert seen == {"status": "running", "probe": N_Q, "stab": 2 * BATCH}


# ---------------------------------------------------------------------------
# Correction 2: deadline enforced inside probe / batch loops and after the end
# ---------------------------------------------------------------------------

def test_deadline_expires_inside_probe_loop(tmp_path):
    clock = FakeClock()

    def slow(n):
        if n == 4:
            clock.now = 4000.0          # past the 3600 s cap during the 4th probe
    out, made, call = _run(tmp_path, clock=clock, on_probe=slow)
    with pytest.raises(ComputeCapExceeded):
        call()
    a = _attempt(out)
    assert a["status"] == "deadline_expired" and a["incomplete"] is True
    assert a["stage"] == "probe"
    assert made[0].probe_calls == 4     # stopped immediately, did not finish the loop
    assert made[0].batch_calls == 0     # never reached generation
    # The probe that finished after the deadline is not persisted.
    assert len(_lines(attempt_dir_for(out) / "probe.jsonl")) == 3
    assert not out.exists()


def test_deadline_expires_inside_generation_batches(tmp_path):
    clock = FakeClock()

    def slow(n):
        if n == 2:
            clock.now = 4000.0          # during the 2nd batch of stability-seed42
    out, made, call = _run(tmp_path, clock=clock, on_batch=slow)
    with pytest.raises(ComputeCapExceeded):
        call()
    a = _attempt(out)
    assert a["status"] == "deadline_expired" and a["stage"] == "stability-seed42"
    assert made[0].batch_calls == 2     # no further batches were started
    assert len(_lines(attempt_dir_for(out) / "stability-seed42.jsonl")) == BATCH
    assert not out.exists()


def test_deadline_expires_after_final_condition(tmp_path):
    clock = FakeClock()
    last_batch = 6 * N_BATCHES_PER_STAGE    # 3 stability passes + 3 conditions

    def slow(n):
        if n == last_batch:
            clock.now = 4000.0
    out, made, call = _run(tmp_path, clock=clock, on_batch=slow)
    with pytest.raises(ComputeCapExceeded):
        call()
    a = _attempt(out)
    assert a["status"] == "deadline_expired" and a["incomplete"] is True
    assert made[0].batch_calls == last_batch
    assert "finalize" not in a["completed_stages"]
    assert not out.exists() and not filters_path_for(out).exists()


def test_deadline_checked_at_finalize_even_with_all_batches_in_time(tmp_path):
    """Time can also run out between the last batch and writing the output."""
    clock = FakeClock()
    questions = load_fixtures(FIXTURES)
    gold = {q.stem: q.gold_letter for q in questions}

    class LateChat(FakeHandle):
        def chat(self, user_text):
            # chat() is called when assembling records, after all generation.
            if self.batch_calls == 6 * N_BATCHES_PER_STAGE:
                clock.now = 4000.0
            return super().chat(user_text)

    out = tmp_path / "run-fakesha.jsonl"
    with pytest.raises(ComputeCapExceeded):
        real_run(CFG, out, commit="fakesha", handle_factory=lambda cfg: LateChat(gold),
                 question_loader=lambda cfg: list(questions), clock=clock)
    a = _attempt(out)
    assert a["status"] == "deadline_expired" and a["stage"] == "finalize"
    assert a["completed_stages"][-1] == "biasing_hint"
    assert not out.exists()
