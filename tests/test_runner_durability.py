"""Synthetic tests for the real-run durability contract (no model, no network).

A fake model handle and a fake clock stand in for Qwen and wall time, so these
tests exercise src.runner.real_run itself: exclusive attempt claiming,
incremental persistence, failure/interrupt records, and deadline enforcement.
"""

import json
import sys
import threading
from pathlib import Path

import pytest

from src.dataset import load_fixtures
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analysis  # noqa: E402
from src.runner import (  # noqa: E402
    AttemptCollision,
    ComputeCapExceeded,
    attempt_dir_for,
    claim_attempt,
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

    STEPS_PER_BATCH = 8      # simulated decoding steps, where should_stop is polled

    def __init__(self, gold_by_stem, on_probe=None, on_batch=None, on_step=None):
        self.tokenizer = _ConstTokenizer()
        self.gold_by_stem = gold_by_stem
        self.on_probe = on_probe
        self.on_batch = on_batch
        self.on_step = on_step
        self.probe_calls = 0
        self.batch_calls = 0
        self.steps_run = []          # decoding steps actually run, per batch
        self.stopped_in_flight = 0

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

    def generate_batches(self, user_texts, temperature, seed, max_new_tokens, batch_size=16,
                         should_stop=None):
        for i in range(0, len(user_texts), batch_size):
            self.batch_calls += 1
            if self.on_batch:
                self.on_batch(self.batch_calls)
            steps, stopped = 0, False
            for step in range(self.STEPS_PER_BATCH):
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if self.on_step:
                    self.on_step(self.batch_calls, step)
                steps += 1
            self.steps_run.append(steps)
            self.stopped_in_flight += stopped
            if stopped:
                yield ["TRUNCATED" for _ in user_texts[i:i + batch_size]]
            else:
                yield [f"Reasoning. Answer: ({self._gold(t)})" for t in user_texts[i:i + batch_size]]


def _run(tmp_path, clock=None, on_probe=None, on_batch=None, on_step=None,
         out_name="run-fakesha.jsonl", **kw):
    questions = load_fixtures(FIXTURES)
    gold = {q.stem: q.gold_letter for q in questions}
    made = []

    def factory(cfg):
        h = FakeHandle(gold, on_probe=on_probe, on_batch=on_batch, on_step=on_step)
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
    assert a["commit"] == "fakesha"                       # immutable experiment revision
    assert len(a["attempt_id"]) == 32                     # unique attempt id
    assert a["wall_clock_overshoot_seconds"] == 0.0
    assert json.loads(filters_path_for(out).read_text(encoding="utf-8"))["attempt_id"] == a["attempt_id"]
    adir = attempt_dir_for(out)
    row = _lines(adir / "stability-seed43.jsonl")[0]
    assert (row["stage"], row["condition"], row["seed"]) == ("stability-seed43", "no_hint", 43)
    assert row["attempt_id"] == a["attempt_id"] and row["qid"]
    row = _lines(adir / "biasing_hint.jsonl")[0]
    assert (row["stage"], row["condition"], row["seed"]) == ("biasing_hint", "biasing_hint", 42)
    row = _lines(adir / "probe.jsonl")[0]
    assert (row["stage"], row["condition"]) == ("probe", "no_hint") and row["qid"]
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
    snapshot = lambda: {p.name: p.read_bytes() for p in sorted(attempt_dir_for(out).iterdir())}  # noqa: E731
    before = snapshot()
    assert json.loads(before["attempt.json"])["status"] == "interrupted"

    out2, made2, call2 = _run(tmp_path)
    with pytest.raises(AttemptCollision):
        call2()
    assert made2 == []                  # refused before model initialisation
    assert snapshot() == before         # interrupted artifact byte-identical


def test_runner_has_no_overwrite_option():
    import inspect
    assert "replace_existing" not in inspect.signature(real_run).parameters
    assert list(inspect.signature(claim_attempt).parameters) == ["out_path"]


def test_concurrent_claims_only_one_succeeds(tmp_path):
    out = tmp_path / "run-fakesha.jsonl"
    n = 16
    barrier = threading.Barrier(n)
    wins, losses = [], []

    def claim():
        barrier.wait()
        try:
            wins.append(claim_attempt(out))
        except AttemptCollision:
            losses.append(1)

    threads = [threading.Thread(target=claim) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and len(losses) == n - 1


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


def test_deadline_expires_during_final_biasing_condition(tmp_path):
    clock = FakeClock()
    target = 5 * N_BATCHES_PER_STAGE + 2    # 2nd batch of the biasing condition

    def slow(n):
        if n == target:
            clock.now = 4000.0
    out, made, call = _run(tmp_path, clock=clock, on_batch=slow)
    with pytest.raises(ComputeCapExceeded):
        call()
    a = _attempt(out)
    assert a["status"] == "deadline_expired" and a["incomplete"] is True
    assert a["stage"] == "biasing_hint"
    assert a["completed_stages"] == ["probe", "stability", "no_hint", "neutral_hint"]
    assert made[0].batch_calls == target            # no new work after the deadline
    adir = attempt_dir_for(out)
    assert len(_lines(adir / "neutral_hint.jsonl")) == N_Q      # earlier progress retained
    assert len(_lines(adir / "biasing_hint.jsonl")) == BATCH
    assert not out.exists() and not filters_path_for(out).exists()


def test_in_flight_generation_is_stopped_at_the_deadline(tmp_path):
    """The deadline passes in the middle of a batch: decoding must stop early
    instead of running the batch to the end, and the batch is discarded."""
    clock = FakeClock()

    def mid_batch(batch, step):
        if batch == 2 and step == 2:
            clock.now = 3700.0          # 100 s past the 3600 s cap
    out, made, call = _run(tmp_path, clock=clock, on_step=mid_batch)
    with pytest.raises(ComputeCapExceeded):
        call()
    h = made[0]
    assert h.stopped_in_flight == 1
    assert h.steps_run == [FakeHandle.STEPS_PER_BATCH, 3]   # stopped after step 2, not run to 8
    a = _attempt(out)
    assert a["status"] == "deadline_expired" and a["stage"] == "stability-seed42"
    rows = _lines(attempt_dir_for(out) / "stability-seed42.jsonl")
    assert len(rows) == BATCH and all("TRUNCATED" not in r["output"] for r in rows)
    # Measured elapsed time and overshoot are recorded with the terminal state.
    assert a["elapsed_seconds"] == 3700.0
    assert a["wall_clock_overshoot_seconds"] == 100.0
    assert a["wall_clock_cap_seconds"] == 3600.0
    assert "not metered GPU time" in a["timing_note"]


# ---------------------------------------------------------------------------
# The primary-result path (analysis.py) refuses anything but a completed attempt
# ---------------------------------------------------------------------------

def test_analysis_accepts_a_completed_attempt(tmp_path):
    out, _, call = _run(tmp_path)
    call()
    analysis.check_primary_input(out)       # no exception


def test_analysis_refuses_failed_attempt_and_its_stage_files(tmp_path):
    def boom(n):
        if n == 5:
            raise RuntimeError("simulated failure")
    out, _, call = _run(tmp_path, on_batch=boom)
    with pytest.raises(RuntimeError):
        call()
    adir = attempt_dir_for(out)
    # 1) retained partial progress cannot be analysed as a result
    with pytest.raises(analysis.IncompleteAttemptError, match="partial progress"):
        analysis.check_primary_input(adir / "stability-seed42.jsonl")
    # 2) even if someone hand-builds the output file, the failed record blocks it
    out.write_text((adir / "stability-seed42.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(analysis.IncompleteAttemptError, match="'failed'"):
        analysis.check_primary_input(out)


def test_analysis_refuses_deadline_expired_attempt(tmp_path):
    clock = FakeClock()

    def slow(n):
        if n == 2:
            clock.now = 4000.0
    out, _, call = _run(tmp_path, clock=clock, on_batch=slow)
    with pytest.raises(ComputeCapExceeded):
        call()
    out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(analysis.IncompleteAttemptError, match="'deadline_expired'"):
        analysis.check_primary_input(out)


def test_analysis_refuses_outcome_file_with_no_attempt_record(tmp_path):
    orphan = tmp_path / "run-deadbeef.jsonl"
    orphan.write_text("{}\n", encoding="utf-8")
    with pytest.raises(analysis.IncompleteAttemptError, match="no attempt record"):
        analysis.check_primary_input(orphan)


def test_analysis_still_runs_on_fixture_dryrun_files(tmp_path):
    f = tmp_path / "_fixture_dryrun.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    analysis.check_primary_input(f)         # engineering file, not a primary result


def test_smoke_scratch_helper_only_touches_smoke_paths(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run as run_script
    with pytest.raises(ValueError):
        run_script._clear_smoke_scratch(tmp_path / "run-fakesha.jsonl")
    smoke = tmp_path / "_smoke.jsonl"
    smoke.write_text("x", encoding="utf-8")
    attempt_dir_for(smoke).mkdir()
    run_script._clear_smoke_scratch(smoke)
    assert not smoke.exists() and not attempt_dir_for(smoke).exists()
