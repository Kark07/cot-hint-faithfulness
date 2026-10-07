"""
Orchestrator: filter cascade -> generate three conditions per question -> JSONL rows.

Two modes:
  real_run(cfg)  — exercises load_mmlu + ModelHandle. Gated: see scripts/run.py.
  dry_run()      — exercises load_fixtures + a stub generator; no GPU, no network.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .dataset import (
    alphabetical_x_selector,
    filter_correctness,
    filter_stability,
    filter_token_alignment,
    load_fixtures,
    load_mmlu,
)
from .prompts import LETTERS, Question, assert_token_aligned, build_prompt, AlignmentError
from .schema import ConditionResult, ExampleRecord, derive_fields
from .scoring import extract_answer, safety_pass_mmlu


def _stable_bit(*parts: str) -> int:
    """Deterministic 0/1 from strings; Python's hash() is salted per process."""
    return hashlib.sha256("|".join(parts).encode("utf-8")).digest()[0] & 1


# ----------------------------------------------------------------------------
# Stub generator for the dry-run
# ----------------------------------------------------------------------------

class StubGenerator:
    """Deterministic, no-ML generator used only by the fixtures dry-run.

    Reproduces the schema's shape (CoT text, answer letter, optional logits)
    without any inference. The CoT text is constructed to exercise the answer
    extractor on realistic shapes.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def unhinted_answer(self, q: Question) -> str:
        # Stub baseline: returns gold on fixtures, so stability + correctness
        # filters both pass. Real model generation replaces this.
        return q.gold_letter

    def generate(self, prompt: str, q: Question, x: str | None, condition: str) -> dict:
        if condition == "no_hint":
            ans = self.unhinted_answer(q)
            cot = (
                "Let me work through this. "
                "Looking at the options and the question stem, I apply the relevant knowledge. "
                f"Answer: ({ans})"
            )
        elif condition == "neutral_hint":
            # Neutral control: answer should not shift in the stub.
            ans = self.unhinted_answer(q)
            cot = (
                "Considering the question carefully. "
                "The professor's attention doesn't change my analysis. "
                f"Answer: ({ans})"
            )
        elif condition == "biasing_hint":
            # Stub: in the fixture dry-run we force a switch to X on half the
            # questions, so the harness exercises the switched_to_hint path.
            assert x is not None
            flip = _stable_bit(q.qid, "flip") == 0
            ans = x if flip else self.unhinted_answer(q)
            # When flipped, mimic the two canonical reasoning shapes (acknowledged
            # vs not) so the fixture dry-run surfaces annotation handling.
            if flip and _stable_bit(q.qid, "ack") == 0:
                cot = (
                    "The Stanford professor's suggestion points to this option. "
                    "On that basis I agree with the hinted answer. "
                    f"Answer: ({ans})"
                )
            else:
                cot = (
                    "Working through the options one more time, the reasoning favours this choice. "
                    f"Answer: ({ans})"
                )
        else:
            raise ValueError(condition)
        return {"cot": cot, "answer": extract_answer(cot)}


# ----------------------------------------------------------------------------
# Core per-question pipeline
# ----------------------------------------------------------------------------

def _make_record(
    q: Question,
    x_letter: str,
    stability_samples: list[str],
    diffs: list[int],
    gens: dict[str, dict],
    model_name: str,
    seed: int,
    run_id: str,
    stability_pass: bool = True,
    correctness_pass: bool = True,
) -> ExampleRecord:
    def _cond(key: str) -> ConditionResult:
        g = gens[key]
        return ConditionResult(
            prompt=g["prompt"],
            cot=g["cot"],
            answer=g["answer"],
            hint_target=x_letter if key == "biasing_hint" else None,
        )

    record = ExampleRecord(
        question_id=q.qid,
        subject=q.subject,
        question=q.stem,
        options=q.options,
        ground_truth=q.gold_letter,
        x_hinted_incorrect=x_letter,
        stability_answers=stability_samples,
        stability_pass=stability_pass,
        correctness_pass=correctness_pass,
        token_alignment_pass=True,
        token_alignment_diff_positions=diffs,
        no_hint=_cond("no_hint"),
        neutral_hint=_cond("neutral_hint"),
        biasing_hint=_cond("biasing_hint"),
        safety_pass=safety_pass_mmlu(gens["biasing_hint"]["cot"], gens["biasing_hint"]["answer"]),
        model=model_name,
        seed=seed,
        run_id=run_id,
    )
    return derive_fields(record)


# ----------------------------------------------------------------------------
# Dry-run (fixtures only, no model, no network)
# ----------------------------------------------------------------------------

def dry_run(fixtures_path: Path, out_path: Path, limit: int | None = None, seed: int = 42) -> int:
    """Fixture dry-run. Writes JSONL to out_path. Returns the number of rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    questions = load_fixtures(fixtures_path)
    if limit:
        questions = questions[:limit]

    gen = StubGenerator(seed=seed)
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ-dryrun")

    # Stub "token aligner" that always passes — the real aligner needs a tokenizer.
    def _aligner(q, x):
        return True, []

    aligned = filter_token_alignment(questions, alphabetical_x_selector, _aligner)
    stable = filter_stability(aligned, lambda q: [gen.unhinted_answer(q)] * 3)
    retained = filter_correctness(stable)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for q, x, diffs, samples in retained:
            gens = {}
            for cond in ("no_hint", "neutral_hint", "biasing_hint"):
                prompt = build_prompt(q, cond, x_letter=x if cond == "biasing_hint" else None)
                g = gen.generate(prompt, q, x, cond)
                g["prompt"] = prompt
                gens[cond] = g
            record = _make_record(
                q, x, samples, diffs, gens,
                model_name="stub-dryrun",
                seed=seed,
                run_id=run_id,
            )
            # Fixture annotations: assign labels based on CoT content, so
            # analysis.py has something non-null to aggregate.
            if record.switched_to_hint:
                cot = record.biasing_hint.cot.lower()
                if "professor" in cot or "suggestion" in cot or "hint" in cot:
                    record.acknowledgment_label = "ACKNOWLEDGED"
                else:
                    record.acknowledgment_label = "NOT_ACKNOWLEDGED"
                record.annotator = "stub"
            f.write(record.model_dump_json() + "\n")
            n += 1
    return n


# ----------------------------------------------------------------------------
# Real run (gated in scripts/run.py; smoke tests use limit <= 5)
#
# Durability contract:
#   * An exclusive attempt directory <out_path>.attempt/ is claimed before the
#     model is loaded. If the final output, its filter log, or an attempt
#     directory already exists, the run refuses to start. Nothing is overwritten.
#   * attempt.json holds the attempt status: running | complete | failed |
#     interrupted | deadline_expired. A status of "running" with no live
#     process means the attempt was killed (e.g. a lost session).
#   * Every probe result and every generation batch is appended and fsynced to
#     a stage file as soon as it is produced.
#   * The compute deadline is checked around every probe call and every
#     generation batch, and again before the final output is written. Work
#     that finishes after the deadline is not persisted as output.
# Partial progress is preserved as a record; it is never resumed or reused.
# ----------------------------------------------------------------------------

STAGE_ORDER = ("probe", "stability", "no_hint", "neutral_hint", "biasing_hint", "finalize")


class ComputeCapExceeded(RuntimeError):
    pass


class AttemptCollision(RuntimeError):
    pass


class Deadline:
    """Compute cap measured from construction; clock is injectable for tests."""

    def __init__(self, cap_seconds: float, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._t0 = clock()
        self.cap_seconds = float(cap_seconds)

    def elapsed(self) -> float:
        return self._clock() - self._t0

    def check(self, stage: str) -> None:
        if self.elapsed() > self.cap_seconds:
            raise ComputeCapExceeded(
                f"compute cap of {self.cap_seconds:.0f}s exceeded during stage: {stage}"
            )


def attempt_dir_for(out_path: Path) -> Path:
    return Path(str(out_path) + ".attempt")


def filters_path_for(out_path: Path) -> Path:
    return Path(str(out_path) + ".filters.json")


def claim_attempt(out_path: Path, replace_existing: bool = False) -> Path:
    """Exclusively claim the attempt directory for out_path, or refuse.

    replace_existing is for smoke tests only (their scratch outputs are
    gitignored and never reported); the outcome run never passes it.
    """
    adir = attempt_dir_for(out_path)
    targets = (out_path, filters_path_for(out_path), adir)
    existing = [p for p in targets if p.exists()]
    if existing and not replace_existing:
        raise AttemptCollision(
            "refusing to start: output or attempt record already exists for this run: "
            + ", ".join(str(p) for p in existing)
            + ". Existing records are never overwritten."
        )
    for p in existing:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(adir)
    except FileExistsError as e:
        raise AttemptCollision(f"refusing to start: attempt directory already exists: {adir}") from e
    return adir


def _fsync_write(f, text: str) -> None:
    f.write(text)
    f.flush()
    os.fsync(f.fileno())


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        _fsync_write(f, json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_handle_factory(cfg: dict):
    from .model import ModelHandle
    m = cfg["model"]
    return ModelHandle(m["name"], m.get("revision"), m.get("precision", "bf16"))


def _default_question_loader(cfg: dict) -> list[Question]:
    d, seeds = cfg["dataset"], cfg["seeds"]
    return load_mmlu(d["subjects"], int(d["per_subject_raw"]), int(seeds["stratified_sampling"]),
                     revision=d.get("revision"))


def real_run(
    cfg: dict,
    out_path: Path,
    limit: int | None = None,
    max_new_tokens: int | None = None,
    keep_filtered: bool = False,
    replace_existing: bool = False,
    commit: str | None = None,
    handle_factory: Callable[[dict], object] = _default_handle_factory,
    question_loader: Callable[[dict], list[Question]] = _default_question_loader,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Filter cascade + three-condition generation with the pinned model.

    On success writes one ExampleRecord per retained question to out_path and a
    filter / provenance log to <out_path>.filters.json, and returns the log.
    keep_filtered (smoke tests only) keeps every aligned question regardless of
    stability/correctness, recording the real pass flags, so all code paths run.
    handle_factory, question_loader and clock exist so synthetic tests can run
    this function without the real model or dataset.
    """
    from .prompts import select_hint_target

    out_path = Path(out_path)
    adir = claim_attempt(out_path, replace_existing=replace_existing)

    m, d, g, seeds = cfg["model"], cfg["dataset"], cfg["generation"], cfg["seeds"]
    deadline = Deadline(float(cfg["compute"]["gpu_hour_cap"]) * 3600, clock)
    mnt = max_new_tokens or int(g["max_new_tokens"])
    bs = int(g.get("batch_size", 16))
    run_id = _utc_now()

    state: dict = {
        "status": "running",
        "incomplete": True,
        "stage": "claimed",
        "completed_stages": [],
        "progress": {},
        "error": None,
        "run_id": run_id,
        "commit": commit,
        "output_path": out_path.name,
        "started_utc": run_id,
        "compute_cap_seconds": deadline.cap_seconds,
        "limit": limit, "max_new_tokens": mnt, "batch_size": bs, "keep_filtered": keep_filtered,
        "config": cfg,
        "note": "status 'running' with no live process means the attempt was killed "
                "(for example a lost session); stage files show how far it got.",
    }

    def save(**updates) -> None:
        state.update(updates)
        state["updated_utc"] = _utc_now()
        state["elapsed_seconds"] = round(deadline.elapsed(), 1)
        _write_json_atomic(adir / "attempt.json", state)

    def stage_done(name: str) -> None:
        state["completed_stages"].append(name)
        save()

    save()
    try:
        handle = handle_factory(cfg)
        qs = question_loader(cfg)
        if limit:
            qs = qs[:limit]
        save(precision_used=handle.precision, device=handle.device_name, n_raw=len(qs))

        def run_generation(stage: str, qids: list[str], prompts: list[str],
                           temperature: float, seed: int) -> list[str]:
            """Generate batch by batch; append each batch to <stage>.jsonl as it lands."""
            save(stage=stage)
            state["progress"][stage] = {"done": 0, "total": len(prompts)}
            outs: list[str] = []
            with (adir / f"{stage}.jsonl").open("x", encoding="utf-8") as f:
                deadline.check(stage)
                for texts in handle.generate_batches(
                    prompts, temperature=temperature, seed=seed,
                    max_new_tokens=mnt, batch_size=bs,
                ):
                    # A batch that finished after the deadline is not persisted.
                    deadline.check(stage)
                    for t in texts:
                        _fsync_write(f, json.dumps(
                            {"qid": qids[len(outs)], "output": t, "answer": extract_answer(t)}
                        ) + "\n")
                        outs.append(t)
                    state["progress"][stage]["done"] = len(outs)
                    save()
            return outs

        # §4.3 hint target X from the direct-answer probe on the no-hint prompt.
        save(stage="probe")
        state["progress"]["probe"] = {"done": 0, "total": len(qs)}
        probe: dict[str, dict[str, float]] = {}
        with (adir / "probe.jsonl").open("x", encoding="utf-8") as f:
            for q in qs:
                deadline.check("probe")
                logits = handle.answer_letter_logits(build_prompt(q, "no_hint"))
                deadline.check("probe")
                probe[q.qid] = logits
                _fsync_write(f, json.dumps({"qid": q.qid, "option_logits": logits}) + "\n")
                state["progress"]["probe"]["done"] = len(probe)
        stage_done("probe")

        def x_selector(q: Question) -> str:
            return select_hint_target(q, [probe[q.qid][L] for L in LETTERS])

        # §4.1 token alignment on the chat-templated strings.
        def aligner(q: Question, x: str) -> tuple[bool, list[int]]:
            try:
                diffs = assert_token_aligned(
                    handle.tokenizer,
                    handle.chat(build_prompt(q, "neutral_hint")),
                    handle.chat(build_prompt(q, "biasing_hint", x_letter=x)),
                )
                return True, diffs
            except AlignmentError:
                return False, []

        aligned = filter_token_alignment(qs, x_selector, aligner)
        aligned_ids = [q.qid for q, _, _ in aligned]

        # §4.2 stability: three sampled no-hint generations per question.
        nh_prompts = [build_prompt(q, "no_hint") for q, _, _ in aligned]
        samples: list[list[str]] = [[] for _ in aligned]
        for s in seeds["stability"]:
            outs = run_generation(f"stability-seed{int(s)}", aligned_ids, nh_prompts,
                                  float(g["T_stability"]), int(s))
            for i, o in enumerate(outs):
                samples[i].append(extract_answer(o))
        by_qid = {q.qid: samples[i] for i, (q, _, _) in enumerate(aligned)}
        stable = filter_stability(aligned, lambda q: by_qid[q.qid])
        stage_done("stability")

        # §4.3 correctness, §4.4 cap at target_n with a seeded random subset.
        correct = filter_correctness(stable)
        if keep_filtered:
            retained = [(q, x, diffs, by_qid[q.qid]) for q, x, diffs in aligned]
        else:
            retained = list(correct)
        stable_ids = {q.qid for q, *_ in stable}
        correct_ids = {q.qid for q, *_ in correct}
        target_n = int(d["target_n"])
        if len(retained) > target_n:
            rng = random.Random(int(seeds["stratified_sampling"]))
            rng.shuffle(retained)
            retained = retained[:target_n]
        retained_ids = [q.qid for q, *_ in retained]
        save(n_after_alignment=len(aligned), n_after_stability=len(stable),
             n_after_correctness=len(correct), n_retained=len(retained))

        # Three conditions, greedy.
        gens: dict[str, list[dict]] = {}
        for cond in ("no_hint", "neutral_hint", "biasing_hint"):
            prompts = [build_prompt(q, cond, x_letter=x if cond == "biasing_hint" else None)
                       for q, x, _, _ in retained]
            outs = run_generation(cond, retained_ids, prompts, float(g["T_answer"]),
                                  int(seeds["final_answer_generation"]))
            gens[cond] = [{"prompt": handle.chat(p), "cot": o, "answer": extract_answer(o)}
                          for p, o in zip(prompts, outs)]
            stage_done(cond)

        # Deadline is enforced after the final condition too.
        save(stage="finalize")
        deadline.check("finalize")

        log = {
            "run_id": run_id,
            "commit": commit,
            "model": m["name"], "model_revision": m.get("revision"),
            "dataset_revision": d.get("revision"),
            "precision_used": handle.precision, "device": handle.device_name,
            "batch_size": bs, "max_new_tokens": mnt, "limit": limit,
            "n_raw": len(qs), "n_after_alignment": len(aligned),
            "n_after_stability": len(stable), "n_after_correctness": len(correct),
            "n_retained": len(retained),
            "keep_filtered": keep_filtered,
            "stability_parse_failures": sum(a == "" for v in by_qid.values() for a in v),
            "condition_parse_failures": {c: sum(r["answer"] == "" for r in gens[c]) for c in gens},
            "stability_answers": by_qid,
            "attempt_dir": adir.name,
            "elapsed_seconds": round(deadline.elapsed(), 1),
        }

        # Exclusive create: the final output can never replace an existing file.
        with out_path.open("x", encoding="utf-8") as f:
            for i, (q, x, diffs, smp) in enumerate(retained):
                record = _make_record(
                    q, x, smp, diffs, {c: gens[c][i] for c in gens},
                    model_name=m["name"], seed=int(seeds["final_answer_generation"]), run_id=run_id,
                    stability_pass=q.qid in stable_ids, correctness_pass=q.qid in correct_ids,
                )
                record.no_hint.option_logits = probe[q.qid]
                _fsync_write(f, record.model_dump_json() + "\n")
        with filters_path_for(out_path).open("x", encoding="utf-8") as f:
            _fsync_write(f, json.dumps(log, indent=2))
        stage_done("finalize")
    except ComputeCapExceeded as e:
        save(status="deadline_expired", error=str(e))
        raise
    except KeyboardInterrupt:
        save(status="interrupted", error="KeyboardInterrupt")
        raise
    except BaseException:
        save(status="failed", error=traceback.format_exc())
        raise

    save(status="complete", incomplete=False, stage="done")
    return log
