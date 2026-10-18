"""The optimizer benchmark's known-answer scenarios, offline: litellm is replaced by a scripted model
that behaves like the real one where it matters (needs 2 turns to submit; answers wrongly when told
to skip the calculator), so each scenario has a known right outcome and costs nothing."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from tests.test_example import _resp

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "math_agent"


@pytest.fixture
def bench_spec(tmp_path, monkeypatch):
    pytest.importorskip("litellm")
    problems = [{"id": f"p{i}", "question": f"What is {i} + {i}?", "answer": 2 * i} for i in range(60)]
    data = tmp_path / "problems.jsonl"
    data.write_text("".join(json.dumps(p) + "\n" for p in problems))
    monkeypatch.syspath_prepend(str(EXAMPLE))
    for m in ("agent", "evaluator", "config", "bench"):
        sys.modules.pop(m, None)
    import importlib.util
    spec = importlib.util.spec_from_file_location("math_bench_spec", EXAMPLE / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PROBLEMS", data)

    async def fake_acompletion(model, messages, tools, **kw):
        task = next(m["content"] for m in messages if m["role"] == "user")
        i = int(task.split("What is ")[1].split(" +")[0])
        if "nearest hundred" in task and messages[-1]["role"] == "tool":   # the harmful edit: round
            return _resp([("submit_answer", {"answer": round(2 * i, -2)})])
        if messages[-1]["role"] == "tool":                   # second turn: submit (wrong on 1 in 10, so the
            return _resp([("submit_answer", {"answer": 2 * i + (i % 10 == 9)})])   # good harness isn't 100%)
        return _resp([("calculator", {"expression": f"{i}"}),           # first turn: two calculator calls
                      ("calculator", {"expression": f"{i}+{i}"})])

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return mod


def test_known_answer_scenarios_pass(bench_spec, tmp_path):
    from harness_evolve import bench

    kats = [s for s in bench_spec.scenarios() if s.expect is not None]
    assert [s.name for s in kats] == ["accept-good", "reject-bad", "reject-noise", "accept-threshold"]
    results = asyncio.run(bench.run(kats, tmp_path / "bench"))
    by = {r["name"]: r for r in results}
    assert by["accept-good"]["passed"] is True and by["accept-good"]["accepted"] == ["r1"]
    assert by["accept-good"]["rounds"][0]["delta_S"] > 0.5            # 0% -> 100% in the fake
    assert by["reject-bad"]["passed"] is True and by["reject-bad"]["accepted"] == []
    assert by["reject-noise"]["passed"] is True and by["reject-noise"]["accepted"] == []
    assert by["accept-threshold"]["passed"] is True and by["accept-threshold"]["rounds"][0]["delta_S"] > 0.5
    md = (tmp_path / "bench" / "bench.md").read_text()
    assert "| accept-good | accepted | PASS | ['r1']" in md


def test_partial_rerun_updates_instead_of_replacing(bench_spec, tmp_path):
    from harness_evolve import bench

    kats = [s for s in bench_spec.scenarios() if s.expect is not None]
    out = tmp_path / "bench"
    asyncio.run(bench.run(kats, out))
    import shutil
    shutil.rmtree(out / "reject-bad")
    asyncio.run(bench.run(kats, out, only=["reject-bad"]))
    names = [r["name"] for r in json.loads((out / "bench.json").read_text())]
    assert names == ["accept-good", "reject-bad", "reject-noise", "accept-threshold"]


def test_judge_rules():
    from harness_evolve.bench import ACCEPTED, NOT_ACCEPTED, judge

    done = {"phase": "done", "accepted": ["r1"]}
    assert judge(ACCEPTED, done) is True and judge(NOT_ACCEPTED, done) is False
    assert judge(None, done) is None
    assert judge(ACCEPTED, {"phase": "propose", "accepted": ["r1"]}) is False   # unfinished run fails


def test_scripted_llm_answers_proposer_and_critic():
    from harness_evolve.bench import scripted_llm

    llm = scripted_llm({"component": "settings", "hypothesis": "h", "edits": []}, critic_accepts=False)
    assert json.loads(asyncio.run(llm("You are a strict reviewer ...", "x")))["verdict"] == "reject"
    assert json.loads(asyncio.run(llm("You improve the harness ...", "x")))["component"] == "settings"


def test_known_answers_hold_with_confirm(bench_spec, tmp_path, monkeypatch):
    from harness_evolve import bench

    monkeypatch.setattr(bench_spec, "CONFIRM", True)
    kats = [s for s in bench_spec.scenarios() if s.expect is not None]
    by = {r["name"]: r for r in asyncio.run(bench.run(kats, tmp_path / "bench"))}
    assert all(r["passed"] for r in by.values())
    assert "confirmed on fresh trials" in by["accept-good"]["rounds"][0]["reason"]


def test_noise_many_runs_every_round_and_a_deterministic_agent_accepts_nothing(bench_spec, tmp_path):
    from harness_evolve import bench

    (sc,) = [s for s in bench_spec.scenarios() if s.name == "noise-many"]
    (r,) = asyncio.run(bench.run([sc], tmp_path / "bench"))
    assert r["phase"] == "done" and len(r["rounds"]) == 40 and r["accepted"] == []
    assert len({x["hypothesis"] for x in r["rounds"]}) == 40
