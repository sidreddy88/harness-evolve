"""The math-agent example end to end, offline: litellm is replaced by a scripted
model, the proposer by a fixed edit. Checks the wiring of agent, evaluator,
profile, config and CLI without spending anything."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "math_agent"


def _resp(tool_calls=None, content=""):
    calls = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)),
                             model_dump=lambda n=n, a=a, i=i: {"id": f"c{i}", "type": "function",
                                                               "function": {"name": n, "arguments": json.dumps(a)}})
             for i, (n, a) in enumerate(tool_calls or [])]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
                           usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100))


@pytest.fixture
def example(tmp_path, monkeypatch):
    pytest.importorskip("litellm")      # the example's agent uses litellm; the core package doesn't
    problems = [{"id": f"p{i}", "question": f"What is {i} + {i}?", "answer": 2 * i} for i in range(60)]
    data = tmp_path / "problems.jsonl"
    data.write_text("".join(json.dumps(p) + "\n" for p in problems))
    monkeypatch.syspath_prepend(str(EXAMPLE))
    for m in ("agent", "evaluator", "config"):
        sys.modules.pop(m, None)
    import config
    monkeypatch.setattr(config, "PROBLEMS", data)

    async def fake_acompletion(model, messages, tools, **kw):
        last = messages[-1]
        q = next(m["content"] for m in messages if m["role"] == "user")
        i = int(q.split("What is ")[1].split(" +")[0])
        if last["role"] == "tool":
            # Tool descriptions saying "every step" make the scripted model careful.
            careful = "every" in json.dumps(tools)
            right = 2 * i if (careful or i % 3) else 2 * i + 1
            return _resp([("submit_answer", {"answer": right})])
        return _resp([("calculator", {"expression": f"{i}+{i}"})])

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async def proposer(system, prompt):
        from harness_evolve.llm import LLMReply
        if "verdict" in system:     # the critic
            return LLMReply('{"verdict": "accept", "reasons": []}', 0.001)
        return LLMReply(json.dumps({"component": "tool_descriptions", "hypothesis": "use the calculator for every step",
                                    "edits": [{"file": "tool_descriptions.json",
                                               "find": "Evaluate an arithmetic expression.",
                                               "replace": "Evaluate an arithmetic expression; "
                                                          "use it for every step."}]}),
                        0.001)
    monkeypatch.setattr(config, "litellm_llm", lambda *a, **k: proposer)
    return config


def test_example_runs_end_to_end_offline(example, tmp_path, capsys):
    from harness_evolve.cli import _status

    opt = example.build(tmp_path / "run", budget_usd=5.0, rounds=1)
    import asyncio
    state = asyncio.run(opt.run_until_stopped())
    assert state.phase == "done", state.stop_reason
    assert state.accepted == ["r1"]                         # the careful wording fixed 1 in 3 problems
    text = _status(tmp_path / "run")
    assert "accepted: ['r1']" in text and "tool_descriptions" in text
    report = json.loads((tmp_path / "run" / "report" / "report.json").read_text())
    assert report["evolved"]["cases"] == 10


def test_cli_status_and_diff_on_a_missing_run(tmp_path, capsys):
    from harness_evolve.cli import main

    assert main(["status", str(tmp_path / "nope")]) == 0
    assert "no run at" in capsys.readouterr().out
    assert main(["diff", str(tmp_path / "nope")]) == 0
    assert "no accepted edits" in capsys.readouterr().out


def test_stuck_detection_reminds_then_stops(example, tmp_path, monkeypatch):
    import shutil

    import agent
    import litellm

    async def keeps_calculating(model, messages, tools, **kw):
        return _resp([("calculator", {"expression": "1+1"})])
    monkeypatch.setattr(litellm, "acompletion", keeps_calculating)
    h = tmp_path / "h"
    shutil.copytree(EXAMPLE / "harness", h)
    s = json.loads((h / "settings.json").read_text())
    s.update(remind_after=2, stop_after=3, max_turns=8)
    (h / "settings.json").write_text(json.dumps(s))
    run = asyncio.run(agent.solve("What is 1 + 1?", h, "m", (0.0, 0.0)))
    outs = [st["output"] for st in run.steps]
    assert run.answer is None and run.turns == 4 and len(outs) == 4
    assert "submit_answer" not in outs[0] and "2 times" in outs[1] and outs[3].startswith("Stopped")


def test_switches_plan_adds_a_call_and_calculator_off_removes_the_tool(example, tmp_path, monkeypatch):
    import shutil

    import agent
    import litellm

    seen = []

    async def fake(model, messages, tools=None, **kw):
        seen.append(tools)
        if tools is None:
            return _resp(content="1. add the numbers")
        return _resp([("submit_answer", {"answer": 2})])
    monkeypatch.setattr(litellm, "acompletion", fake)
    h = tmp_path / "h"
    shutil.copytree(EXAMPLE / "harness", h)
    s = json.loads((h / "settings.json").read_text())
    s.update(plan="on", calculator="off")
    (h / "settings.json").write_text(json.dumps(s))
    run = asyncio.run(agent.solve("What is 1 + 1?", h, "m", (0.0, 0.0)))
    assert run.answer == 2 and run.turns == 2 and run.steps[0]["name"] == "plan"
    assert seen[0] is None and [t["function"]["name"] for t in seen[1]] == ["submit_answer"]
