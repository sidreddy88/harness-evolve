"""CommandEvaluator and `harness-evolve init`: an agent in any language, measured through a command,
and a starter project that runs end to end once the TODOs are filled in. Offline: the agent is a script."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import textwrap

import pytest

from harness_evolve import cli, evidence, profiles
from harness_evolve.bench import scripted_llm
from harness_evolve.command import CommandEvaluator
from harness_evolve.evaluator import ProviderFailure

AGENT = textwrap.dedent('''
    import json, os, sys, time
    harness, case = sys.argv[1], sys.argv[2]
    assert os.environ["HE_CASE_ID"] == case and os.environ["HE_HARNESS_DIR"] == harness
    if case == "boom":
        sys.exit("agent crashed")
    if case == "slow":
        time.sleep(5)
    if case == "garbage":
        print("not json")
        sys.exit(0)
    if case == "billing":
        print(json.dumps({"passed": False, "cost_usd": 0.0, "provider_failure": "credit balance too low"}))
        sys.exit(0)
    print("some log line")
    print(json.dumps({"passed": case == "good", "cost_usd": 0.01, "detail": "answered 7",
                      "answer": "expected 7", "steps": [{"name": "search", "input": {"q": case}, "output": "ok"}]}))
''')


def _evaluator(tmp_path, **kw):
    script = tmp_path / "agent.py"
    script.write_text(AGENT)
    return CommandEvaluator([sys.executable, str(script), "{harness}", "{case}"], **kw)


def test_results_errors_and_hidden_answers(tmp_path):
    out = asyncio.run(_evaluator(tmp_path, timeout_s=2).evaluate(tmp_path, ["good", "bad", "boom", "garbage",
                                                                             "slow"], 2))
    r = out.result.per_case
    assert r["good"].passes == 2 and r["bad"].passes == 0 and r["bad"].escalations == 2
    assert r["boom"].verdicts == ["ERROR", "ERROR"] and r["garbage"].verdicts == ["ERROR", "ERROR"]
    assert r["slow"].verdicts == ["TIMEOUT", "TIMEOUT"]
    assert out.cost_usd == pytest.approx(0.04)
    by = {(t["instance_id"], t["trial"]): t for t in out.trajectories}
    assert "agent crashed" in by[("boom", 1)]["detail"] and by[("good", 2)]["steps"][0]["name"] == "search"
    assert "expected 7" not in evidence.outcome(by[("good", 1)])          # the answer is hidden


def test_provider_failure_stops_the_run(tmp_path):
    with pytest.raises(ProviderFailure, match="credit"):
        asyncio.run(_evaluator(tmp_path).evaluate(tmp_path, ["billing"], 1))
    with pytest.raises(TypeError):
        CommandEvaluator("python run_case.py")


def test_init_writes_a_starter_and_refuses_to_overwrite(tmp_path, capsys):
    assert cli.main(["init", str(tmp_path / "proj")]) == 0
    files = {p.relative_to(tmp_path / "proj").as_posix() for p in (tmp_path / "proj").rglob("*") if p.is_file()}
    assert {"README.md", "config.py", "run_case.py", "cases.jsonl", "harness/settings.json",
            "harness/system.prompt", "harness/task.prompt", "harness/tool_descriptions.json"} <= files
    readme = (tmp_path / "proj" / "README.md").read_text()
    assert "# proj:" in readme and "`{question}`" in readme and "{{" not in readme
    assert cli.main(["init", str(tmp_path / "proj")]) == 1
    assert cli.main(["init", str(tmp_path / "proj"), "--force"]) == 0


def test_starter_runs_end_to_end_once_filled_in(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    cli.main(["init", str(proj)])
    # The user's work: real cases, their agent (here a fake that answers right only when told to
    # be careful), and a description in config.py.
    (proj / "cases.jsonl").write_text("".join(json.dumps({"id": f"c{i}", "question": "q", "expected": "x"}) + "\n"
                                              for i in range(12)))
    (proj / "run_case.py").write_text(textwrap.dedent('''
        import json, sys
        from pathlib import Path
        if sys.argv[1] == "--grade":
            print(json.dumps({"passed": Path(sys.argv[3]).read_text() == "x"}))
            sys.exit(0)
        careful = "carefully" in (Path(sys.argv[1]) / "system.prompt").read_text()
        print(json.dumps({"passed": careful, "cost_usd": 0.001}))
    '''))
    (proj / "anchors.jsonl").write_text('{"case": "c0", "output": "x", "passed": true}\n'
                                        '{"case": "c0", "output": "y", "passed": false}\n')
    cfg = (proj / "config.py").read_text().replace('"TODO: describe your agent"', '"Answers questions."')
    (proj / "config.py").write_text(cfg)
    monkeypatch.setenv("MODEL", "some/model")
    previous = profiles._active
    try:
        spec = importlib.util.spec_from_file_location("starter_config", proj / "config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        opt = mod.build(tmp_path / "run", budget_usd=1.0, rounds=1)
        edit = {"component": "prompts", "hypothesis": "ask for care",
                "edits": [{"file": "system.prompt", "find": "answers questions accurately",
                           "replace": "answers questions carefully and accurately"}]}
        opt.proposer_llm = opt.critic_llm = scripted_llm(edit)
        opt.cfg.allow_unpriced_llm = True
        state = asyncio.run(opt.run_until_stopped())
    finally:
        profiles._active = previous
    assert state.phase == "done" and state.accepted == ["r1"]
    assert state.health["anchors"]["problems"] == []          # 2 anchors + an automatic blank, all right


def test_starter_anchors_stop_a_grader_that_passes_everything(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    cli.main(["init", str(proj)])
    (proj / "cases.jsonl").write_text("".join(json.dumps({"id": f"c{i}", "question": "q", "expected": "x"}) + "\n"
                                              for i in range(12)))
    (proj / "run_case.py").write_text('import json; print(json.dumps({"passed": True, "cost_usd": 0.001}))\n')
    (proj / "anchors.jsonl").write_text('{"case": "c0", "output": "x", "passed": true}\n'
                                        '{"case": "c0", "output": "y", "passed": false}\n')
    cfg = (proj / "config.py").read_text().replace('"TODO: describe your agent"', '"Answers questions."')
    (proj / "config.py").write_text(cfg)
    monkeypatch.setenv("MODEL", "some/model")
    previous = profiles._active
    try:
        spec = importlib.util.spec_from_file_location("starter_config3", proj / "config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        opt = mod.build(tmp_path / "run", budget_usd=1.0, rounds=1)
        opt.proposer_llm = opt.critic_llm = scripted_llm({})
        state = asyncio.run(opt.run_until_stopped())
    finally:
        profiles._active = previous
    assert state.phase == "anchors" and "grader anchors" in state.stop_reason
    assert "'y' should fail but passed" in state.stop_reason and "a blank output should fail" in state.stop_reason
    assert not (tmp_path / "run" / "evals").exists()                 # no agent ran
    assert state.spent_usd == pytest.approx(0.003)                       # grading 3 outputs is counted


def test_starter_config_refuses_to_run_with_the_todo_description(tmp_path, monkeypatch):
    cli.main(["init", str(tmp_path / "proj")])
    previous = profiles._active
    try:
        spec = importlib.util.spec_from_file_location("starter_config2", tmp_path / "proj" / "config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with pytest.raises(SystemExit, match="describe your agent"):
            mod.build(tmp_path / "run")
    finally:
        profiles._active = previous
