"""`harness-evolve check`: free pre-run checks. Offline: no model calls, no agent runs."""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import textwrap

import pytest

from harness_evolve import check, cli, litellm_llm, profiles
from harness_evolve.acceptance import calibrate_delta
from harness_evolve.bench import scripted_llm
from harness_evolve.loop import Optimizer, OptimizerConfig

from .conftest import FIXTURE_HARNESS


class Grades:
    """An evaluator that can grade an output alone: passes it if it is "right"."""

    async def grade(self, case_id, output):
        return {"passed": output == "right", "cost_usd": 0.0}


ANCHORS = [{"case": "c0", "output": "right", "passed": True}, {"case": "c0", "output": "wrong", "passed": False}]


@pytest.fixture(autouse=True)
def _contract():
    profiles.use(dataclasses.replace(profiles.active(), task_contract="the diagnosis names the right file"))


def _opt(tmp_path, harness=FIXTURE_HARNESS, proposer=None, evaluator=None, **cfg):
    ids = [f"c{i}" for i in range(60)]
    base = dict(evolve_cases=ids[:40], guard_cases=[], final_cases=ids[40:], smoke_cases=ids[:3], anchors=ANCHORS,
                budget_usd=100.0, default_case_cost_usd=0.05, max_rounds=3, trials=2,
                health_baseline={"tool_failure_rate": {"grep_codebase": 0.1}, "cost_per_trial_usd": 0.05,
                                 "pass_rate": 0.6})
    llm = proposer or scripted_llm({})
    return Optimizer(tmp_path / "run", harness, OptimizerConfig(**{**base, **cfg}), evaluator=evaluator or Grades(),
                     proposer_llm=llm, critic_llm=llm, critic_patterns=[])


def _levels(findings, name):
    return [f.level for f in findings if f.name == name]


def test_healthy_config_passes(tmp_path):
    findings = check.run(_opt(tmp_path))
    assert check.exit_code(findings) == 0
    assert check.exit_code(findings, strict=True) == 0, check.render(findings)
    assert not (tmp_path / "run").exists()                      # checking writes nothing


def test_heldout_overlap_fails(tmp_path):
    ids = [f"c{i}" for i in range(60)]
    findings = check.run(_opt(tmp_path, evolve_cases=ids[:40], final_cases=ids[35:]))
    assert check.FAIL in _levels(findings, "cases") and check.exit_code(findings) == 1
    assert "5 held-out" in check.render(findings)


def test_no_evolve_cases_fails(tmp_path):
    assert _levels(check.run(_opt(tmp_path, evolve_cases=[], smoke_cases=[])), "cases") == [check.FAIL]


def test_small_or_unheld_case_sets_warn(tmp_path):
    findings = check.run(_opt(tmp_path, evolve_cases=["a", "b", "c"], final_cases=[], smoke_cases=[]))
    text = check.render(findings)
    assert "only 3 evolve cases" in text and "no held-out cases" in text
    assert check.exit_code(findings) == 0 and check.exit_code(findings, strict=True) == 1


def test_budget_that_cant_finish_round0_fails(tmp_path):
    findings = check.run(_opt(tmp_path, budget_usd=1.0))       # round 0 alone: 40 x 2 x $0.05 x 1.25 = $5
    assert _levels(findings, "budget") == [check.FAIL]


def test_budget_short_of_all_rounds_warns(tmp_path):
    # smoke $0.15 + round 0 $4 + held-out $6 + $4/round, all x1.25: $20 leaves room for 1 of 3 rounds
    msg = [f.message for f in check.run(_opt(tmp_path, budget_usd=20.0)) if f.name == "budget"][0]
    assert msg.startswith("$20.00 covers ~1 of 3 rounds")


def test_default_case_cost_warns(tmp_path):
    findings = check.run(_opt(tmp_path, default_case_cost_usd=OptimizerConfig.default_case_cost_usd,
                              budget_usd=1000.0))
    assert any("built-in" in f.message for f in findings if f.level == check.WARN)


@pytest.mark.parametrize("n,trials,flips", [(50, 1, 0), (50, 2, 0), (100, 1, 0.04), (200, 2, 0.06)])
def test_band_matches_calibration(n, trials, flips):
    """The band check predicts is the one round 0 would calibrate (bootstrap noise, within ~15%)."""
    k = round(n * flips)
    pairs = {f"c{i}": ((True, False) if i < k // 2 else (False, True) if i < k else (True, True))
             for i in range(n)}
    got = calibrate_delta(pairs, trials_per_eval=trials)
    assert check.band(n, trials, k / n) == pytest.approx(got, rel=0.15)


def test_bad_starting_harness_fails(tmp_path):
    h = tmp_path / "h"
    shutil.copytree(FIXTURE_HARNESS, h)
    s = json.loads((h / "settings.json").read_text())
    s["max_iterations"] = 99                                     # bounds (5, 25)
    (h / "settings.json").write_text(json.dumps(s))
    assert _levels(check.run(_opt(tmp_path, harness=h)), "harness") == [check.FAIL]
    assert _levels(check.run(_opt(tmp_path, harness=tmp_path / "missing")), "harness") == [check.FAIL]


def test_unbounded_setting_warns(tmp_path):
    h = tmp_path / "h"
    shutil.copytree(FIXTURE_HARNESS, h)
    s = json.loads((h / "settings.json").read_text())
    s["temperature_x100"] = 70
    (h / "settings.json").write_text(json.dumps(s))
    assert "temperature_x100" in check.render(check.run(_opt(tmp_path, harness=h)))


def test_missing_baseline_and_smoke_warn(tmp_path):
    profiles.use(dataclasses.replace(profiles.active(), health_baseline={}))
    findings = check.run(_opt(tmp_path, health_baseline={}, smoke_cases=[]))
    assert _levels(findings, "tripwires") == [check.WARN] and _levels(findings, "smoke stage") == [check.WARN]


def test_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "_litellm_price", lambda model: model == "known/model")
    priced = litellm_llm("unknown/model", prices=(0.3, 1.2))
    assert _levels(check.run(_opt(tmp_path, proposer=priced)), "prices") == [check.OK, check.OK]
    assert _levels(check.run(_opt(tmp_path, proposer=litellm_llm("known/model"))), "prices") == [check.OK] * 2
    unpriced = litellm_llm("unknown/model")
    assert _levels(check.run(_opt(tmp_path, proposer=unpriced)), "prices") == [check.FAIL] * 2
    assert _levels(check.run(_opt(tmp_path, proposer=unpriced, allow_unpriced_llm=True)),
                   "prices") == [check.WARN] * 2


CONFIG = textwrap.dedent('''
    from pathlib import Path
    from harness_evolve import AgentProfile, Optimizer, OptimizerConfig, profiles
    from harness_evolve.bench import scripted_llm

    def build(run_dir, budget_usd=1.0, rounds=3):
        profiles.use(AgentProfile.build(name="t", harness_root=Path({harness!r}), agent_description="t",
                                        setting_bounds={{"max_iterations": (5, 25), "retry_on_no_submission": (0, 2),
                                                        "file_read_char_limit": (2000, 60000),
                                                        "grep_max_matches": (10, 200)}}))
        ids = [f"c{{i}}" for i in range(60)]
        cfg = OptimizerConfig(evolve_cases=ids[:40], guard_cases=[], final_cases=ids[40:], budget_usd=budget_usd,
                              max_rounds=rounds, default_case_cost_usd=0.01)
        llm = scripted_llm({{}})
        return Optimizer(run_dir, Path({harness!r}), cfg, evaluator=None, proposer_llm=llm, critic_llm=llm,
                         critic_patterns=[])
''')


def test_cli(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "config.py"
    cfg.write_text(CONFIG.format(harness=str(FIXTURE_HARNESS)))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["check", str(cfg), "--budget", "0.01"]) == 1          # can't finish round 0
    assert "FAIL  budget" in capsys.readouterr().out
    assert cli.main(["check", str(cfg), "--budget", "50"]) == 0
    assert cli.main(["check", str(cfg), "--budget", "50", "--strict"]) == 1  # no baseline, no smoke stage
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "__pycache__") == ["config.py"]  # nothing written
    sys.modules.pop("harness_evolve_config", None)


def test_tiny_case_set_says_nothing_can_be_accepted(tmp_path):
    findings = check.run(_opt(tmp_path, evolve_cases=["a", "b"], final_cases=[], smoke_cases=[], trials=1))
    assert "no gain can be accepted" in check.render(findings)


def test_grader_findings(tmp_path):
    assert _levels(check.run(_opt(tmp_path)), "grader") == [check.OK]
    assert _levels(check.run(_opt(tmp_path, anchors=[])), "grader") == [check.WARN]
    assert _levels(check.run(_opt(tmp_path, anchors=[{"case": "c0", "output": 3}])), "grader") == [check.FAIL]

    class NoGrade:
        pass
    assert _levels(check.run(_opt(tmp_path, evaluator=NoGrade())), "grader") == [check.FAIL]
    text = check.render(check.run(_opt(tmp_path, anchors=[{"case": "zz", "output": "x", "passed": False}],
                                       always_pass_tripwire=None)))
    assert "['zz']" in text and "no anchor expects a pass" in text and "always_pass_tripwire is off" in text
    profiles.use(dataclasses.replace(profiles.active(), task_contract=""))
    assert "no task_contract" in check.render(check.run(_opt(tmp_path)))
