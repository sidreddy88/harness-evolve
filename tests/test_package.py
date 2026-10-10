"""The parts written for the standalone package: harness loader, priced
replies, generic grader, profile errors, minimal harness directories."""
from __future__ import annotations

import json

import pytest

from harness_evolve import candidates, health, profiles
from harness_evolve.grader import TrajectoryGrader
from harness_evolve.harness import load_harness
from harness_evolve.llm import LLMReply, reply_cost
from harness_evolve.profiles import AgentProfile


def _harness(tmp_path, settings=None, tools=None, prompts=None):
    d = tmp_path / "h"
    d.mkdir()
    if settings is not None:
        (d / "settings.json").write_text(json.dumps(settings))
    if tools is not None:
        (d / "tool_descriptions.json").write_text(json.dumps(tools))
    for name, text in (prompts or {}).items():
        (d / f"{name}.prompt").write_text(text)
    return d


def test_load_harness_renders_templates_and_hides_doc_keys(tmp_path):
    d = _harness(tmp_path, {"_doc": {"max_turns": "turn budget"}, "max_turns": 8},
                 {"search": "Search the docs."}, {"task": "Answer: {question}"})
    h = load_harness(d)
    assert h.setting("max_turns") == 8 and "_doc" not in h.settings
    assert h.render("task", question="why?") == "Answer: why?"
    assert h.tool_description("search") == "Search the docs."
    with pytest.raises(KeyError):
        h.render("missing")


def test_a_prompt_only_harness_validates(tmp_path):
    base = _harness(tmp_path, prompts={"task": "Answer: {question}"})
    cand = tmp_path / "c"
    candidates.apply_edit(base, cand, {"task.prompt": "Answer briefly: {question}"})
    candidates.validate(base, cand)
    bad = tmp_path / "bad"
    candidates.apply_edit(base, bad, {"task.prompt": "Answer: {question} {secret}"})
    with pytest.raises(candidates.InvalidCandidate, match="placeholders"):
        candidates.validate(base, bad)


def test_setting_choices_and_ranges_come_from_the_profile(tmp_path):
    base = _harness(tmp_path, {"mode": "fast", "max_turns": 8}, prompts={"task": "x"})
    profiles.use(AgentProfile.build(name="t", harness_root=base, agent_description="T.",
                                    setting_bounds={"mode": frozenset({"fast", "careful"}),
                                                    "max_turns": (2, 20)}))
    for bad_settings, msg in (({"mode": "yolo", "max_turns": 8}, "not one of"),
                              ({"mode": "fast", "max_turns": 99}, "outside")):
        cand = tmp_path / f"c{msg[:3]}"
        candidates.apply_edit(base, cand, {"settings.json": json.dumps(bad_settings)})
        with pytest.raises(candidates.InvalidCandidate, match=msg):
            candidates.validate(base, cand)


def test_llm_reply_is_a_str_that_carries_its_cost():
    r = LLMReply("hello", cost_usd=0.002)
    assert r == "hello" and r.upper() == "HELLO" and reply_cost(r) == 0.002
    assert reply_cost("plain") is None


def test_generic_grader_counts_repeats_errors_and_submission():
    g = TrajectoryGrader()
    rec = {"instance_id": "a", "verdict": "FAIL", "submitted": False,
           "steps": [{"name": "search", "input": "q", "output": "x"},
                     {"name": "search", "input": "q", "output": "x"},
                     {"name": "read", "input": "f", "output": "Error: missing"}]}
    grade = g.grade(rec)
    assert (grade.calls, grade.redundant_calls, grade.tool_errors, grade.accepted) == (3, 1, 1, False)
    assert "1 never submitted" in g.render(g.summarize([grade]))


def test_health_checks_skip_what_the_baseline_does_not_define():
    w = health.Window(trajectories=[], verdicts=["FAIL"] * 20, costs=[0.5] * 20, guard_verdicts=[])
    assert health.check(w, round0=True, baseline={}) == []
    probs = health.check(w, round0=True, baseline={"cost_per_trial_usd": 0.1, "pass_rate": 0.6})
    assert any("over 3x normal" in p for p in probs) and any("round 0 pass rate" in p for p in probs)


def test_no_profile_is_an_explicit_error(monkeypatch):
    monkeypatch.setattr(profiles, "_active", None)
    with pytest.raises(RuntimeError, match="no agent profile"):
        profiles.active()
