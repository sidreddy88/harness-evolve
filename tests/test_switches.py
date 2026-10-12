"""Structural switches: declared in the profile, shown to the proposer as behaviour, validated."""
from __future__ import annotations

import json
import shutil

import pytest

from harness_evolve import AgentProfile, candidates, profiles, proposer

SWITCHES = {"plan": {"off": "solves directly", "on": "writes a plan first (one extra call)"}}


def _harness(tmp_path, **settings):
    d = tmp_path / "h"
    d.mkdir()
    s = {"_doc": {"plan": "whether the agent plans first"}, "max_turns": 8, "plan": "off", **settings}
    (d / "settings.json").write_text(json.dumps(s))
    (d / "system.prompt").write_text("Solve it.")
    return d


@pytest.fixture
def profile(tmp_path):
    previous = profiles._active
    p = profiles.use(AgentProfile.build(name="t", harness_root=tmp_path, agent_description="An agent.",
                                        setting_bounds={"max_turns": (1, 10)}, switches=SWITCHES))
    yield p
    profiles._active = previous


def test_switch_options_become_the_allowed_values_and_the_proposer_is_told(profile):
    assert profile.setting_bounds["plan"] == frozenset({"off", "on"})
    assert "STRUCTURAL SWITCHES" in profile.proposer_system
    plain = AgentProfile.build(name="t", harness_root=".", agent_description="An agent.")
    assert "STRUCTURAL SWITCHES" not in plain.proposer_system and "{switches}" not in plain.proposer_system


def test_settings_table_shows_each_option_as_behaviour(profile, tmp_path):
    table = proposer.settings_table(_harness(tmp_path))
    line = next(x for x in table.splitlines() if x.startswith("- plan"))
    assert "[STRUCTURAL SWITCH]" in line and "'on': writes a plan first" in line
    assert "- max_turns = 8, allowed 1-10" in table


def test_an_unknown_switch_option_is_invalid(profile, tmp_path):
    base = _harness(tmp_path)
    cand = tmp_path / "c"
    shutil.copytree(base, cand)
    s = json.loads((cand / "settings.json").read_text())
    s["plan"] = "sometimes"
    (cand / "settings.json").write_text(json.dumps(s))
    with pytest.raises(candidates.InvalidCandidate, match="plan"):
        candidates.validate(base, cand)
    s["plan"] = "on"
    (cand / "settings.json").write_text(json.dumps(s))
    candidates.validate(base, cand)
