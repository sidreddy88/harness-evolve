"""Every test runs with a profile shaped like the first agent this optimizer
improved (a code-diagnosis agent): its harness is the fixture under
fixtures/diagnosis_harness/, with the same components, setting bounds and
health baselines, so the ported tests check the optimizer's mechanics unchanged."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness_evolve import profiles
from harness_evolve.grader import TrajectoryGrader
from harness_evolve.profiles import AgentProfile

FIXTURE_HARNESS = Path(__file__).parent / "fixtures" / "diagnosis_harness"

COMPONENT_FILES = {
    "task_prompt": lambda f: f == "task_prompt.prompt",
    "prompt_fragments": lambda f: f.endswith(".prompt") and f != "task_prompt.prompt",
    "tool_descriptions": lambda f: f == "tool_descriptions.json",
    "settings": lambda f: f == "settings.json",
}
SETTING_BOUNDS = {
    "max_iterations": (5, 25),
    "retry_on_no_submission": (0, 2),
    "file_read_char_limit": (2000, 60000),
    "grep_max_matches": (10, 200),
}
HEALTH_BASELINE = {
    "tool_failure_rate": {"verify_symbol_in_repo": 0.0, "find_callers": 0.04,
                          "grep_codebase": 0.12, "get_file_contents": 0.03},
    "failure_prefixes": {"verify_symbol_in_repo": ["NOT_FOUND", "VERIFY_ERROR"],
                         "find_callers": ["No callers found"], "grep_codebase": ["No matches"]},
    "cost_per_trial_usd": 0.148,
    "pass_rate": 0.657,
}


def diagnosis_profile() -> AgentProfile:
    return AgentProfile.build(
        name="diagnosis", harness_root=FIXTURE_HARNESS,
        agent_description=("DiagnosisAgent diagnoses the root cause of software incidents with "
                           "repository tools (read files, grep, search, verify symbols) and finishes "
                           "by calling submit_diagnosis with verbatim evidence from the code."),
        safety_mechanisms=["grounding claims in verbatim code", "verifying symbols before naming them",
                           "calling submit_diagnosis"],
        setting_bounds=SETTING_BOUNDS, component_files=COMPONENT_FILES,
        component_families={"task_prompt": "prompt", "prompt_fragments": "prompt",
                            "tool_descriptions": "tools", "settings": "settings"},
        grader=TrajectoryGrader(), health_baseline=HEALTH_BASELINE)


@pytest.fixture(autouse=True)
def _profile():
    profiles.use(diagnosis_profile())
    yield
