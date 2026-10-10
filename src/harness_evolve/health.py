"""
Tripwires: stop a long run the moment its measurements stop being trustworthy.

The worst loss in this project so far wasn't a crash. It was a broken tool
that looked like data: verify_symbol_in_repo returned NOT_FOUND on 87 of 87
calls (an invalid GitHub token, swallowed as "no results"), and two whole
runs were scored on it before anyone read the traces. An 8-hour unattended
run can spend its budget the same way. So while cases finish, the loop checks
the recent trajectories against what a healthy run looks like and pauses
(resumably, like a provider failure) when something is off:

- a tool's failure rate (errors, NOT_FOUND, empty results) far above normal
- replays erroring or timing out instead of producing a verdict
- every guard case failing (the always-pass cases: the easy path is broken)
- cost per trial far above normal, or zero (a price missing from the meter)
- the pass rate collapsing, in round 0 only (a candidate is allowed to be
  bad; that's what the acceptance rules are for)

"Normal" comes from the profile's health_baseline (measure it on a pilot round
0 of your own agent). Every check is skipped when its baseline is missing.
Thresholds are deliberately loose: a tripwire that fires on noise would end
unattended runs for nothing. They're sized to catch breakage, which in every
case seen so far was all-or-nothing (a tool failing on every call, a price
missing from the meter), not drift.

health_baseline keys (all optional):
    tool_failure_rate     {tool: normal failure rate}, e.g. {"search": 0.12}
    failure_prefixes      {tool: [output prefixes that mean the call failed]};
                          outputs starting with "Error" always count
    cost_per_trial_usd    normal cost of one trial
    pass_rate             normal round-0 pass rate
"""
from __future__ import annotations

from dataclasses import dataclass

BASELINE: dict = {}

MIN_CALLS = 15          # per tool, before its failure rate is judged
MIN_REPLAYS = 10


class Tripwire(RuntimeError):
    """Measurements look broken; the run pauses so a human can look."""


def _tool_failed(name: str, output: str, prefixes: dict | None = None) -> bool:
    out = output.lstrip()
    if out.startswith("Error"):
        return True
    return any(out.startswith(p) for p in (prefixes or {}).get(name, ()))


@dataclass
class Window:
    """The most recent finished replays of one evaluation."""
    trajectories: list[dict]
    verdicts: list[str]
    costs: list[float]              # per trial
    guard_verdicts: list[str]


def check(w: Window, round0: bool, baseline: dict = BASELINE) -> list[str]:
    problems: list[str] = []
    calls: dict[str, list[bool]] = {}
    for t in w.trajectories:
        for st in t.get("steps") or []:
            calls.setdefault(st.get("name", "?"), []).append(
                _tool_failed(st.get("name", ""), str(st.get("output") or ""), baseline.get("failure_prefixes")))
    for tool, normal in (baseline.get("tool_failure_rate") or {}).items():
        fails = calls.get(tool, [])
        if len(fails) >= MIN_CALLS:
            rate = sum(fails) / len(fails)
            limit = max(0.5, normal + 0.35)
            if rate >= limit:
                problems.append(f"{tool} failing on {sum(fails)}/{len(fails)} recent calls "
                                f"({rate:.0%}; normal {normal:.0%}, limit {limit:.0%})")
    n = len(w.verdicts)
    if n >= MIN_REPLAYS:
        broken = sum(v in ("ERROR", "TIMEOUT") for v in w.verdicts)
        if broken / n > 0.10:
            problems.append(f"{broken}/{n} recent replays ended in ERROR/TIMEOUT, not a verdict")
        mean_cost = sum(w.costs) / len(w.costs) if w.costs else 0.0
        if mean_cost == 0.0:
            problems.append(f"{n} recent replays cost $0: a model price is missing or the meter is off")
        elif baseline.get("cost_per_trial_usd") and mean_cost > 3 * baseline["cost_per_trial_usd"]:
            problems.append(f"mean cost ${mean_cost:.3f}/trial, over 3x normal "
                            f"(${baseline['cost_per_trial_usd']:.3f})")
        if round0 and n >= 20 and baseline.get("pass_rate"):
            rate = sum(v == "PASS" for v in w.verdicts) / n
            if rate < 0.5 * baseline["pass_rate"]:
                problems.append(f"round 0 pass rate {rate:.0%} on {n} recent replays, under half "
                                f"of the pilot's {baseline['pass_rate']:.0%}: the original harness "
                                f"shouldn't change, so something else did")
    if len(w.guard_verdicts) >= 4 and not any(v == "PASS" for v in w.guard_verdicts):
        problems.append(f"all {len(w.guard_verdicts)} guard-case trials failed")
    return problems


def smoke_problems(trajectories: list[dict], verdicts: list[str], costs: list[float],
                   baseline: dict | None = None) -> list[str]:
    """Stricter checks on the handful of smoke replays that gate a long run:
    every replay must have produced tool calls, a verdict and a cost."""
    problems = []
    for t in trajectories:
        if not t.get("steps"):
            problems.append(f"{t.get('instance_id')} trial {t.get('trial')}: no tool calls recorded")
    for v in verdicts:
        if v not in ("PASS", "FAIL"):
            problems.append(f"a smoke replay ended {v}")
    if any(c <= 0 for c in costs):
        problems.append("a smoke replay cost $0: price missing or meter off")
    calls = {}
    for t in trajectories:
        for st in t.get("steps") or []:
            calls.setdefault(st.get("name", "?"), []).append(
                _tool_failed(st.get("name", ""), str(st.get("output") or ""), (baseline or {}).get("failure_prefixes")))
    for tool, fails in calls.items():
        if len(fails) >= 3 and all(fails):
            problems.append(f"{tool} failed on all {len(fails)} smoke calls")
    return problems
