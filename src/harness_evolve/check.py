"""
`harness-evolve check CONFIG.py`: can this run measure anything, before it spends a cent?

Builds the optimizer from the config (into a throwaway run directory) and reads its settings, the
agent profile and the starting harness. It makes no model calls and runs no agent, so it is free and
gives the same answer every time; it exits 1 on any FAIL (and, with --strict, on any WARN), so CI can
run it before a scheduled run. Each check is a mistake a real run already paid for:

- a proposer/critic model with no price (an $80 cap once recorded $0 per case)
- held-out cases that are also evolve cases (the report would overstate gains)
- too few cases to resolve the gain you want (rounds end "inside the noise band" whatever they find)
- a budget that can't finish round 0, or covers fewer rounds than configured
- tripwire baselines missing, so health.py skips its checks silently
- no smoke stage, so a broken setup costs a whole round 0 instead of ~$1
- a starting harness that fails the validation every candidate must pass
- a grader nothing has checked: no anchors, an evaluator that can't grade them, no task contract

What it can't see without running anything (a tool failing on every call, replays costing $0, a
grader that mislabels its anchors) is for the run's own first phases: anchors, then the smoke stage,
then round 0's always-pass tripwire. `check` warns when any of them is off.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path

from harness_evolve import anchors, candidates, profiles

FAIL, WARN, OK = "FAIL", "WARN", "OK"

MIN_EVOLVE = 30            # below this the noise band is too wide to accept anything but huge gains
TYPICAL_FLIP_RATE = 0.05   # share of cases whose verdict flips between two trials of the same harness
WIDE_BAND = 0.10           # a typical band above 10 points: warn
Z = 2.0                    # acceptance.calibrate_delta's defaults
MIN_FLIPS = 3.0
MARGIN = 1.25              # budget.Budget's reservation margin


@dataclass
class Finding:
    level: str
    name: str
    message: str
    fix: str = ""


def _default(name: str):
    from harness_evolve.loop import OptimizerConfig
    return next(f.default for f in fields(OptimizerConfig) if f.name == name)


def harness(base: Path) -> list[Finding]:
    if not base.is_dir():
        return [Finding(FAIL, "harness", f"starting harness {base} is not a directory",
                        "point the config's base harness at the folder holding your prompts and settings")]
    files = candidates.harness_files(base)
    if not files:
        return [Finding(FAIL, "harness", f"starting harness {base} has no files", "add your prompts and settings")]
    try:
        candidates.validate(base, base)
    except candidates.InvalidCandidate as exc:
        return [Finding(FAIL, "harness", f"the starting harness fails the validation every candidate must pass: {exc}",
                        "fix the harness (or the profile's setting_bounds) so the starting values are valid")]
    out = []
    settings = json.loads(files["settings.json"]) if "settings.json" in files else {}
    bounds = profiles.active().setting_bounds
    unknown = sorted(set(bounds) - set(settings))
    if unknown:
        out.append(Finding(WARN, "settings", f"bounds or switches for settings not in settings.json: {unknown}",
                           "add them to settings.json or drop them from the profile; the proposer can't change them"))
    unbounded = sorted(k for k, v in settings.items() if not k.startswith("_") and k not in bounds
                       and isinstance(v, int | float) and not isinstance(v, bool))
    if unbounded:
        out.append(Finding(WARN, "settings", f"numeric settings without bounds: {unbounded}",
                           "give each a (lo, hi) in setting_bounds; otherwise any value passes validation"))
    if not out:
        out.append(Finding(OK, "harness", f"{len(files)} files, valid; every numeric setting is bounded"))
    return out


def cases(cfg) -> list[Finding]:
    out = []
    if not cfg.evolve_cases:
        return [Finding(FAIL, "cases", "no evolve cases", "list the cases the optimizer measures each candidate on")]
    for name in ("evolve_cases", "guard_cases", "final_cases", "smoke_cases", "transfer_cases"):
        ids = list(getattr(cfg, name))
        dups = sorted({c for c in ids if ids.count(c) > 1})
        if dups:
            out.append(Finding(WARN, "cases", f"{name} lists cases twice: {dups[:5]}", "remove the duplicates"))
    tuned = set(cfg.evolve_cases) | set(cfg.guard_cases)
    leak = sorted(tuned & set(cfg.final_cases))
    if leak:
        out.append(Finding(FAIL, "cases", f"{len(leak)} held-out case(s) are also evolve/guard cases: {leak[:5]}",
                           "keep final_cases disjoint; the held-out report would overstate the gain"))
    both = sorted(set(cfg.evolve_cases) & set(cfg.guard_cases))
    if both:
        out.append(Finding(WARN, "cases", f"cases in both evolve_cases and guard_cases: {both[:5]}",
                           "a case is either measured for gains or guarded, not both"))
    if not cfg.final_cases:
        out.append(Finding(WARN, "cases", "no held-out cases: the run ends without a held-out report",
                           "keep ~30% of your cases as final_cases; evolve-set gains overfit"))
    if len(cfg.evolve_cases) < MIN_EVOLVE:
        out.append(Finding(WARN, "cases", f"only {len(cfg.evolve_cases)} evolve cases",
                           f"use {MIN_EVOLVE}+ real cases; with fewer, only very large gains can be accepted"))
    if not out:
        out.append(Finding(OK, "cases", f"{len(cfg.evolve_cases)} evolve, {len(cfg.guard_cases)} guard, "
                                        f"{len(cfg.final_cases)} held-out; no overlap"))
    return out


def band(n: int, trials: int, flip_rate: float = 0.0) -> float:
    """The noise band round 0 would calibrate on n cases (pass-rate units): the rule-of-three floor if no
    pair disagrees, or the bootstrap's value if `flip_rate` of the pairs do (calibrate_delta)."""
    floor = Z * math.sqrt(MIN_FLIPS) / n
    noisy = Z * math.sqrt(flip_rate / n)
    return max(floor, noisy) / math.sqrt(max(trials, 1))


def resolution(cfg) -> list[Finding]:
    n = len(cfg.evolve_cases) + len(cfg.guard_cases)
    if n < 2:
        return []
    if cfg.calibration_trials < 2:
        return [Finding(WARN, "noise band", f"calibration_trials={cfg.calibration_trials}: round 0 can't measure "
                                            f"noise, so the fixed default_delta={cfg.default_delta:.2f} is used",
                        "set calibration_trials=2")]
    best, typical = band(n, cfg.trials), band(n, cfg.trials, TYPICAL_FLIP_RATE)
    msg = (f"{n} cases x {cfg.trials} trial(s): gains under ~{best * 100:.1f} points can't be accepted even if the "
           "agent never flips a verdict")
    if typical > best:
        msg += f"; ~{typical * 100:.1f} points if {TYPICAL_FLIP_RATE:.0%} of cases flip between trials"
    if best >= 1.0:
        msg = (f"{n} cases x {cfg.trials} trial(s): the noise band is wider than the whole pass-rate range, "
               "so no gain can be accepted")
    if typical > WIDE_BAND:
        return [Finding(WARN, "noise band", msg, "add cases or trials, or expect only large gains to be accepted")]
    return [Finding(OK, "noise band", msg)]


def estimate(cfg) -> dict:
    """Rough dollar cost of each phase at default_case_cost_usd per case per trial (proposer and critic
    calls not included)."""
    c, ng = cfg.default_case_cost_usd, len(cfg.evolve_cases) + len(cfg.guard_cases)
    per_round = ng * cfg.trials * c * (3 if cfg.confirm else 1)      # confirm re-measures both: +2 evals
    return {"smoke": len(cfg.smoke_cases) * c, "round0": ng * cfg.calibration_trials * c,
            "per_round": per_round, "rounds": per_round * cfg.max_rounds,
            "final": len(cfg.final_cases) * cfg.final_trials * c * 2}             # original and final harness


def budget(cfg) -> list[Finding]:
    e, out = estimate(cfg), []
    if cfg.default_case_cost_usd == _default("default_case_cost_usd"):
        out.append(Finding(WARN, "budget", f"default_case_cost_usd is the built-in ${cfg.default_case_cost_usd}, "
                                           "so these estimates are guesses",
                           "set it near your agent's measured cost per case per trial"))
    start = (e["smoke"] + e["round0"]) * MARGIN
    full = (e["smoke"] + e["round0"] + e["rounds"] + e["final"]) * MARGIN
    summary = (f"~${full:.2f} for the whole run (smoke ${e['smoke']:.2f}, round 0 ${e['round0']:.2f}, "
               f"${e['per_round']:.2f}/round x {cfg.max_rounds}, held-out ${e['final']:.2f}, "
               f"with the {MARGIN}x reservation margin; proposer/critic calls extra)")
    if cfg.budget_usd < start:
        out.append(Finding(FAIL, "budget", f"${cfg.budget_usd:.2f} can't finish round 0 (needs ~${start:.2f}); "
                                           + summary, "raise --budget or use fewer cases or trials"))
    elif cfg.budget_usd < full:
        spare = cfg.budget_usd / MARGIN - e["smoke"] - e["round0"] - e["final"]
        fit = max(0, int(spare // e["per_round"])) if e["per_round"] else cfg.max_rounds
        out.append(Finding(WARN, "budget", f"${cfg.budget_usd:.2f} covers ~{fit} of {cfg.max_rounds} rounds "
                                           "with the held-out phase; " + summary,
                           "raise --budget or lower --rounds; a stopped run resumes with a higher cap"))
    else:
        out.append(Finding(OK, "budget", f"${cfg.budget_usd:.2f} cap; " + summary))
    return out


def _litellm_price(model: str) -> bool | None:
    """True/False: litellm's local price table has/lacks the model. None: litellm isn't installed."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")     # don't fetch the table over the network
    try:
        import litellm
    except ImportError:
        return None
    litellm.suppress_debug_info = True                                 # no provider-list banner
    try:
        info = litellm.get_model_info(model)
    except Exception:
        return False
    return bool(info.get("input_cost_per_token") or info.get("output_cost_per_token"))


def llms(proposer_llm, critic_llm, allow_unpriced: bool) -> list[Finding]:
    out, seen = [], {}
    for role, llm in (("proposer", proposer_llm), ("critic", critic_llm)):
        model = getattr(llm, "model", None)
        if model is None:
            out.append(Finding(OK, "prices", f"{role}: custom LLM; its cost is read from each reply at run time"
                                             + (" (unpriced replies count $0: allow_unpriced_llm=True)"
                                                if allow_unpriced else " (an unpriced reply stops the run)")))
            continue
        if getattr(llm, "prices", None):
            out.append(Finding(OK, "prices", f"{role}: {model} at ${llm.prices[0]}/${llm.prices[1]} per M tokens"))
            continue
        priced = seen.setdefault(model, _litellm_price(model))
        if priced:
            out.append(Finding(OK, "prices", f"{role}: {model} priced from litellm's table"))
        elif priced is None:
            out.append(Finding(WARN, "prices", f"{role}: {model}, but litellm isn't installed to check its price",
                               "pip install 'harness-evolve[litellm]'"))
        else:
            out.append(Finding(WARN if allow_unpriced else FAIL, "prices",
                               f"{role}: litellm has no price for {model}"
                               + (", so its calls count $0 against the cap" if allow_unpriced
                                  else ", so the run stops at its first call"),
                               "pass prices=(in, out) $ per million tokens to litellm_llm"))
    return out


def health(cfg) -> list[Finding]:
    base = {**profiles.active().health_baseline, **cfg.health_baseline}
    out = []
    missing = [k for k in ("tool_failure_rate", "cost_per_trial_usd", "pass_rate") if not base.get(k)]
    if len(missing) == 3:
        out.append(Finding(WARN, "tripwires", "no health_baseline: the tool-failure, cost and round-0 pass-rate "
                                              "tripwires are off (they skip silently without one)",
                           "measure a pilot round 0 and set health_baseline in the profile or config"))
    elif missing:
        out.append(Finding(WARN, "tripwires", f"health_baseline lacks {missing}: those tripwires are off",
                           "add them from a pilot round 0"))
    else:
        out.append(Finding(OK, "tripwires", "health_baseline sets tool failure rates, cost and pass rate"))
    if not cfg.smoke_cases:
        e = estimate(cfg)
        out.append(Finding(WARN, "smoke stage", f"no smoke_cases: a broken setup (failing tool, $0 replays) is found "
                                                f"after round 0 (~${e['round0']:.2f}), not before",
                           "set smoke_cases to 3-5 cases; they run once before round 0"))
    else:
        out.append(Finding(OK, "smoke stage", f"{len(cfg.smoke_cases)} smoke case(s) run once before round 0"))
    return out


def grader(cfg, evaluator) -> list[Finding]:
    out = []
    if not cfg.anchors:
        out.append(Finding(WARN, "grader", "no anchors: nothing checks that the grader can tell right from wrong "
                                           "before the run trusts it",
                           "add anchors (anchors.jsonl, see anchors.py): a few correct outputs and clearly wrong "
                           "ones; a blank output is added for each case automatically"))
    else:
        bad = anchors.problems(cfg.anchors)
        if bad:
            out.append(Finding(FAIL, "grader", "; ".join(bad[:3]), "fix anchors.jsonl"))
        elif not callable(getattr(evaluator, "grade", None)):
            out.append(Finding(FAIL, "grader", "anchors are set but the evaluator can't grade an output on its own",
                               "give CommandEvaluator a grade_command, or add grade(case_id, output) to your "
                               "evaluator"))
        else:
            known = (set(cfg.evolve_cases) | set(cfg.guard_cases) | set(cfg.final_cases)
                     | set(cfg.smoke_cases) | set(cfg.transfer_cases))
            stray = sorted({a["case"] for a in cfg.anchors} - known)
            if stray:
                out.append(Finding(WARN, "grader", f"anchors for cases in no case list: {stray[:5]}",
                                   "check the case ids; the grader looks each case up"))
            if not any(a["passed"] for a in cfg.anchors):
                out.append(Finding(WARN, "grader", "no anchor expects a pass, so a grader that fails everything "
                                                   "passes them", "add a correct output or two"))
            n_blank = len(anchors.with_blanks(cfg.anchors)) - len(cfg.anchors)
            if not out:
                out.append(Finding(OK, "grader", f"{len(cfg.anchors)} anchors (+{n_blank} blank outputs) are "
                                                 "graded before anything else"))
    if cfg.always_pass_tripwire is None:
        out.append(Finding(WARN, "grader", "always_pass_tripwire is off: a grader that passes everything won't "
                                           "stop round 0", "leave it at 0.98 unless the agent really is that good"))
    if not profiles.active().task_contract:
        out.append(Finding(WARN, "grader", "no task_contract: the critic judges grader gaming without being told "
                                           "what counts as correct",
                           "pass task_contract=\"...\" to AgentProfile.build"))
    return out


def run(opt) -> list[Finding]:
    cfg = opt.cfg
    return (harness(opt.base) + cases(cfg) + resolution(cfg) + budget(cfg)
            + llms(opt.proposer_llm, opt.critic_llm, cfg.allow_unpriced_llm) + grader(cfg, opt.evaluator)
            + health(cfg))


def render(findings: list[Finding]) -> str:
    lines = []
    for f in findings:
        lines.append(f"{f.level:<4}  {f.name}: {f.message}")
        if f.fix and f.level != OK:
            lines.append(f"      fix: {f.fix}")
    n = {lvl: sum(f.level == lvl for f in findings) for lvl in (FAIL, WARN, OK)}
    lines.append(f"\n{n[FAIL]} FAIL, {n[WARN]} WARN, {n[OK]} OK")
    return "\n".join(lines)


def exit_code(findings: list[Finding], strict: bool = False) -> int:
    bad = {FAIL, WARN} if strict else {FAIL}
    return 1 if any(f.level in bad for f in findings) else 0
