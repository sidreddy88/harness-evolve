"""
Agent profiles: what the optimizer needs to know about the agent it improves.

The loop, acceptance rules, budget, state, history and report are the same for
every agent. A profile supplies the rest: where the harness lives, what the
agent does (so the proposer and critic understand it), which mechanisms must
never be weakened, which files each component may edit, allowed ranges for
settings, structural switches, and optionally a trajectory grader and health
thresholds.

Switches are settings that change how the agent works rather than what it is
told (planning step on/off, a tool on/off, how much history it sees): each has
named options, documented by what they do. The proposer is told to check them
before rewording prompts. Our evidence (the r9 fix-agent run): a control-flow
change transferred to a second model and a wording change didn't; arXiv
2609.20804 found such components help or hurt depending on the model and task,
so they are worth searching per agent rather than fixing once.

    from harness_evolve import AgentProfile, profiles
    profile = AgentProfile.build(
        name="my-agent", harness_root=Path("harness/"),
        agent_description="MyAgent answers support tickets using a search tool ...",
        safety_mechanisms=["citing the source document for every claim"],
        setting_bounds={"max_turns": (3, 20)})
    profiles.use(profile)

One optimizer run optimizes one agent, so the active profile is process-wide.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Default edit surface: every *.prompt file, the tool descriptions and the settings.
DEFAULT_COMPONENT_FILES: dict[str, Callable[[str], bool]] = {
    "prompts": lambda f: f.endswith(".prompt"),
    "tool_descriptions": lambda f: f == "tool_descriptions.json",
    "settings": lambda f: f == "settings.json",
}

PROPOSER_TEMPLATE = """You improve the harness of an AI agent: the prompt text, tool descriptions
and settings around a frozen model.

THE AGENT: {agent_description}

You will see the current harness files, evidence from the agent's recent runs (what
it did, where it failed or wasted turns, what it cost), and the history of edits
already tried and how they scored.

Propose TWO alternative edits, each exactly one mechanism, one component and one
hypothesis, with different mechanisms (ideally different components), and say which
you expect to help more. Target failures or waste you can point to in the evidence.
The ACCEPTANCE RULES section says exactly what this run can detect: an edit is
accepted if it raises the pass rate by more than the noise band (it may then cost
more), or if it is cheaper by more than cost noise at an unchanged pass rate. Both
levers count; pick the one the evidence supports.

If the evidence has "Cost by prompt source" (where the money goes: the model's own
output and each part of the input), read it first and aim at the largest share you can
plausibly shrink. If the model's output dominates, the lever is what the harness asks it to
write, not how briefly it writes. Never instruct the agent to be brief, to save tokens
or to do less: that makes agents reluctant to do the work and lowers accuracy. Change
what the harness asks for, not how hard the model tries. Describe how tools behave
rather than adding commands; do not add emphasis (MUST, NEVER, CRITICAL). Removing text
that isn't earning its place is a valid edit.

Constraints: never mention specific tasks, inputs, file paths or answers from the
evidence. {constraints}Never weaken: {safety}. Do not repeat an edit the history shows
was already rejected.

Settings are behaviour, not just numbers: read the SETTINGS section for what each one
does and its allowed range. Do not re-propose an idea from REJECTED IDEAS unless the
evidence shows something new, and then say what.

{switches}Return STRICT JSON only:
{{"candidates": [
   {{"component": one of {components},
    "hypothesis": "what failure/waste this fixes and why the edit should fix it",
    "edits": [{{"file": "<harness file name>", "find": "<exact text currently in the file>",
               "replace": "<new text>"}}]}},
   {{... a second, different candidate ...}}],
 "preferred": 0 or 1,
 "why": "why the preferred one should help more"}}
Each "find" must occur exactly once in its file."""

SWITCHES_TEXT = """Some settings are STRUCTURAL SWITCHES (marked in the SETTINGS section): they change how
the agent works, not what it is told. Before proposing a wording change, check whether
switching one addresses the failure or waste in the evidence; if one does, make it one of
your two candidates. A switch is changed by editing its value in settings.json to another
listed option.

"""

CRITIC_TEMPLATE = """You are a strict reviewer of changes to an agent harness in an automated
evolution loop. The harness is evolved against the very tasks it is measured on,
so anti-overfitting review is critical. The change was written by another model
in response to aggregated failure modes. Catch changes that would be cheating,
degenerate, or self-destructive.

THE AGENT: {agent_description} The harness you are reviewing is its prompt text,
tool descriptions and settings.
{context}{contract}
REJECT if ANY of:

1. LEAKAGE / TASK-SPECIALIZATION: the diff encodes knowledge of specific evaluation
   tasks: task ids, names, file paths or identifiers from particular inputs, expected
   answers, magic constants that only make sense for one task, or branching on
   task-identifying features. General practice is fine. Litmus test: would this change
   still make sense, and still help, on an unfamiliar task of the same kind?
2. DEGENERATE: the diff is effectively a no-op while claiming a mechanism, or it
   deletes or weakens a safety mechanism without a working replacement: {safety}.
   Making the agent give up sooner to look cheaper is degenerate.
3. GRADER GAMING: the change tries to infer, target or influence how outputs are
   scored rather than improving the work itself.{gaming}
4. UNDECLARED BUNDLING: the diff contains changes not covered by the declared
   component and hypothesis, or one declared edit secretly bundles several
   independent mechanisms.
5. MEMORY LEAKAGE: the change injects task-specific data (inputs, answers, names) into
   the prompt as if it were general guidance.
6. UNBOUNDED WORK: an added check, retry or "keep verifying" instruction with no
   exit, or anything that could spend the whole turn budget without finishing.
   (A numeric setting changed within its allowed range is bounded by code: that is
   NOT unbounded work.)
7. NEAR-DUPLICATE: the change re-proposes an idea listed under ALREADY REJECTED
   (the same mechanism in different words) without citing evidence that is new.
8. FALSE TOOL CLAIM: added text states how a tool behaves in a way that contradicts
   the tool's own description (shown under CURRENT TOOL DESCRIPTIONS).

Otherwise ACCEPT. Review intent and content, not style or wording quality.
Return STRICT JSON only:
{{"verdict": "accept" | "reject", "reasons": ["..."], "risk_notes": ["..."]}}"""


@dataclass(frozen=True)
class AgentProfile:
    name: str
    harness_root: Path
    setting_bounds: dict
    components: tuple
    component_files: dict[str, Callable[[str], bool]]
    proposer_system: str
    critic_system: str
    grader: object | None = None          # optional: .grade(traj) -> obj with .accepted; .summarize; .render
    health_baseline: dict = field(default_factory=dict)
    component_families: dict = field(default_factory=dict)   # component -> family, for the exploration rule
    switches: dict = field(default_factory=dict)   # setting -> {option: what it does}
    task_contract: str = ""                        # what counts as correct; the critic checks gaming against it

    @classmethod
    def build(cls, name: str, harness_root: str | Path, agent_description: str,
              safety_mechanisms: Sequence[str] = (), constraints: str = "",
              production_context: str = "", gaming_examples: str = "",
              setting_bounds: dict | None = None,
              component_files: dict[str, Callable[[str], bool]] | None = None,
              component_families: dict | None = None, grader: object | None = None,
              health_baseline: dict | None = None,
              switches: dict[str, dict[str, str]] | None = None,
              task_contract: str = "") -> AgentProfile:
        """A profile with the proposer and critic instructions written for your agent.

        agent_description: what the agent does, its tools, how it finishes a task.
        safety_mechanisms: things an edit must never weaken (e.g. "citing evidence").
        constraints: extra rules for the proposer (e.g. "edits must also help on X").
        production_context: how evaluation differs from production, for the critic.
        gaming_examples: what gaming your grader would look like, for the critic.
        task_contract: what counts as a correct result, independent of how the grader checks it (e.g.
            "the final number equals the problem's answer"). The critic judges grader gaming against it:
            in arXiv 2610.11464 an outer judge missed rubric gaming until it was given the task contract.
        setting_bounds: {setting: (lo, hi)} for numbers, or frozenset({...}) of allowed values.
        switches: {setting: {option: what the agent does with it}} for structural settings (e.g.
            {"plan": {"off": "solves directly", "on": "writes a plan first, one extra model call"}}).
            Their allowed values are the options; the proposer sees each option's effect.
        """
        files = component_files or DEFAULT_COMPONENT_FILES
        safety = "; ".join(safety_mechanisms) if safety_mechanisms else "the agent's checks on its own output"
        components = tuple(files)
        switches = {k: dict(v) for k, v in (switches or {}).items()}
        bounds = dict(setting_bounds or {})
        for key, options in switches.items():
            bounds[key] = frozenset(options)
        proposer = PROPOSER_TEMPLATE.format(
            switches=SWITCHES_TEXT if switches else "",
            agent_description=agent_description.strip(), safety=safety,
            constraints=(constraints.strip() + " ") if constraints else "", components=list(components))
        critic = CRITIC_TEMPLATE.format(
            agent_description=agent_description.strip(), safety=safety,
            context=("\n" + production_context.strip() + "\n") if production_context else "",
            contract=(f"\nWHAT COUNTS AS CORRECT (the task contract): {task_contract.strip()}\nFor rule 3, "
                      "an edit that aims at what the grader accepts rather than at this contract is gaming.\n"
                      if task_contract.strip() else ""),
            gaming=(" " + gaming_examples.strip()) if gaming_examples else "")
        return cls(name=name, harness_root=Path(harness_root), setting_bounds=bounds,
                   components=components, component_files=dict(files), proposer_system=proposer,
                   critic_system=critic, grader=grader, health_baseline=dict(health_baseline or {}),
                   component_families=dict(component_families or {c: c for c in components}),
                   switches=switches, task_contract=task_contract.strip())


_active: AgentProfile | None = None


def use(profile: AgentProfile) -> AgentProfile:
    """Make `profile` the one this process optimizes."""
    global _active
    _active = profile
    return profile


def active() -> AgentProfile:
    if _active is None:
        raise RuntimeError("no agent profile: call harness_evolve.profiles.use(AgentProfile.build(...)) first")
    return _active
