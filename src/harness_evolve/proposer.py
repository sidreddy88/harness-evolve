"""
Proposer: reads the evidence and proposes ONE edit to the harness.

GEPA-style reflective mutation (Agrawal et al., "GEPA: Reflective Prompt
Evolution Can Outperform Reinforcement Learning", ICLR 2026): the proposer
sees the agent's actual execution traces and the edit history, diagnoses in
natural language what went wrong, and targets the edit at that failure,
instead of a random perturbation of a scalar score.

One declared edit per candidate is RRSI's annealed L0 budget at its
floor (b_min = 1). With a handful of rounds there is nothing to anneal, and a
single edit keeps every measured change attributable to one mechanism.

The edit is returned as exact search/replace operations on harness files,
not full file rewrites: prompts can be tens of thousands of characters, and
asking a model to reproduce one verbatim to change a paragraph is expensive and
invites silent drift in the parts it didn't mean to touch.

The instructions (the proposer's system prompt) come from the agent profile
(profiles.py), written for the agent being optimized.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from harness_evolve.candidates import InvalidCandidate, harness_files

LLM = Callable[[str, str], Awaitable[str]]

def _profile():
    from harness_evolve import profiles
    return profiles.active()


@dataclass
class Proposal:
    component: str
    hypothesis: str
    edits: list[dict] = field(default_factory=list)




def settings_table(harness_dir: Path) -> str:
    """Each setting with its current value, allowed range and what it does, so the
    proposer reads settings as behaviour. r7's proposer never touched settings,
    even one that targeted the exact failure its evidence named."""
    import json as _json
    raw = _json.loads((Path(harness_dir) / "settings.json").read_text())
    docs = raw.get("_doc", {})
    lines = []
    switches = _profile().switches
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if key in switches:
            opts = "; ".join(f"{o!r}: {what}" for o, what in switches[key].items())
            lines.append(f"- {key} = {value!r} [STRUCTURAL SWITCH] {docs.get(key, '')} Options: {opts}")
            continue
        bounds = _profile().setting_bounds.get(key)
        if isinstance(bounds, frozenset):
            rng = f", one of {sorted(bounds)}"
        else:
            rng = f", allowed {bounds[0]}-{bounds[1]}" if bounds else ""
        lines.append(f"- {key} = {value!r}{rng}: {docs.get(key, '(undocumented)')}")
    return "\n".join(lines)


def build_prompt(harness_dir: Path, evidence: str, history: str, feedback: str = "",
                 allowed_components: tuple[str, ...] | None = None,
                 run_context: dict | None = None) -> str:
    files = harness_files(harness_dir)
    parts = ["=== CURRENT HARNESS FILES ==="]
    for name, text in files.items():
        parts.append(f"--- {name} ({len(text)} chars) ---\n{text}")
    parts.append(f"=== SETTINGS (behaviour you can change) ===\n{settings_table(harness_dir)}")
    ctx = run_context or {}
    if ctx.get("acceptance"):
        parts.append(f"=== ACCEPTANCE RULES FOR THIS RUN ===\n{ctx['acceptance']}")
    if ctx.get("near_misses"):
        parts.append("=== NEAR MISSES (positive but within the noise band, so not accepted; you may "
                     "build on one or combine it with a new mechanism, saying which) ===\n"
                     + "\n".join(f"- {r}" for r in ctx["near_misses"]))
    if ctx.get("rejected"):
        parts.append("=== REJECTED IDEAS (measured; don't repeat without new evidence) ===\n"
                     + "\n".join(f"- {r}" for r in ctx["rejected"]))
    parts.append(f"=== EVIDENCE FROM RECENT RUNS ===\n{evidence}")
    parts.append(f"=== EDIT HISTORY ===\n{history}")
    if allowed_components:
        parts.append("=== EXPLORATION CONSTRAINT FOR THIS ROUND ===\n"
                     f"Recent candidates in this run all edited other components and none was "
                     f"accepted. This round's edit must target one of: {list(allowed_components)}. "
                     f"Read those files as behaviour you can change, not just wording.")
    if feedback:
        parts.append(f"=== YOUR PREVIOUS ATTEMPT THIS ROUND WAS REJECTED ===\n{feedback}\n"
                     f"Fix those problems, or propose a different edit.")
    return "\n\n".join(parts)


def parse_all(text: str) -> list[Proposal]:
    """Both output shapes: {"candidates": [...], "preferred": i} (preferred first)
    or a single {"component", "hypothesis", "edits"}. Invalid entries are dropped;
    raises only if none is valid."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        data = json.loads(t)
    except json.JSONDecodeError as exc:
        raise InvalidCandidate(f"proposer did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "candidates" not in data:
        return [_one(data)]
    raw = [c for c in data.get("candidates") or [] if isinstance(c, dict)]
    pref = data.get("preferred", 0)
    if isinstance(pref, int) and 0 <= pref < len(raw):
        raw = [raw[pref]] + [c for i, c in enumerate(raw) if i != pref]
    out, errors = [], []
    for c in raw:
        try:
            out.append(_one(c))
        except InvalidCandidate as exc:
            errors.append(str(exc))
    if not out:
        raise InvalidCandidate("; ".join(errors) or "no candidates")
    return out


def parse(text: str) -> Proposal:
    return parse_all(text)[0]


def _one(data: dict) -> Proposal:
    comp = data.get("component")
    components = _profile().components
    if comp not in components:
        raise InvalidCandidate(f"component must be one of {components}, got {comp!r}")
    edits = data.get("edits") or []
    if not edits or not all(isinstance(e, dict) and {"file", "find", "replace"} <= set(e) for e in edits):
        raise InvalidCandidate("edits must be a non-empty list of {file, find, replace}")
    return Proposal(comp, str(data.get("hypothesis", "")).strip(), edits)


def apply_ops(harness_dir: Path, edits: list[dict]) -> dict[str, str]:
    """Search/replace ops -> {file: new full content}. Each find must match once."""
    files = harness_files(harness_dir)
    out: dict[str, str] = {}
    for e in edits:
        name = e["file"]
        if name not in files:
            raise InvalidCandidate(f"no harness file named {name!r}")
        text = out.get(name, files[name])
        n = text.count(e["find"])
        if n != 1:
            raise InvalidCandidate(f"{name}: find text occurs {n} times (must be exactly 1): "
                                   f"{e['find'][:80]!r}")
        out[name] = text.replace(e["find"], e["replace"])
    return out


def _check(harness_dir: Path, proposal: Proposal,
           allowed_components: tuple[str, ...] | None) -> dict[str, str]:
    if allowed_components and proposal.component not in allowed_components:
        raise InvalidCandidate(f"this round must edit one of {list(allowed_components)}, "
                               f"got {proposal.component!r}")
    if allowed_components:
        owns = _profile().component_files[proposal.component]
        stray = sorted({e["file"] for e in proposal.edits if not owns(e["file"])})
        if stray:
            raise InvalidCandidate(f"component {proposal.component!r} can't edit {stray}")
    return apply_ops(harness_dir, proposal.edits)


async def propose_all(harness_dir: Path, evidence: str, history: str, llm: LLM,
                      feedback: str = "", allowed_components: tuple[str, ...] | None = None,
                      run_context: dict | None = None) -> list[tuple[Proposal, dict[str, str]]]:
    """Every valid candidate from one proposer call, preferred first."""
    text = await llm(_profile().proposer_system, build_prompt(harness_dir, evidence, history, feedback,
                                          allowed_components, run_context))
    out, errors = [], []
    for proposal in parse_all(text):
        try:
            out.append((proposal, _check(harness_dir, proposal, allowed_components)))
        except InvalidCandidate as exc:
            errors.append(f"{proposal.component}: {exc}")
    if not out:
        raise InvalidCandidate("; ".join(errors))
    return out


async def propose(harness_dir: Path, evidence: str, history: str, llm: LLM,
                  feedback: str = "", allowed_components: tuple[str, ...] | None = None,
                  run_context: dict | None = None) -> tuple[Proposal, dict[str, str]]:
    return (await propose_all(harness_dir, evidence, history, llm, feedback,
                              allowed_components, run_context))[0]
