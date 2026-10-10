"""
Leakage critic: screens a candidate's diff BEFORE any money is spent on it.

Adapted from RRSI's critic (rrsi/critic.py, github.com/google-research/rrsi,
Apache-2.0): a deterministic precheck, then an LLM review of intent and
content. The six reject classes and their wording follow RRSI's prompt,
generalized here. A rejection goes back to the proposer for a bounded number of
repairs; a candidate that can't be repaired is recorded in the history without
a measurement.

The critic's instructions come from the agent profile (profiles.py): what the
agent does, which safety mechanisms must not be weakened, and how evaluation
differs from production (RRSI's litmus test: "would this still help on an
unfamiliar task?").
"""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

LLM = Callable[[str, str], Awaitable[str]]   # (system, prompt) -> text

GENERIC_PATTERNS = [
    (r"sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}", "credential in diff"),
]

@dataclass
class Review:
    accept: bool
    reasons: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)


def domain_patterns(case_ids: list[str], repos: list[str], truth_paths: list[str]) -> list[tuple[str, str]]:
    """Denylist from the evaluation data itself: any evolve OR held-out case id,
    repo name, or file path touched by a true fix is leakage by definition."""
    pats = []
    for cid in case_ids:
        pats.append((re.escape(cid), f"names evaluation case {cid}"))
    for repo in repos:
        owner, _, name = repo.partition("/")
        for token in {owner, name} - {""}:
            if len(token) >= 4:
                pats.append((rf"(?i)\b{re.escape(token)}\b", f"names repository {repo}"))
    for path in truth_paths:
        pats.append((re.escape(path), f"names a file touched by a true fix ({path})"))
    return pats


def precheck(diff: str, patterns: list[tuple[str, str]]) -> list[str]:
    # Only added lines count: removing a leaky line is fine.
    added = "\n".join(line[1:] for line in diff.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    return sorted({why for pat, why in GENERIC_PATTERNS + patterns if re.search(pat, added)})


async def review(diff: str, component: str, hypothesis: str, llm: LLM,
                 patterns: list[tuple[str, str]], attempts: int = 3,
                 rejected: list[str] | tuple = (), tool_descriptions: str = "") -> Review:
    hard = precheck(diff, patterns)
    if hard:
        return Review(False, [f"precheck: {h}" for h in hard])
    if not diff.strip():
        return Review(False, ["empty diff"])
    payload = (f"DECLARED COMPONENT: {component}\nHYPOTHESIS: {hypothesis}\n\n"
               f"=== DIFF ===\n{diff[:60_000]}")
    if rejected:
        payload += "\n\n=== ALREADY REJECTED IN THIS RUN ===\n" + "\n".join(f"- {r}" for r in rejected)
    if tool_descriptions:
        payload += f"\n\n=== CURRENT TOOL DESCRIPTIONS ===\n{tool_descriptions[:12_000]}"
    last = ""
    for _ in range(attempts):
        from harness_evolve import profiles
        last = await llm(profiles.active().critic_system, payload)
        try:
            v = json.loads(_strip_fences(last))
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict) and v.get("verdict") in ("accept", "reject"):
            return Review(v["verdict"] == "accept", list(v.get("reasons") or []),
                          list(v.get("risk_notes") or []))
    return Review(False, [f"critic returned no valid verdict after {attempts} attempts: {last[:200]}"])


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()
