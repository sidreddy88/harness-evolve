"""
CommandEvaluator: measure a harness by running a command once per case and trial, so an agent in any
language (or behind an API) can be optimized without writing a Python evaluator.

    from harness_evolve.command import CommandEvaluator
    evaluator = CommandEvaluator(["node", "run_case.js", "{harness}", "{case}"], concurrency=4)

Placeholders {harness} (the candidate harness directory), {case} (the case id) and {trial} (1, 2, ...)
are filled in each argument; the same values are in the environment as HE_HARNESS_DIR, HE_CASE_ID and
HE_TRIAL. The command runs the agent with that harness on that case, grades it, and prints one JSON
object as the LAST line of its stdout:

    {"passed": true,                 required
     "cost_usd": 0.0123,             required (what the agent's model calls cost)
     "submitted": true,              optional: did the agent finish with an answer (default: passed)
     "detail": "answered 42",        optional: a short note the proposer sees
     "answer": "expected 41",        optional: hidden from the proposer (put the right answer here)
     "steps": [{"name": "search", "input": {...}, "output": "..."}],   optional: tool calls, as evidence
     "turns": 3,                     optional
     "provider_failure": "..."}      optional: billing/auth/rate-limit problem; stops the run, resumable

A non-zero exit, a timeout or unreadable output scores the trial as an error (it counts as a failure,
and is shown to the proposer with the end of stderr), not as a provider failure.

`grade_command` (optional) grades a given output without running the agent, for grader anchors
(anchors.py): placeholders {case} and {output} (a file holding the output text); it prints one JSON
object as the last line, {"passed": bool, "cost_usd": float (optional, e.g. an LLM judge's cost)}.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from harness_evolve.acceptance import CaseResult, EvalResult
from harness_evolve.evaluator import EvalOutcome, ProviderFailure
from harness_evolve.evidence import HIDDEN_MARKER


class CommandEvaluator:
    def __init__(self, command: Sequence[str], concurrency: int = 4, timeout_s: float = 600.0,
                 cwd: str | Path | None = None, env: dict | None = None,
                 grade_command: Sequence[str] | None = None):
        if isinstance(command, str):
            raise TypeError("command must be a list of arguments, e.g. "
                            "['python', 'run_case.py', '{harness}', '{case}']")
        self.command, self.timeout_s = list(command), timeout_s
        self.cwd, self.env = cwd, env
        self.grade_command = list(grade_command) if grade_command else None
        if self.grade_command is None:
            self.grade = None                    # anchors.py: this evaluator can't grade an output alone
        self.gate = asyncio.Semaphore(concurrency)

    async def _trial(self, harness_dir: Path, case: str, trial: int) -> dict:
        values = {"harness": str(Path(harness_dir).resolve()), "case": case, "trial": str(trial)}
        argv = [a.format(**values) for a in self.command]
        env = {**os.environ, **(self.env or {}), "HE_HARNESS_DIR": values["harness"], "HE_CASE_ID": case,
               "HE_TRIAL": values["trial"]}
        base = {"instance_id": case, "trial": trial, "steps": [], "submitted": False, "cost": {"cost_usd": 0.0}}
        async with self.gate:
            proc = await asyncio.create_subprocess_exec(*argv, cwd=self.cwd, env=env, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return {**base, "verdict": "TIMEOUT", "detail": f"no result after {self.timeout_s:g}s"}
        lines = [x for x in out.decode(errors="replace").splitlines() if x.strip()]
        try:
            res = json.loads(lines[-1])
            if not isinstance(res, dict) or "passed" not in res or "cost_usd" not in res:
                raise ValueError("needs at least passed and cost_usd")
        except (IndexError, ValueError) as exc:
            tail = err.decode(errors="replace").strip()[-400:]
            why = f"exit {proc.returncode}" if proc.returncode else f"unreadable result ({exc})"
            return {**base, "verdict": "ERROR", "detail": f"{why}: {tail}"}
        if res.get("provider_failure"):
            raise ProviderFailure(str(res["provider_failure"]), cost_usd=float(res.get("cost_usd") or 0.0))
        passed = bool(res["passed"])
        detail = str(res.get("detail", ""))
        if res.get("answer") is not None:
            detail += f" {HIDDEN_MARKER} {res['answer']}"
        return {**base, "verdict": "PASS" if passed else "FAIL", "detail": detail.strip(),
                "steps": list(res.get("steps") or []), "turns": res.get("turns"),
                "submitted": bool(res.get("submitted", passed)), "cost": {"cost_usd": float(res["cost_usd"])}}

    async def grade(self, case_id: str, output: str) -> dict:
        """Grade `output` for `case_id` with grade_command, without running the agent."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(output)
        try:
            argv = [a.format(case=case_id, output=f.name) for a in self.grade_command]
            proc = await asyncio.create_subprocess_exec(*argv, cwd=self.cwd, env={**os.environ, **(self.env or {})},
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return {"error": f"no result after {self.timeout_s:g}s"}
        finally:
            os.unlink(f.name)
        lines = [x for x in out.decode(errors="replace").splitlines() if x.strip()]
        try:
            res = json.loads(lines[-1])
            if not isinstance(res, dict) or "passed" not in res:
                raise ValueError("needs passed")
        except (IndexError, ValueError) as exc:
            why = f"exit {proc.returncode}" if proc.returncode else f"unreadable result ({exc})"
            return {"error": f"{why}: {err.decode(errors='replace').strip()[-300:]}"}
        return res

    async def evaluate(self, harness_dir: Path, case_ids: list[str], trials: int) -> EvalOutcome:
        trajs = await asyncio.gather(*(self._trial(harness_dir, c, t + 1)
                                       for c in case_ids for t in range(trials)))
        per_case = {}
        for c in case_ids:
            mine = [t for t in trajs if t["instance_id"] == c]
            verdicts = [t["verdict"] for t in mine]
            per_case[c] = CaseResult(passes=verdicts.count("PASS"), trials=len(mine), verdicts=verdicts,
                                     escalations=sum(not t["submitted"] for t in mine),
                                     cost_usd=sum(t["cost"]["cost_usd"] for t in mine))
        return EvalOutcome(result=EvalResult(per_case), cost_usd=sum(t["cost"]["cost_usd"] for t in trajs),
                           trajectories=trajs)
