"""
A small tool-using agent that solves math word problems, with everything that
shapes its behaviour in a harness directory (harness/): its prompts, tool
descriptions and settings. harness-evolve edits copies of that directory.

Tools: calculator(expression) and submit_answer(answer). The run ends when the
agent submits, runs out of turns, stops calling tools after its nudges, or makes
more calculator calls than stop_after (stuck detection: after remind_after calls
it is reminded to submit). Harnesses without these settings have no limit.
"""
from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass, field

from harness_evolve import load_harness

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> float:
    """Arithmetic only: numbers, + - * / // % ** and parentheses."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")
    return ev(ast.parse(expression.replace(",", ""), mode="eval"))


def _tools(h) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "calculator", "description": h.tool_description("calculator"),
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                           "required": ["expression"]}}},
        {"type": "function", "function": {
            "name": "submit_answer", "description": h.tool_description("submit_answer"),
            "parameters": {"type": "object", "properties": {"answer": {"type": "number"}},
                           "required": ["answer"]}}},
    ]


@dataclass
class Run:
    answer: float | None = None
    steps: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    turns: int = 0


async def solve(question: str, harness_dir, model: str, prices: tuple[float, float]) -> Run:
    """prices: ($ per million input tokens, $ per million output tokens)."""
    import litellm

    h = load_harness(harness_dir)
    messages = [{"role": "system", "content": h.render("system")},
                {"role": "user", "content": h.render("task", question=question)}]
    run, nudges = Run(), int(h.setting("nudges"))
    remind_after, stop_after = h.settings.get("remind_after"), h.settings.get("stop_after")
    calc_calls, stopped = 0, False
    for _ in range(int(h.setting("max_turns"))):
        run.turns += 1
        resp = await litellm.acompletion(model=model, messages=messages, tools=_tools(h), max_tokens=4000)
        u = resp.usage
        run.cost_usd += (u.prompt_tokens * prices[0] + u.completion_tokens * prices[1]) / 1e6
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [c.model_dump() for c in calls] or None})
        if not calls:
            if nudges <= 0:
                break
            nudges -= 1
            messages.append({"role": "user", "content": h.render("nudge")})
            continue
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if c.function.name == "submit_answer":
                try:
                    run.answer = float(args.get("answer"))
                    out = "Submitted."
                except (TypeError, ValueError):
                    out = "Error: answer must be a number."
            elif c.function.name == "calculator":
                calc_calls += 1
                if stop_after is not None and calc_calls > stop_after:
                    out, stopped = f"Stopped: more than {stop_after} calculator calls without submitting.", True
                else:
                    try:
                        out = repr(calculate(str(args.get("expression", ""))))
                    except Exception as exc:
                        out = f"Error: {exc}"
                    if remind_after is not None and calc_calls >= remind_after and "remind" in h.templates:
                        out += "\n" + h.render("remind", calls=calc_calls)
            else:
                out = f"Error: unknown tool {c.function.name}"
            run.steps.append({"name": c.function.name, "input": args, "output": out})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
            if stopped:
                break
        if run.answer is not None or stopped:
            break
    return run
