"""
LLM calls for the proposer and critic.

An LLM is any async callable `(system, prompt) -> str`. Return an `LLMReply`
(a str that also carries `cost_usd`) so proposer and critic calls count
against the run's budget; a plain str has no known cost, and the loop stops
the run (resumably) rather than spend blind, unless the config says
`allow_unpriced_llm=True`.

`litellm_llm(model, prices)` is a ready adapter if litellm is installed.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable


class LLMReply(str):
    """The reply text, plus what the call cost (None if unknown)."""
    cost_usd: float | None

    def __new__(cls, text: str, cost_usd: float | None = None) -> "LLMReply":
        obj = super().__new__(cls, text)
        obj.cost_usd = cost_usd
        return obj


LLM = Callable[[str, str], Awaitable[str]]


def reply_cost(reply: object) -> float | None:
    return getattr(reply, "cost_usd", None)


def litellm_llm(model: str, prices: tuple[float, float] | None = None, max_tokens: int = 16000,
                **kwargs) -> LLM:
    """An LLM backed by litellm. `prices` = ($ per million input tokens, $ per
    million output tokens); without it, litellm's own price table is used and
    an unknown model's cost is None."""
    async def call(system: str, prompt: str) -> LLMReply:
        import litellm
        resp = await litellm.acompletion(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **kwargs)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        cost = None
        if prices and usage is not None:
            cost = (usage.prompt_tokens * prices[0] + usage.completion_tokens * prices[1]) / 1e6
        else:
            try:
                cost = litellm.completion_cost(completion_response=resp)
            except Exception:
                cost = None
        return LLMReply(text, cost)
    return call
