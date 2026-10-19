"""
harness-evolve: improve an agent's harness from its own failure traces.

    from harness_evolve import AgentProfile, Optimizer, OptimizerConfig, profiles
    profiles.use(AgentProfile.build(name=..., harness_root=..., agent_description=...))
    opt = Optimizer(run_dir, base_harness, OptimizerConfig(evolve_cases=[...]),
                    evaluator=MyEvaluator(), proposer_llm=llm, critic_llm=llm, critic_patterns=[])
    state = asyncio.run(opt.run_until_stopped())
"""
from harness_evolve.acceptance import CaseResult, EvalResult
from harness_evolve.evaluator import EvalOutcome, Evaluator, ProviderFailure, UnpricedModel
from harness_evolve.harness import Harness, load_harness
from harness_evolve.llm import LLMReply, litellm_llm
from harness_evolve.loop import LocalPublisher, Optimizer, OptimizerConfig
from harness_evolve.profiles import AgentProfile

__all__ = ["AgentProfile", "CaseResult", "EvalOutcome", "EvalResult", "Evaluator", "Harness", "LLMReply",
           "LocalPublisher", "Optimizer", "OptimizerConfig", "ProviderFailure", "UnpricedModel",
           "litellm_llm", "load_harness"]
__version__ = "0.3.0"
