# Methods: what harness-evolve does, and where each idea comes from

Each part of the optimizer, the file it lives in, the source it comes from, and what we changed. Settings
are fields of `OptimizerConfig` (`loop.py`) or `AcceptanceConfig` (`acceptance.py`); "off by default" means
you turn it on there.

## The loop (`loop.py`)

Round 0 runs the starting harness on the evolve cases (2 trials each) and calibrates the noise band. Each
round then proposes one edit, checks it for free, screens it with a critic, evaluates it (the only
expensive step), and decides with fixed rules. An optional final phase compares the original and evolved
harness on held-out cases. Every phase is checkpointed; every case result is cached by the candidate's
content hash, so a stopped run resumes without paying twice.

## Proposing an edit (`proposer.py`, `evidence.py`)

| What | How | Source |
|---|---|---|
| Reflective, trace-driven proposals | The proposer reads the agent's trajectories (tool calls, outputs, verdicts, cost) and the edit history, diagnoses what went wrong in words, and targets an edit at that failure, instead of mutating a score blindly | GEPA: Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", ICLR 2026. Ours is GEPA-style (hand-built; no GEPA library, no candidate pool) |
| One mechanism per candidate | Each candidate is one component, one hypothesis, one mechanism, so every measured change traces to one idea | RRSI's annealed edit budget at its floor (b = 1): Xia et al., "RRSI: Regularized Recursive Self-Improvement of Agent Harnesses", arXiv 2609.24972 |
| Two candidates per round | The proposer returns two alternatives with different mechanisms; if the critic rejects the preferred one, the other is tried | RRSI (two proposals per round) |
| Search/replace edits | Edits are exact find/replace operations on harness files, not full rewrites, so a long prompt isn't reproduced (and silently drifted) to change one paragraph | Ours |
| Settings shown as behaviour | Each setting appears with what it does and its allowed range | Ours: in our first runs the proposer never touched settings until they were shown this way |
| Answers kept from the proposer | Anything after `HIDDEN_MARKER` in a trajectory's `detail` is cut before the proposer sees it | Ours |
| Agent self-feedback | If trajectories include `self_feedback` (the agent's own words about the setup after a failing run), it is shown as evidence | Idea from HarnessCompass, arXiv 2608.01918 ("proactive first-person feedback") |
| Cost by prompt source | If trajectories include `cost_by_source`, the proposer sees where the money goes and is told to change what the harness asks for, never to ask the agent to be brief | Ours |

## Screening an edit (`candidates.py`, `critic.py`)

| What | How | Source |
|---|---|---|
| Free structural checks | A candidate must keep the same files, valid JSON, the same setting keys and types, settings within their bounds (ranges or allowed values), the same tools, and no new template placeholders, before anything is spent | RRSI's deterministic checks after the critic |
| Leakage critic | A regex precheck (credentials, task ids, names from the evaluation data) then an LLM review with reject classes: task specialization, degenerate or safety-weakening changes, grader gaming, undeclared bundling, memory leakage, unbounded work, near-duplicates of rejected ideas, false tool claims. Rejections go back to the proposer for a bounded number of repairs | Adapted from RRSI's critic (google-research/rrsi, Apache-2.0); the instructions are generated from your agent's profile |

## Deciding (`acceptance.py`)

| Rule | How | Source |
|---|---|---|
| Noise band | δ = 2 × the bootstrapped standard deviation of the pass-rate change between two runs of the unchanged harness (round 0's trial pairs) | RRSI (`calibrate.py`) |
| Floor | Never accept a candidate clearly worse than the best so far: S′ ≥ S* − δ | RRSI |
| Clear gain | A gain beyond δ may cost more, up to 10% plus 44.5% per point gained | RRSI's coding settings (β₀ = 0.10, β₁ = 44.5) |
| Inside the band | A score change counts for nothing (w_s = 0); the candidate wins only if it is cheaper by more than the cost noise band | RRSI (w_s = 0); the cost band is ours, calibrated the same way from trial-pair costs (`cost_band`, off by default) |
| Give-up guard | Veto if the share of runs that never submit an answer rises by more than 5 points or its own noise band (calibrated from round 0's trial pairs), whichever is larger | RRSI's domain veto on a rising no-submission rate |
| Guard cases | Veto if a guard case (one that must keep passing) fails on every trial of the candidate while the incumbent passed it at least once. A single failed trial isn't enough: guards that pass 80–95% of the time would veto about a third of candidates by chance | Ours |
| Cost only on shared successes | Compare cost only on cases both harnesses solve, so a cheaper failure never counts as a saving (`cost_on_shared_successes`, off by default) | Idea from the Hugging Face / FineEnvs "Multi-harness RL" article: efficiency is rewarded only on correct answers |
| Near-miss archive | Candidates just inside the noise band are kept and shown to later rounds to build on (`near_miss_fraction`, off by default) | Idea from the Darwin Gödel Machine (arXiv 2505.22954) and Huxley-Gödel Machine (arXiv 2510.21614): the best-scoring variant isn't always the best one to build on |
| Two-model check | An accepted edit must also hold on a second model's evaluation (`transfer_evaluator`, optional) | Ours, motivated by our finding that a prompt edit evolved for one model did nothing on another |
| Exploration rule | After N rejected candidates in a row, the next must change a component family none of them touched (`diversity_after`, off by default) | Simplified from RRSI's component novelty |

## Reporting (`report.py`, `pass_k.py`)

| What | How | Source |
|---|---|---|
| Held-out comparison | Original and evolved harness on held-out cases, k ≥ 2 trials each | Standard practice; RRSI shows why (an unregularized evolved harness scored 92.8 on its own tasks and 40.3 held out) |
| Rerun baseline | The evolved harness's single run is compared with the original's best-of-2, estimated without bias from its own k trials. An evolved harness that costs one run should beat that, or you could have just rerun | AI2: Wang et al., "Rethinking the Evaluation of Harness Evolution for Agents", arXiv 2607.12227 |
| pass@k and pass^k | pass@k = chance at least one of k tries passes; pass^k = chance all k pass (reliability) | Unbiased estimators as in Chen et al. (HumanEval, 2021); pass^k as used in tau-bench (Yao et al., 2024) |
| Paired bootstrap | Each difference gets a 95% interval from a bootstrap over cases | Standard |

## Running unattended (`loop.py`, `health.py`, `budget.py`, `state.py`)

All ours, from running the optimizer for hours unattended: a hard budget checked before every case;
provider failures (billing, auth, rate limits) stop the run instead of being scored as failures; an
unpriced model reply stops the run instead of spending blind; tripwires pause it when measurements look
broken (a tool failing far above normal, $0 costs, errored runs); a watchdog exits cleanly if it stops
making progress; an optional smoke stage checks a few cases before real spend.

## Benchmarking the optimizer (`bench.py`)

Known-answer scenarios with a scripted proposer and critic (a real fix must be accepted, a harmful edit and
a neutral rewording must not), plus open scenarios with the real proposer. Ours. Lesson from the first run:
measure a planted edit's effect before treating it as a known answer.
