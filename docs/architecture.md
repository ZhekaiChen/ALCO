# Architecture Overview

This project studies action-level large-language-model reasoning and reinforcement learning for the Traveling Salesman Problem (TSP).

The central environment loop is:
1. generate or load a TSP instance,
2. compute an LKH3 reference solution from scratch,
3. prompt a language model with the current partial route,
4. parse exactly one next-node action from the final tag,
5. validate the action,
6. call LKH3 again with the fixed prefix enforced,
7. compute reward from the completed constrained tour,
8. continue until a complete route is obtained or the episode fails.

## Major subsystem boundaries

### `third_party/slime/`
The imported THU-DM SLIME framework.
Keep it isolated. Project-specific training logic should not be spread inside SLIME unless absolutely necessary.

### `third_party/LKH3/`
The LKH3 source tree and local build artifacts.
This repository should use transparent wrapper code that writes problem files, parameter files, and captures solver outputs.

### `src/tsp_action_rl/data/`
Data classes, serializers, and validation utilities for:
- TSP instances,
- rollout states,
- rollout step logs,
- episode logs,
- SFT export records.

### `src/tsp_action_rl/solvers/`
Solver wrappers for:
- full reference solve,
- constrained completion using the fixed-route mechanism,
- parsing solver outputs,
- debug artifact handling.

### `src/tsp_action_rl/prompts/`
Prompt rendering logic.
This layer converts structured rollout state into text prompts for zero-shot, SFT, and RL-adjacent inference workflows.

### `src/tsp_action_rl/parsing/`
Structured extraction of the final action tag and validation of the model output.
The parser must not depend on free-form reasoning text.

### `src/tsp_action_rl/rollout/`
The step-level and episode-level execution engine.
This area owns:
- state transitions,
- action validation,
- logging,
- done conditions,
- reward interface calls.

### `src/tsp_action_rl/sft/`
Mining high-quality reasoning traces and exporting them into a small SFT dataset.

### `src/tsp_action_rl/rl/`
Adapters between the project environment and the SLIME training stack.
Do not mix solver details directly into generic RL training code.

### `src/tsp_action_rl/evaluation/`
Metric definitions, reports, and reusable evaluation drivers.

### `tests/fixtures/`
Frozen examples for schemas, sample prompts, sample outputs, and tiny smoke-test problem instances.
These fixtures serve as persistent protocol references.

## Design invariants

- Public node ids are 1-based.
- `partial_route` is a fixed ordered prefix.
- Reward is online, not fully precomputed.
- Reasoning text is preserved as a research artifact.
- Only the final tagged node is used as the explicit action.
- Reward logic stays configurable.

## Recommended code growth order

1. Static specs and fixtures
2. Data schemas and serializers
3. Transparent LKH3 full-solve wrapper
4. Transparent constrained-completion wrapper
5. Prompt/render/parse pipeline
6. Rollout episode driver and logs
7. SFT trace mining
8. RL integration with SLIME
