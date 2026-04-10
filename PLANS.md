# PLANS.md

Use this file for any task that touches multiple subsystems, changes architecture, or spans multiple coding rounds.

## Current execution plan

### Goal
Bootstrap a research-grade repository for action-level LLM reasoning and reinforcement learning on TSP with online LKH3-based reward generation.

### Scope of the current phase
1. Repository audit and skeleton reorganization
2. Static specification files and fixtures
3. TSP data schema definitions
4. LKH3 wrapper and constrained-completion skeleton
5. Zero-shot rollout harness skeleton

### Out of scope for the current phase
- Final reward formula selection
- Full RL algorithm tuning
- Large-scale training runs
- Performance optimization
- Multi-problem generalization beyond TSP

### Affected areas
- `AGENTS.md`
- `README.md`
- `docs/`
- `specs/`
- `tests/fixtures/`
- `src/tsp_action_rl/`
- `scripts/`
- `configs/`

### Key assumptions
- Public-facing node ids are 1-based and follow TSPLIB convention.
- `partial_route` is a fixed ordered prefix.
- Reward is computed online using constrained LKH3 completion.
- Reasoning traces are logged and preserved.
- SLIME and LKH3 stay isolated in `third_party/`.

### Open decisions
- Exact reward formula
- Whether the current node is included explicitly in every prompt
- Whether invalid actions terminate immediately or receive a recoverable penalty
- Exact local model backend and training stack details
- Exact evaluation splits and experiment registry layout

### Planned steps
1. Audit the current repository and summarize reusable code.
2. Ensure the static spec files in `docs/`, `specs/`, and `tests/fixtures/` exist.
3. Create or align the code skeleton under `src/tsp_action_rl/`.
4. Implement schema-aware serializers and validators.
5. Implement the transparent LKH3 wrapper for full solves.
6. Implement constrained-completion support using fixed-route constraints.
7. Build the rollout prompt/parser/logging pipeline.
8. Add smoke tests and example scripts.

### Validation steps
- Schema examples validate against JSON schemas.
- Sample prompt/output parsing succeeds.
- LKH3 full solve smoke test runs on a small instance.
- Constrained completion test preserves the prefix.
- Rollout step logs contain reasoning text and final tag extraction.

### Rollback notes
- Avoid invasive edits in third-party code.
- Keep skeleton changes separate from behavior changes.
- If a subsystem is incomplete, leave an honest TODO rather than a fake abstraction.

## Template for future plans

### Goal
[Describe the task in one paragraph.]

### Scope
- [In scope item 1]
- [In scope item 2]

### Out of scope
- [Out of scope item 1]
- [Out of scope item 2]

### Affected files
- `[path/to/file1]`
- `[path/to/file2]`

### Assumptions
- [Assumption 1]
- [Assumption 2]

### Open decisions / TODOs
- [Open question 1]
- [Open question 2]

### Implementation steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Validation
- [Validation check 1]
- [Validation check 2]

### Rollback notes
- [Rollback or containment note]
