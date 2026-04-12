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

## Implementation round: 2026-04-10 (Phase 0/1 foundations)

### Goal
Create the minimum runnable foundation for repository alignment and the TSP data layer without pretending solver or rollout completeness.

### Scope
- Align missing skeleton directories/files with `specs/project_layout.md`.
- Implement typed TSP instance and rollout state models with strict boundary validation.
- Implement reproducible random Euclidean TSP generation and stable JSON I/O.
- Add smoke tests against persistent fixtures.

### Out of scope
- LKH3 full solve implementation and constrained completion behavior.
- Prompt parser, rollout loop, SFT mining, RL environment, and evaluation metrics implementation.

### Affected files
- `pyproject.toml`
- `configs/`
- `scripts/generate_tsp_data.py`
- `src/tsp_action_rl/`
- `tests/test_data_layer.py`
- `third_party/slime/`
- `third_party/LKH3/`

### Open decisions / TODOs
- Whether `step_index` should be interpreted as `len(partial_route)` or action count from an initial seeded node.
- Final schema-validation backend choice (`jsonschema` runtime validation vs. strict project-native validators only).
- Exact config loading stack and dependency policy for YAML-heavy experiment configs.

### Validation
- Fixture loading for `tsp_instance_minimal.json` and `rollout_state_prefix3.json`.
- Deterministic generation with fixed seed.
- 1-based contiguous node-id enforcement.
- JSON save/load round-trip stability.

## Implementation round: 2026-04-10 (Phase 2 LKH integration)

### Goal
Implement the primary Python-side LKH integration using the `lkh` package with explicit full-solve and fixed-prefix constrained-completion paths.

### Scope
- Add project-owned LKH integration module under `src/tsp_action_rl/solvers/`.
- Add explicit solver config support with configurable executable path.
- Implement full reference solve path and constrained completion path.
- Persist inspectable debug artifacts when debug mode is enabled.
- Add smoke tests for integration behavior with mocked `lkh.solve(...)`.
- Add short integration notes and usage examples.

### Out of scope
- Reward formula design and rollout loop integration.
- RL training integration details.
- Advanced solver fallback backends.

### Affected files
- `configs/lkh.yaml`
- `src/tsp_action_rl/config/lkh.py`
- `src/tsp_action_rl/config/__init__.py`
- `src/tsp_action_rl/solvers/lkh_integration.py`
- `src/tsp_action_rl/solvers/__init__.py`
- `scripts/solve_tsp_with_lkh.py`
- `tests/test_lkh_wrapper.py`
- `README.md`

### Assumptions
- Pinned local archive path is canonical: `third_party/LKH3/LKH-3.0.14.tar`.
- LKH executable exists and is reachable by configured path.
- `lkh.solve(...)` is the primary integration entrypoint.
- Fixed-prefix constraints are expressed using `FIXED_EDGES_SECTION` for prefix edges.

### Open decisions / TODOs
- Whether to enforce strict directed-prefix semantics only via fixed edges or add stronger sequence constraints if future evidence requires it.
- Whether to add optional raw solver stdout capture beyond current `lkh` package capabilities.

### Validation
- Full-solve smoke test with mocked `lkh` module.
- Constrained-solve smoke test verifies fixed-edge serialization and prefix preservation check.
- Negative test verifies explicit failure when constrained result violates required prefix.

## Implementation round: 2026-04-11 (Phase 2.5 local LKH build + real validation)

### Goal
Build LKH-3 locally from the pinned archive and validate the existing Python `lkh.solve(...)` integration end-to-end with the real solver.

### Scope
- Inspect and extract `third_party/LKH3/LKH-3.0.14.tar` into `third_party/LKH3/LKH-3.0.14/`.
- Build the local solver executable and pin its path explicitly in config/docs.
- Keep the current solver integration architecture and validate it with real runs.
- Add real integration tests (separate from mocked unit tests).
- Verify debug artifact generation and fixed-prefix constrained completion behavior.

### Out of scope
- Solver architecture refactors.
- Zero-shot rollout/SFT/RL implementation.
- Any fallback backend strategy expansion.

### Affected files
- `PLANS.md`
- `configs/lkh.yaml`
- `scripts/setup_lkh.sh`
- `tests/test_lkh_real_integration.py`
- `README.md`

### Assumptions
- Canonical pinned archive path remains `third_party/LKH3/LKH-3.0.14.tar`.
- Local toolchain for `make` build is available.
- Expected executable output path is `third_party/LKH3/LKH-3.0.14/LKH`.

### Validation
- Build succeeds and executable exists at configured path.
- Python `lkh` package import works.
- Real reference solve passes on a small generated instance.
- Real constrained solve passes and preserves ordered fixed prefix.
- Debug mode emits inspectable solver artifacts.

## Implementation round: 2026-04-11 (Phase 3 zero-shot rollout harness)

### Goal
Implement prompt/render/parse/model-IO/rollout/logging infrastructure for zero-shot action-level next-node prediction on TSP.

### Scope
- Add a common inference interface for local and API-style model backends.
- Implement strict prompt rendering from rollout state and instance data.
- Implement strict final-tag parsing and action validation.
- Implement zero-shot episode runner with state transitions and structured step/episode logs.
- Add CLI entrypoint for running one or more zero-shot episodes and writing logs under `outputs/`.
- Add parser tests and rollout smoke tests.

### Out of scope
- SFT trace extraction.
- RL training/environment updates.
- Reward design experimentation beyond placeholder-compatible logging fields.
- Solver-layer redesign.

### Affected files
- `PLANS.md`
- `configs/zeroshot_eval.yaml`
- `src/tsp_action_rl/inference/`
- `src/tsp_action_rl/prompts/`
- `src/tsp_action_rl/parsing/`
- `src/tsp_action_rl/rollout/`
- `scripts/run_zeroshot_rollout.py`
- `tests/test_parser.py`
- `tests/test_rollout_env.py`
- `README.md`

### Assumptions
- Node ids remain TSPLIB 1-based in all public-facing prompt/log outputs.
- `partial_route` is always treated as an ordered fixed prefix.
- Local deterministic backend is acceptable for runnable zero-shot harness smoke tests.
- API backend may remain TODO if it fails clearly and explicitly.

### Validation
- Parser tests pass for valid/invalid fixture model outputs.
- Prompt output contains required fields per `specs/prompt_template.md`.
- Rollout smoke tests pass for successful episode and invalid-action failure paths.
- CLI runs at least one full episode and writes structured logs under `outputs/zeroshot/`.

## Implementation round: 2026-04-11 (Phase 3.5 real API backend + real zero-shot traces)

### Goal
Integrate a real DMXAPI OpenAI-compatible backend and support configurable real zero-shot trace collection runs.

### Scope
- Implement one real DMXAPI backend under the existing model interface.
- Read API credentials/settings from env vars and/or config (no secrets in code/config docs).
- Keep request format OpenAI-compatible and default model `claude-opus-4-6-thinking` with thinking effort `high`.
- Extend zero-shot run configuration for node-count sweeps, integer coordinate ranges, random start policy, rollout step policy, and auto-complete behavior.
- Collect real structured logs under `outputs/zeroshot/` suitable for later SFT trace extraction.

### Out of scope
- SFT extraction implementation.
- RL training implementation.
- Solver architecture redesign.

### Affected files
- `PLANS.md`
- `configs/zeroshot_eval.yaml`
- `src/tsp_action_rl/inference/`
- `src/tsp_action_rl/rollout/`
- `scripts/run_zeroshot_rollout.py`
- `tests/`
- `README.md`

### Assumptions
- DMXAPI endpoint is OpenAI-compatible (`/chat/completions`).
- API key is provided at runtime via env var.
- Each rollout step remains a stateless request carrying full structured prompt context.

### Validation
- DMX backend unit tests for request/response handling pass.
- Existing parser strictness remains unchanged.
- Rollout smoke tests pass with updated configurable policies.
- CLI supports node counts `10,25,50`, integer coordinates in `[0,10000]`, random start policy, and logs structured outputs.

### Phase 3.5 execution notes (narrow operational path)
- Keep `DMXOpenAICompatibleBackend` as the single real API backend in this round.
- Keep parser strictness unchanged: only `<FINAL_NEXT_NODE>INTEGER</FINAL_NEXT_NODE>`.
- Extend rollout config/runner for:
  - random start-node policy,
  - prediction-step budget policy (`node_count_minus_2` default for real runs),
  - optional auto-complete of the final remaining node.
- Extend CLI/config for matrix runs over `node_counts` and `instances_per_node_count`.
- Keep LKH layer unchanged and only wire existing `LKHIntegration` into real zero-shot runs.

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
