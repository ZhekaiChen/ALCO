# Project Layout Specification

This file defines the recommended repository layout for the TSP action-level LLM-RL project.

## Canonical file paths to mention in Codex prompts

Always tell Codex to read these files first when they exist:
- `AGENTS.md`
- `PLANS.md`
- `docs/architecture.md`
- `specs/project_layout.md`
- `specs/prompt_template.md`
- `specs/schemas/tsp_instance.schema.json`
- `specs/schemas/rollout_state.schema.json`
- `specs/schemas/rollout_step.schema.json`
- `specs/schemas/episode_log.schema.json`
- `tests/fixtures/tsp_instance_minimal.json`
- `tests/fixtures/rollout_state_prefix3.json`
- `tests/fixtures/rollout_step_valid.json`
- `tests/fixtures/rollout_step_invalid_repeated_node.json`
- `tests/fixtures/episode_log_success.json`
- `tests/fixtures/sample_prompt_step1.txt`
- `tests/fixtures/sample_model_output_valid.txt`
- `tests/fixtures/sample_model_output_invalid.txt`

## Recommended repository tree

```text
repo_root/
├── AGENTS.md
├── PLANS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── paths.yaml
│   ├── tsp_generation.yaml
│   ├── lkh.yaml
│   ├── prompts.yaml
│   ├── zeroshot_eval.yaml
│   ├── sft.yaml
│   └── rl.yaml
├── docs/
│   └── architecture.md
├── specs/
│   ├── project_layout.md
│   ├── prompt_template.md
│   └── schemas/
│       ├── tsp_instance.schema.json
│       ├── rollout_state.schema.json
│       ├── rollout_step.schema.json
│       └── episode_log.schema.json
├── scripts/
│   ├── setup_env.sh
│   ├── setup_lkh.sh
│   ├── generate_tsp_data.py
│   ├── solve_tsp_with_lkh.py
│   ├── run_zeroshot_rollout.py
│   ├── extract_sft_data.py
│   ├── run_sft.py
│   ├── run_rl.py
│   └── evaluate_model.py
├── src/
│   └── tsp_action_rl/
│       ├── __init__.py
│       ├── config/
│       ├── data/
│       ├── solvers/
│       ├── prompts/
│       ├── parsing/
│       ├── rollout/
│       ├── inference/
│       ├── sft/
│       ├── rl/
│       ├── evaluation/
│       └── utils/
├── tests/
│   ├── fixtures/
│   │   ├── tsp_instance_minimal.json
│   │   ├── rollout_state_prefix3.json
│   │   ├── rollout_step_valid.json
│   │   ├── rollout_step_invalid_repeated_node.json
│   │   ├── episode_log_success.json
│   │   ├── sample_prompt_step1.txt
│   │   ├── sample_model_output_valid.txt
│   │   └── sample_model_output_invalid.txt
│   ├── test_schema_validation.py
│   ├── test_parser.py
│   ├── test_lkh_wrapper.py
│   └── test_rollout_env.py
├── third_party/
│   ├── slime/
│   └── LKH3/
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
└── outputs/
    ├── zeroshot/
    ├── sft/
    ├── rl/
    └── eval/
```

## Directory responsibilities

### `configs/`
Experiment configuration files. Keep constants out of code when practical.

### `docs/`
Narrative design explanations. Human-readable, stable references.

### `specs/`
Static protocol specifications, prompt contracts, and JSON schemas.

### `scripts/`
Command-line entrypoints. Keep them thin; most logic belongs in `src/`.

### `src/tsp_action_rl/`
Project-owned implementation.

### `tests/fixtures/`
Persistent protocol examples. These are part of the project contract, not disposable samples.

### `third_party/`
Version-pinned external source trees. Do not mix their code into project-owned modules.

### `data/`
Generated instances and caches. Usually not committed in full.

### `outputs/`
Logs, traces, training artifacts, reports, and debug outputs.
