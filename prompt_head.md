Read these files first before making changes:

- AGENTS.md
- PLANS.md
- docs/architecture.md
- specs/project_layout.md
- specs/prompt_template.md
- specs/schemas/tsp_instance.schema.json
- specs/schemas/rollout_state.schema.json
- specs/schemas/rollout_step.schema.json
- specs/schemas/episode_log.schema.json
- tests/fixtures/tsp_instance_minimal.json
- tests/fixtures/rollout_state_prefix3.json
- tests/fixtures/rollout_step_valid.json
- tests/fixtures/rollout_step_invalid_repeated_node.json
- tests/fixtures/episode_log_success.json
- tests/fixtures/sample_prompt_step1.txt
- tests/fixtures/sample_model_output_valid.txt
- tests/fixtures/sample_model_output_invalid.txt

Important:
- Follow AGENTS.md as the project invariant source of truth.
- Treat tests/fixtures/ as persistent protocol examples, not disposable samples.
- Keep third_party code isolated.
- Do not overengineer with speculative abstraction layers.
- Fail fast at system boundaries and keep core logic simple.