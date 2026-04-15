"""Phase-4 smoke tests for SFT trace mining and export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tsp_action_rl.sft import (
    TraceFilterConfig,
    build_chat_style_examples,
    build_export_summary,
    build_internal_sft_examples,
    discover_run_directories,
    filter_step_traces,
    load_step_traces_from_runs,
    write_jsonl,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _build_fixture_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "sft_fixture_run"
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES_DIR / "sft_run_summary.json", run_dir / "run_summary.json")
    shutil.copyfile(FIXTURES_DIR / "sft_episode_valid.json", episodes_dir / "sft_episode_valid.json")
    shutil.copyfile(FIXTURES_DIR / "sft_episode_invalid.json", episodes_dir / "sft_episode_invalid.json")
    return run_dir


def _build_episode_only_run_dir(tmp_path: Path, *, root_name: str) -> Path:
    root = tmp_path / root_name
    run_dir = root / "episode_only_run"
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES_DIR / "sft_episode_valid.json", episodes_dir / "sft_episode_valid.json")
    return root


def test_load_and_index_step_traces_with_source_mapping(tmp_path: Path) -> None:
    run_dir = _build_fixture_run_dir(tmp_path)

    discovered = discover_run_directories(run_directories=[run_dir], root_directories=[])
    assert discovered == [run_dir.resolve()]

    corpus = load_step_traces_from_runs(discovered)
    assert corpus.episodes_scanned == 2
    assert corpus.steps_scanned == 2
    assert "sft_fixture_run" in corpus.run_names

    by_episode = {record.episode_id: record for record in corpus.step_records}
    valid = by_episode["sft_episode_valid"]
    invalid = by_episode["sft_episode_invalid"]

    assert valid.run_name == "sft_fixture_run"
    assert valid.instance_id == "tsp_fixture_10"
    assert valid.node_count == 10
    assert valid.step_index == 1
    assert valid.model_id == "claude-opus-4-6-thinking"
    assert valid.source_mapping()["episode_path"].endswith("sft_episode_valid.json")

    # Step-level model_id is missing in invalid fixture, so it falls back to run summary config.api.model_id.
    assert invalid.model_id == "claude-opus-4-6-thinking"


def test_filter_logic_supports_quality_and_metadata_constraints(tmp_path: Path) -> None:
    run_dir = _build_fixture_run_dir(tmp_path)
    corpus = load_step_traces_from_runs([run_dir])

    default_result = filter_step_traces(corpus.step_records, TraceFilterConfig())
    assert len(default_result.kept_records) == 1
    assert default_result.kept_records[0].episode_id == "sft_episode_valid"
    assert default_result.dropped_reason_counts["require_valid_action"] == 1

    strict_result = filter_step_traces(
        corpus.step_records,
        TraceFilterConfig(
            require_parse_success=True,
            require_valid_action=True,
            require_episode_success=True,
            require_solver_completion=True,
            max_final_gap_to_reference=0.02,
            max_step_gap_to_reference=0.03,
            node_counts=(10,),
            model_ids=("claude-opus-4-6-thinking",),
            min_reasoning_chars=5,
        ),
    )
    assert len(strict_result.kept_records) == 1

    filtered_out = filter_step_traces(
        corpus.step_records,
        TraceFilterConfig(model_ids=("some-other-model",)),
    )
    assert len(filtered_out.kept_records) == 0
    assert filtered_out.dropped_reason_counts["model_ids"] >= 1


def test_discover_run_directories_checks_episode_fallback_per_root(tmp_path: Path) -> None:
    run_dir = _build_fixture_run_dir(tmp_path / "root_with_summary")
    episode_only_root = _build_episode_only_run_dir(tmp_path, root_name="root_with_episode_only")

    discovered = discover_run_directories(
        root_directories=[run_dir.parent, episode_only_root],
        run_directories=[],
    )
    discovered_paths = {str(path) for path in discovered}
    assert str(run_dir.resolve()) in discovered_paths
    assert str((episode_only_root / "episode_only_run").resolve()) in discovered_paths


def test_export_formats_and_summary_report(tmp_path: Path) -> None:
    run_dir = _build_fixture_run_dir(tmp_path)
    corpus = load_step_traces_from_runs([run_dir])
    filtered = filter_step_traces(corpus.step_records, TraceFilterConfig())
    kept = filtered.kept_records
    assert len(kept) == 1

    internal_rows = build_internal_sft_examples(kept)
    assert len(internal_rows) == 1
    row = internal_rows[0]
    assert row["prompt_text"] == "Prompt for step 1"
    assert row["reasoning_text"] == "Reason about route globally."
    assert row["final_tag_output"] == "<FINAL_NEXT_NODE>2</FINAL_NEXT_NODE>"
    assert row["next_node_label"] == 2
    assert row["source"]["run_name"] == "sft_fixture_run"
    assert row["source"]["episode_id"] == "sft_episode_valid"

    chat_rows = build_chat_style_examples(kept)
    assert len(chat_rows) == 1
    assert chat_rows[0]["messages"][0]["role"] == "user"
    assert chat_rows[0]["messages"][1]["role"] == "assistant"
    assert "<FINAL_NEXT_NODE>2</FINAL_NEXT_NODE>" in chat_rows[0]["messages"][1]["content"]

    export_path = tmp_path / "sft_examples.jsonl"
    write_jsonl(internal_rows, export_path)
    saved_lines = export_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(saved_lines) == 1
    saved = json.loads(saved_lines[0])
    assert saved["next_node_label"] == 2
    assert saved["source"]["step_index"] == 1

    summary = build_export_summary(
        corpus=corpus,
        filter_config=TraceFilterConfig(),
        filter_result=filtered,
        output_jsonl=str(export_path),
        output_chat_jsonl=None,
    )
    assert summary["runs_scanned"] == 1
    assert summary["episodes_scanned"] == 2
    assert summary["steps_scanned"] == 2
    assert summary["steps_kept"] == 1
    assert summary["counts_by_node_count"]["10"] == 1
    assert summary["counts_by_model_id"]["claude-opus-4-6-thinking"] == 1
    assert summary["parse_success_rate_among_kept"] == 1.0
    assert summary["valid_action_rate_among_kept"] == 1.0
