"""Strict final-tag parser tests."""

from __future__ import annotations

from pathlib import Path

from tsp_action_rl.parsing import parse_final_next_node

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_parse_valid_fixture_output() -> None:
    raw = (FIXTURES_DIR / "sample_model_output_valid.txt").read_text(encoding="utf-8")
    parsed = parse_final_next_node(raw)
    assert parsed.status == "success"
    assert parsed.tag_count == 1
    assert parsed.parsed_next_node == 5
    assert "<FINAL_NEXT_NODE>" not in parsed.reasoning_text
    assert "main candidates are node 4 and node 5" in parsed.reasoning_text


def test_parse_invalid_action_fixture_output_still_parses_tag() -> None:
    raw = (FIXTURES_DIR / "sample_model_output_invalid.txt").read_text(encoding="utf-8")
    parsed = parse_final_next_node(raw)
    assert parsed.status == "success"
    assert parsed.tag_count == 1
    assert parsed.parsed_next_node == 2


def test_parse_missing_tag() -> None:
    parsed = parse_final_next_node("I reason about tradeoffs but forget the final action tag.")
    assert parsed.status == "missing_tag"
    assert parsed.tag_count == 0
    assert parsed.parsed_next_node is None


def test_parse_multiple_tags() -> None:
    raw = "<FINAL_NEXT_NODE>4</FINAL_NEXT_NODE>\n<FINAL_NEXT_NODE>5</FINAL_NEXT_NODE>"
    parsed = parse_final_next_node(raw)
    assert parsed.status == "multiple_tags"
    assert parsed.tag_count == 2
    assert parsed.parsed_next_node is None


def test_parse_malformed_tag() -> None:
    raw = "Detailed reasoning\n<FINAL_NEXT_NODE>node_5</FINAL_NEXT_NODE>"
    parsed = parse_final_next_node(raw)
    assert parsed.status == "malformed_tag"
    assert parsed.tag_count == 1
    assert parsed.parsed_next_node is None

