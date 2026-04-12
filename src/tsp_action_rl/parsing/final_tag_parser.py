"""Strict parser for `<FINAL_NEXT_NODE>INTEGER</FINAL_NEXT_NODE>` contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ParseStatus = Literal["success", "missing_tag", "multiple_tags", "malformed_tag"]

_OPEN_TAG = "<FINAL_NEXT_NODE>"
_CLOSE_TAG = "</FINAL_NEXT_NODE>"
_VALID_TAG_RE = re.compile(r"<FINAL_NEXT_NODE>\s*([0-9]+)\s*</FINAL_NEXT_NODE>", re.DOTALL)
_ANY_CLOSED_TAG_RE = re.compile(r"<FINAL_NEXT_NODE>.*?</FINAL_NEXT_NODE>", re.DOTALL)


@dataclass(frozen=True)
class FinalTagParseResult:
    """Parsed final-tag result and extracted reasoning text."""

    status: ParseStatus
    tag_count: int
    parsed_next_node: int | None
    reasoning_text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "tag_count": self.tag_count,
            "parsed_next_node": self.parsed_next_node,
        }


def parse_final_next_node(raw_model_output: str) -> FinalTagParseResult:
    """Parse one strict final action tag from raw model output."""
    open_count = raw_model_output.count(_OPEN_TAG)
    close_count = raw_model_output.count(_CLOSE_TAG)
    valid_matches = _VALID_TAG_RE.findall(raw_model_output)

    if open_count == 0 and close_count == 0:
        status: ParseStatus = "missing_tag"
        parsed = None
    elif open_count != close_count:
        status = "malformed_tag"
        parsed = None
    elif open_count > 1:
        status = "multiple_tags"
        parsed = None
    elif len(valid_matches) != 1:
        status = "malformed_tag"
        parsed = None
    else:
        status = "success"
        parsed = int(valid_matches[0])

    reasoning_text = _ANY_CLOSED_TAG_RE.sub("", raw_model_output).strip()
    return FinalTagParseResult(
        status=status,
        tag_count=open_count,
        parsed_next_node=parsed,
        reasoning_text=reasoning_text,
    )

