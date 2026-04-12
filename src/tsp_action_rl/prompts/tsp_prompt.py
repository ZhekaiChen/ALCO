"""Prompt rendering for action-level TSP zero-shot rollout."""

from __future__ import annotations

from dataclasses import dataclass

from tsp_action_rl.data import RolloutState, TSPInstance


@dataclass(frozen=True)
class PromptRenderConfig:
    """Prompt rendering options."""

    include_current_position: bool = True
    include_visited_nodes: bool = False
    include_unvisited_nodes: bool = True


def _format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def render_tsp_next_node_prompt(
    *,
    instance: TSPInstance,
    state: RolloutState,
    config: PromptRenderConfig,
) -> str:
    """Render a step prompt that requests exactly one next-node action."""
    lines: list[str] = [
        "You are solving a Traveling Salesman Problem step by step.",
        "Your job in this step is not to output the full tour, but to choose exactly one next node to append to the current partial route.",
        "Think carefully about the global tour quality before deciding.",
        "Provide detailed reasoning about the overall route structure and the tradeoffs among candidate next nodes.",
        "At the end, output exactly one final answer in the format:",
        "<FINAL_NEXT_NODE>INTEGER</FINAL_NEXT_NODE>",
        "Do not output multiple final tags.",
        "",
        "Problem setup:",
        f"- Number of nodes: {instance.node_count}",
        "- Node coordinates:",
    ]
    for node in instance.nodes:
        lines.append(f"  {node.node_id}: ({_format_float(node.x)}, {_format_float(node.y)})")

    lines.extend(
        [
            "",
            "Current partial route:",
            str(list(state.partial_route)),
        ]
    )

    if config.include_visited_nodes:
        lines.extend(
            [
                "",
                "Visited nodes:",
                str(list(state.visited_nodes)),
            ]
        )

    if config.include_unvisited_nodes:
        lines.extend(
            [
                "",
                "Unvisited nodes:",
                str(list(state.unvisited_nodes)),
            ]
        )

    lines.extend(["", "Current node:"])
    if state.current_node is None:
        lines.append("None")
    elif config.include_current_position and state.current_position is not None:
        lines.append(
            f"{state.current_node} at ({_format_float(state.current_position.x)}, {_format_float(state.current_position.y)})"
        )
    else:
        lines.append(str(state.current_node))

    lines.extend(["", "Choose the next node."])
    return "\n".join(lines)

