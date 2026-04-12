#!/usr/bin/env python3
"""Run LKH full solve or fixed-prefix constrained completion on a TSP instance JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsp_action_rl.config import load_lkh_settings
from tsp_action_rl.data import load_tsp_instance
from tsp_action_rl.solvers import LKHIntegration


def parse_partial_route(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-path", type=Path, required=True, help="Path to TSP instance JSON.")
    parser.add_argument(
        "--mode",
        choices=("reference", "constrained"),
        default="reference",
        help="reference: full solve from scratch; constrained: preserve fixed prefix.",
    )
    parser.add_argument(
        "--partial-route",
        type=str,
        default="",
        help="Comma-separated node ids for constrained mode, e.g. '1,2,3'.",
    )
    parser.add_argument("--lkh-config", type=Path, default=Path("configs/lkh.yaml"), help="LKH YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance = load_tsp_instance(args.instance_path)
    settings = load_lkh_settings(args.lkh_config)
    solver = LKHIntegration(settings)

    if args.mode == "reference":
        result = solver.solve_reference(instance)
    else:
        partial_route = parse_partial_route(args.partial_route)
        if not partial_route:
            raise ValueError("--partial-route is required in constrained mode.")
        result = solver.solve_with_fixed_prefix(instance, partial_route)

    payload = {
        "mode": result.mode,
        "tour": list(result.tour),
        "tour_length": result.tour_length,
        "fixed_edges": [list(edge) for edge in result.fixed_edges],
        "solver_params": result.solver_params,
        "debug_paths": result.debug_paths,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

