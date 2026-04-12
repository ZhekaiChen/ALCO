#!/usr/bin/env python3
"""Generate a reproducible random Euclidean TSP instance and save JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from tsp_action_rl.data import generate_random_euclidean_instance, save_tsp_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-count", type=int, required=True, help="Number of nodes (>=2).")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--coord-min", type=float, default=0.0, help="Minimum coordinate value.")
    parser.add_argument("--coord-max", type=float, default=1.0, help="Maximum coordinate value.")
    parser.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Optional explicit instance id. Defaults to deterministic id from node_count+seed.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance = generate_random_euclidean_instance(
        node_count=args.node_count,
        seed=args.seed,
        coordinate_range=(args.coord_min, args.coord_max),
        instance_id=args.instance_id,
    )
    save_tsp_instance(instance, args.output_path)
    print(f"Saved {instance.instance_id} ({instance.node_count} nodes) to {args.output_path}")


if __name__ == "__main__":
    main()

