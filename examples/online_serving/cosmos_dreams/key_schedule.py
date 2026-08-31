# SPDX-License-Identifier: Apache-2.0
"""Emit raw AgiBot ``[16,29]`` chunks from deterministic keyboard schedules."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from controller import (
    AgiBotKeyboardController,
    build_scheduled_action_chunk,
)
from demo_support import load_scene_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="Versioned AgiBot scene JSON bundle.")
    parser.add_argument(
        "--schedule",
        required=True,
        help="Held-key schedule such as 'right+w:8,space:1,none:7' (must total 16 frames).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .pt file containing float32[16,29].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = AgiBotKeyboardController(load_scene_bundle(args.scene))
    action = build_scheduled_action_chunk(controller, args.schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(action, args.output)
    print(f"Saved raw AgiBot action {tuple(action.shape)} to {args.output}")


if __name__ == "__main__":
    main()
