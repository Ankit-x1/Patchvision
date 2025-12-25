#!/usr/bin/env python3
"""
PatchVision - Main Entry Point
Industrial Vision Processing Framework
"""

import argparse
import sys
import yaml

from patchvision.cli.inference import run_inference
from patchvision.cli.visualize import run_visualization
from patchvision.cli.serve import run_server

def main():
    parser = argparse.ArgumentParser(description="PatchVision Industrial Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/industrial.example.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["train", "inference", "visualize", "serve"],
        help="Operation mode",
    )
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config}'")
        sys.exit(1)


    print(f"PatchVision v1.0.0")
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")

    if args.mode == "inference":
        run_inference(args, config)
    elif args.mode == "visualize":
        run_visualization(args, config)
    elif args.mode == "serve":
        run_server(args, config)
    else:
        print(f"Mode '{args.mode}' not implemented yet")
        sys.exit(1)


if __name__ == "__main__":
    main()
