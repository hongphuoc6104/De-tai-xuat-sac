"""Operator entry point for frozen features; no VM or training operations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .features import FeatureRunConfig, run_feature_extraction, verify_feature_release


def main(argv: Sequence[str] | None = None) -> int:
    """Run, inspect or verify feature outputs with meaningful exit codes."""
    parser = argparse.ArgumentParser(prog="python -m histology_data.feature_cli")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="extract available ZIP/directory parts and resume verified output")
    run.add_argument("--config", type=Path, required=True)
    status = commands.add_parser("status", help="read durable progress without starting computations")
    status.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify features/index and scan duplicate content across parts")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--feature-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            values = json.loads(args.config.read_text(encoding="utf-8"))
            result = run_feature_extraction(FeatureRunConfig.from_dict(values))
        elif args.command == "status":
            result = json.loads((args.output / "status.json").read_text(encoding="utf-8"))
        else:
            result = verify_feature_release(args.output, args.feature_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get("status") == "error" else 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
