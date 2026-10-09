#!/usr/bin/env python3
"""Runner and orchestrator for the 8-step Medical AI histology pipeline.

Steps:
1. Metadata Review (1_Metadata_review.py)
2. Objective Lens QC (2_QC_objective_lens.py)
3. Tissue Detection & Tiling (3_Tissue_detection.py)
4. Patient-Level Train/Val/Test Split (4_Patient_level_Split.py)
5. Images Pre-Processing (5_Images_Pre_Processing.py)
6. Strategy A Training (6_Strategy_A_training.py)
7. Model Evaluation (7_Evaluate_Model.py)
8. Patient-Level Cross-Validation (8_Patient_Level_CV_Colab.py)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = WORKSPACE_ROOT / "Script_Colab"
RESULTS_ROOT = WORKSPACE_ROOT / "Results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("medical_pipeline")

STEPS_MAP = {
    1: ("1_Metadata_review.py", "Metadata Review and Overview Extraction"),
    2: ("2_QC_objective_lens.py", "Objective Lens Aware Quality Control"),
    3: ("3_Tissue_detection.py", "Tissue Detection & Tile Extraction"),
    4: ("4_Patient_level_Split.py", "Patient-Level Disjoint Holdout Split"),
    5: ("5_Images_Pre_Processing.py", "Images Pre-Processing (CLAHE / Percentile)"),
    6: ("6_Strategy_A_training.py", "Strategy A Model Training"),
    7: ("7_Evaluate_Model.py", "Comprehensive Model Evaluation & Test Metrics"),
    8: ("8_Patient_Level_CV_Colab.py", "Patient-Level Cross-Validation"),
}


def run_step(step_num: int, raw_root: Path, results_root: Path, dry_run: bool = False, extra_args: list[str] | None = None) -> bool:
    """Execute a single step in the medical pipeline."""
    if step_num not in STEPS_MAP:
        logger.error(f"Invalid step number: {step_num}. Allowed 1..8")
        return False

    script_name, description = STEPS_MAP[step_num]
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--raw_root", str(raw_root),
        "--results_root", str(results_root),
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"=== [Step {step_num}/8] {description} ===")
    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info(f"[DRY-RUN] Step {step_num} command validated successfully.")
        return True

    results_root.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), text=True)
        if proc.returncode != 0:
            logger.error(f"Step {step_num} failed with return code {proc.returncode}.")
            return False
        logger.info(f"✓ Step {step_num} completed successfully.")
        return True
    except Exception as exc:
        logger.error(f"Step {step_num} encountered exception: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Orchestrator for the 8-step Medical AI Histology Pipeline.")
    parser.add_argument("--raw_root", type=Path, default=WORKSPACE_ROOT, help="Root directory of raw dataset")
    parser.add_argument("--results_root", type=Path, default=RESULTS_ROOT, help="Output root for results")
    parser.add_argument("--step", type=int, choices=range(1, 9), help="Run a specific step (1..8)")
    parser.add_argument("--start", type=int, default=1, help="Start step (default: 1)")
    parser.add_argument("--end", type=int, default=8, help="End step (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args, unknown = parser.parse_known_args()

    logger.info("Initializing Medical AI Pipeline Orchestrator...")
    logger.info(f"Workspace root : {WORKSPACE_ROOT}")
    logger.info(f"Raw root       : {args.raw_root}")
    logger.info(f"Results root   : {args.results_root}")

    # Validate essential input presence
    if not (args.raw_root / "train.csv").exists() and not (args.raw_root / "Metadata.xlsx").exists():
        logger.warning("Neither train.csv nor Metadata.xlsx found in raw_root.")

    steps_to_run = [args.step] if args.step else list(range(args.start, args.end + 1))

    for s in steps_to_run:
        success = run_step(s, args.raw_root, args.results_root, dry_run=args.dry_run, extra_args=unknown)
        if not success:
            logger.error(f"Pipeline stopped at step {s} due to failure.")
            sys.exit(1)

    logger.info("✓ Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
