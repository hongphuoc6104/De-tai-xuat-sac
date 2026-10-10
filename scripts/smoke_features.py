"""Run genuine pretrained extraction on a strictly bounded six-PNG fixture."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from histology_data.feature_encoder import ResNet50Encoder
from histology_data.feature_sources import build_source_plan
from histology_data.features import FeatureRunConfig, run_feature_extraction, verify_feature_release
from histology_data.io import atomic_json


class _CountingEncoder:
    def __init__(self, delegate: ResNet50Encoder) -> None:
        self.delegate = delegate
        self.feature_dim = delegate.feature_dim
        self.calls = 0

    @property
    def descriptor(self) -> dict[str, Any]:
        return self.delegate.descriptor

    def encode(self, images: list[Any]) -> Any:
        self.calls += 1
        return self.delegate.encode(images)

    def release_memory(self) -> None:
        self.delegate.release_memory()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-kind", required=True, choices=("zip", "directory"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--offline-weights", action="store_true")
    args = parser.parse_args()
    plan = build_source_plan(args.metadata, args.source_root, args.source_kind,
                             expected_archives=1, expected_pngs=6)
    if plan["observed_pngs"] != 6 or plan["source_errors"]:
        parser.exit(2, "This local smoke accepts exactly six matching PNGs. Prepare a tiny fixture first.\n")
    if plan["coverage"] != {"4": 2, "10": 2, "40": 2}:
        parser.exit(2, "Smoke requires exactly two PNGs per objective lens.\n")
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    try:
        encoder = _CountingEncoder(ResNet50Encoder(args.weights_dir, device="cpu", download=not args.offline_weights))
        config = FeatureRunConfig(args.metadata, args.source_root, args.source_kind,
                                  args.output, args.work_root, args.weights_dir, batch_size=2,
                                  device="cpu", expected_archives=1, expected_pngs=6)
        first = run_feature_extraction(config, encoder=encoder)
        assert first["status"] == "complete" and first["committed_vectors"] == 6
        calls = encoder.calls
        resumed = run_feature_extraction(config, encoder=encoder)
        assert resumed["reused_parts"] == first["completed_parts"] and resumed["new_parts"] == 0
        assert encoder.calls == calls, "Resume reran the encoder on committed patches."
        audit = verify_feature_release(args.output, first["feature_id"])
        assert audit["verified_vectors"] == 6 and audit["training_ready"] is False
        report = {"scope": "six_patch_cpu_smoke_only", "source_kind": args.source_kind,
                  "coverage": plan["coverage"], "weights": encoder.descriptor,
                  "first_run": first, "resume": resumed, "audit": audit,
                  "encoder_forward_calls_first_run": calls, "encoder_forward_calls_resume": encoder.calls - calls,
                  "T4_benchmarked": False, "full_dataset_tested": False,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        atomic_json(args.output / "real_encoder_smoke.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        torch.set_num_threads(previous_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
