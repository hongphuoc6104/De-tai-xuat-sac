#!/usr/bin/env python3
"""Run bounded MIL smoke, export fixtures, or resume a locked nested study."""
from __future__ import annotations

import argparse
import dataclasses
import json
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology_data.io import atomic_json, fingerprint  # noqa: E402
from histology_data.mil_data import FixtureDataset, create_toy_fixture, export_training_fixture  # noqa: E402
from histology_data.mil_model import ModelConfig  # noqa: E402
from histology_data.mil_study import StudyConfig, run_pilot, run_study  # noqa: E402
from histology_data.mil_training import TrainConfig, environment, run_fit  # noqa: E402


class Subset:
    """Keep fixture partitions explicit and preserve the parent data fingerprint."""

    def __init__(self, dataset: FixtureDataset, case_ids: list[str]) -> None:
        if not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids).issubset(dataset.case_ids):
            raise ValueError("Invalid fixture subset.")
        self.dataset = dataset
        self.case_ids = case_ids
        self.records = {case: dataset.records[case] for case in case_ids}
        self.identity = {**dataset.identity, "case_ids": case_ids,
                         "subset_fingerprint": fingerprint({"source": dataset.identity, "case_ids": case_ids})}

    def load_case(self, case_id: str) -> Any:
        if case_id not in self.records:
            raise KeyError("Case is outside this supervised fixture partition.")
        return self.dataset.load_case(case_id)


def _deadline(value: str | None, minutes: float) -> float:
    if value is None:
        return time.monotonic() + minutes * 60
    requested = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if requested.tzinfo is None:
        raise ValueError("Deadline must be timezone-aware UTC/ISO8601.")
    remaining = (requested - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise ValueError("Deadline already passed.")
    return time.monotonic() + remaining


def main(argv: list[str] | None = None) -> int:
    """Expose JSON-configured modes without hidden GPU allocation or auth operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["toy", "export", "smoke", "pilot", "production"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deadline-utc")
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text())
    output = Path(values["output"])
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _sig, _frame: stop.set())
    try:
        if args.mode == "toy":
            path = create_toy_fixture(output, seed=values.get("seed", 42),
                                      case_count=values.get("case_count", 4), rows_per_lens=values.get("rows", 16))
            result = {"phase": "fixture_complete", "path": str(path), "mode": "smoke"}
        elif args.mode == "export":
            path = export_training_fixture(
                Path(values["bundle"]), Path(values["feature_root"]), output,
                outer_fold=values.get("outer", 0), inner_fold=values.get("inner", 0),
                max_cases=values.get("max_cases", 4), max_rows=values.get("max_rows", 64),
            )
            result = {"phase": "fixture_complete", "path": str(path), "mode": "smoke"}
        elif args.mode == "smoke":
            dataset = FixtureDataset(Path(values["fixture"]))
            default_train = [next(c for c in dataset.case_ids if dataset.records[c]["target"] == y) for y in (0, 1)]
            train_ids = values.get("train_cases", default_train)
            validation_ids = values.get("validation_cases", [c for c in dataset.case_ids if c not in train_ids])
            config = TrainConfig(
                model=ModelConfig(variant=values.get("variant", "E4"), dropout=values.get("dropout", .25)),
                max_epochs=values.get("epochs", 2), max_steps=20, seed=values.get("seed", 42),
                device=values.get("device", "cpu"), learning_rate=values.get("learning_rate", 1e-4),
                memory_limit_bytes=values.get("memory_limit_bytes", 2 * 1024**3),
            )
            result = run_fit(
                Subset(dataset, train_ids), Subset(dataset, validation_ids), output, config,
                fit_identity={"mode": "smoke", "fixture_id": dataset.identity_hash},
                deadline=min(_deadline(args.deadline_utc, 10), time.monotonic() + 600),
                recover_stale_lock=values.get("recover_stale_lock", False), stop_requested=stop.is_set,
                lock_recovery_evidence=values.get("lock_recovery_evidence"),
            )
            atomic_json(output / "environment.json", environment())
            atomic_json(output / "smoke_report.json", {**result, "config": dataclasses.asdict(config),
                                                       "clinical_metrics": False})
        elif args.mode == "pilot":
            if args.deadline_utc is None:
                raise ValueError("Pilot requires an explicit bounded --deadline-utc.")
            result = run_pilot(
                Path(values["bundle"]), Path(values["feature_root"]), output,
                deadline=_deadline(args.deadline_utc, 30), variant=values.get("variant", "E4"),
                epochs=values.get("epochs", 2), recover_stale_lock=values.get("recover_stale_lock", False),
                stop_requested=stop.is_set,
            )
        else:
            if args.deadline_utc is None:
                raise ValueError("Production requires an explicit bounded --deadline-utc.")
            config = StudyConfig(
                variants=tuple(values.get("variants", ["E0", "E1", "E2", "E3", "E4"])),
                max_epochs=values.get("epochs", 30), patience=values.get("patience", 5),
                device="cuda", memory_limit_bytes=values.get("memory_limit_bytes", 8 * 1024**3),
            )
            result = run_study(
                Path(values["bundle"]), Path(values["feature_root"]), output, config,
                deadline=_deadline(args.deadline_utc, 300), recover_stale_lock=values.get("recover_stale_lock", False),
                recovery_evidence=values.get("recovery_evidence"), stop_requested=stop.is_set,
            )
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["phase"] in {"complete", "fixture_complete"} else 2
    except Exception as exc:
        print(json.dumps({"phase": "error", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
