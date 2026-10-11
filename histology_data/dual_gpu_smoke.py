"""T4 parity, full-bag checkpoint and repeat-process smoke gates before B1."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dual_common import ArtifactStore, check_deadline
from .dual_member import prepare_member_states
from .dual_mil import make_novelty_model_factory, novelty_model_identity
from .mil_checkpoint import CheckpointStore, FitLock
from .mil_model import ModelConfig
from .mil_training import TrainConfig, run_fit


def run_gpu_smoke(
    dataset: Any,
    factory: Any,
    splits: dict,
    photo_map: dict,
    shared: ArtifactStore,
    store: ArtifactStore,
    output: Path,
    deadline: float,
) -> dict[str, Any]:
    """Use original inner-train/validation only; never reuse these head weights."""
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("GPU smoke requires the actual authorized T4.")
    outer = splits["outer"][0]
    inner = outer["inner_folds"][0]
    scope = f"outer_{outer['fold']}/inner_{inner['fold']}"
    content = shared.read(f"scopes/{scope}/transform.pkl")
    if content is None:
        raise ValueError("GPU smoke requires committed train-only transforms.")
    transforms = pickle.loads(content)
    train_ids = inner["train_cases"]
    anchor = sorted(c for c in train_ids if dataset.records[c]["target"] == 0)[0]
    member = prepare_member_states(dataset, factory, transforms, photo_map, train_ids, anchor, store, scope, deadline)
    states = member["lens_states"]
    head_ids = member["head_train_ids"]
    core_config = ModelConfig(variant="E4")
    model_factory = make_novelty_model_factory(states, beta=0.0, member_id=member["member_id"])
    model_identity = novelty_model_identity(core_config, states, beta=0.0, member_id=member["member_id"])
    config = TrainConfig(
        model=core_config, mode="inner", max_epochs=2, max_steps=None, device="cuda", seed=42, num_threads=1
    )
    lease = Path("/tmp/histology-dual-gpu-0")
    lease.mkdir(parents=True, exist_ok=True)
    with FitLock(lease, recover_stale=store.recover_stale):
        train = factory(head_ids)
        validation = factory(inner["validation_cases"])
        largest = max(
            head_ids, key=lambda case: sum(len(photo_map["case_tile_order"][case][lens]) for lens in ["4", "10", "40"])
        )
        case = train.load_case(largest)
        features = {
            lens: torch.tensor(np.asarray(x), dtype=torch.float32, device="cuda") for lens, x in case.features.items()
        }
        torch.manual_seed(42)
        model = model_factory(config).to("cuda").eval()
        with torch.no_grad():
            left = model(features, case.presence_mask)["logit"]
            right = model.core(features, case.presence_mask)["logit"]
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
        del model, features
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        check_deadline(deadline)
        started = time.monotonic()
        fit = Path(output) / "fit"
        result = run_fit(
            train,
            validation,
            fit,
            config,
            fit_identity={"mode": "dual_T4_smoke", "scope": scope, "bank_member_id": member["member_id"]},
            deadline=deadline,
            model_factory=model_factory,
            model_identity=model_identity,
            recover_stale_lock=store.recover_stale,
            lock_recovery_evidence=store.recovery_evidence.get(str(fit.relative_to(store.root))),
        )
        if result["phase"] != "complete":
            return {"phase": "paused_budget", "stage": "T4_smoke", "fit": result}
        identity = __import__("json").loads((fit / "resolved_config.json").read_text())
        checkpoint = CheckpointStore(fit / "checkpoints", identity).load_latest(map_location="cpu")
        if checkpoint is None or not checkpoint["finished"]:
            raise ValueError("Smoke checkpoint is not committed complete.")
        report = {
            "phase": "complete",
            "stage": "T4_smoke",
            "clinical_metrics": False,
            "outer_test_used": False,
            "parity_beta0": True,
            "full_bags": True,
            "checkpoint_verified": True,
            "epochs": checkpoint["epoch"],
            "duration_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "member_id": member["member_id"],
            "shared_identity_id": shared.identity_id,
            "bank_model_identity": model_identity,
            "weights_reusable_for_study": False,
        }
        existing = store.read("gpu_smoke/receipt.json")
        if existing is not None:
            import json

            prior = json.loads(existing)
            if prior["bank_model_identity"] != model_identity:
                raise ValueError("GPU smoke member identity changed.")
            report = {**prior, "fresh_process_resume_verified": True}
            store.json("gpu_smoke/new_process_resume.json", report)
        else:
            store.json("gpu_smoke/receipt.json", report)
        return report
