"""Bounded case-level MIL fits with verified epoch transactions and deterministic resume."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import random
import resource
import time
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from .io import atomic_json, file_hash, fingerprint
from .mil_checkpoint import CheckpointError, CheckpointIdentityError, CheckpointStore, FitLock
from .mil_model import MILModel, ModelConfig


class CaseDataset(Protocol):
    """One-case-at-a-time source, with immutable identity and supervised case records."""

    case_ids: list[str]
    identity: dict[str, Any]
    records: dict[str, dict[str, Any]]

    def load_case(self, case_id: str) -> Any:
        """Load a verified case; never broadcast its target to its patches."""


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Fixed per-fit protocol; a larger epoch budget requires a distinct fit identity."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    mode: str = "smoke"
    max_epochs: int = 2
    patience: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    seed: int = 42
    device: str = "cpu"
    max_steps: int | None = 20
    memory_limit_bytes: int = 2 * 1024**3
    num_threads: int = 2
    deterministic: bool = True

    def validate(self) -> None:
        """Reject invalid or unbounded smoke configurations before starting a writer."""
        if self.mode not in {"smoke", "inner", "refit"}:
            raise ValueError("mode must be smoke, inner or refit.")
        for name in ("max_epochs", "patience", "num_threads", "memory_limit_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda; GPU runs never silently fall back.")
        if self.mode == "smoke" and (self.max_steps is None or self.max_steps > 20):
            raise ValueError("Smoke training requires a step cap of at most 20.")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1
        ):
            raise ValueError("max_steps must be positive or None.")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"Invalid {name}.")


def environment() -> dict[str, Any]:
    """Record reproducibility context without credentials, personal paths or raw metadata."""
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "system": platform.system(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def source_fingerprint() -> str:
    """Bind a fit to the actual trainer/model/loader/checkpoint implementation bytes."""
    root = Path(__file__).parent
    names = ("mil_model.py", "mil_data.py", "mil_checkpoint.py", "mil_training.py", "mil_study.py", "mil_outputs.py",
             "mil_case_cache.py",
             "bags.py", "readiness.py", "splits.py", "features.py", "io.py")
    return fingerprint({name: file_hash(root / name) for name in names})


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed the stochastic sources used by the trainer and dropout."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def capture_rng() -> dict[str, Any]:
    """Snapshot stochastic state at the same boundary as model and optimizer."""
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Restore RNG after constructing/loading a model, before the next shuffled epoch."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint requires a CUDA environment.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _supervision(dataset: CaseDataset) -> dict[str, dict[str, Any]]:
    ids = dataset.case_ids
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Dataset cases must be nonempty and unique.")
    if set(ids) != set(dataset.records):
        raise ValueError("Supervised case records differ from dataset membership.")
    for record in dataset.records.values():
        if not record.get("patient_id") or type(record.get("target")) is not int or record["target"] not in (0, 1):
            raise ValueError("Each case requires a patient and a reviewed binary integer target.")
    return dataset.records


def verify_partition_isolation(
    train: CaseDataset, validation: CaseDataset | None, *, require_validation_classes: bool = True,
) -> None:
    """Validate supervised partitions independently of feature tensors and ZIP boundaries."""
    train_rows = _supervision(train)
    if {row["target"] for row in train_rows.values()} != {0, 1}:
        raise ValueError("Training partition must contain both case classes.")
    if validation is not None:
        val_rows = _supervision(validation)
        if require_validation_classes and {row["target"] for row in val_rows.values()} != {0, 1}:
            raise ValueError("Validation partition must contain both case classes.")
        if set(train.case_ids) & set(validation.case_ids):
            raise ValueError("Case leakage between train and validation.")
        if {r["patient_id"] for r in train_rows.values()} & {r["patient_id"] for r in val_rows.values()}:
            raise ValueError("Patient leakage between train and validation.")


def _case_tensors(case: Any, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if type(case.target) is not int or case.target not in (0, 1):
        raise ValueError("Only a binary case target is supported.")
    # Source arrays are read-only/provenance-bound; a writable, per-case copy avoids
    # PyTorch's undefined behavior warning without ever mutating the feature cache.
    features = {str(lens): torch.from_numpy(np.array(value, copy=True)).to(device)
                for lens, value in case.features.items()}
    target = torch.tensor(float(case.target), dtype=torch.float32, device=device)
    return features, target


def _check_resources(config: TrainConfig) -> None:
    # Linux ru_maxrss is KiB. A conservative peak cap also bounds any previous allocation in this process.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if platform.system() != "Darwin" else 1)
    if peak > config.memory_limit_bytes:
        raise MemoryError(f"Process peak RSS {peak} exceeded memory_limit_bytes={config.memory_limit_bytes}.")


def evaluate(
    model: MILModel, dataset: CaseDataset, device: str, *, config: TrainConfig | None = None,
    deadline: float | None = None,
    case_callback: Callable[[Any, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Forward each case without gradients, returning case-level loss and predictions."""
    model.eval()
    predictions = []
    loss_sum = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for case_id in dataset.case_ids:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Evaluation deadline reached; no partial predictions are committed.")
            if config is not None:
                _check_resources(config)
            case = dataset.load_case(case_id)
            features, target = _case_tensors(case, torch.device(device))
            output = model(features, presence_mask=case.presence_mask)
            logit = output["logit"].reshape(())
            loss = criterion(logit, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite evaluation loss.")
            loss_sum += float(loss)
            predictions.append({
                "case_id": case.case_id, "patient_id": case.patient_id, "target": case.target,
                "probability": float(torch.sigmoid(logit)), "logit": float(logit),
            })
            if case_callback is not None:
                case_callback(case, output)
    return {"loss": loss_sum / len(predictions), "predictions": predictions}


def run_fit(
    train: CaseDataset, validation: CaseDataset | None, output: Path, config: TrainConfig,
    *, fit_identity: dict[str, Any], deadline: float, recover_stale_lock: bool = False,
    lock_recovery_evidence: dict[str, Any] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    event_hook: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run/resume one fit; only a fully committed epoch advances durable progress.

    ``deadline`` is monotonic and mandatory. Hook injection is for isolated tests;
    checkpoint payloads are trusted only after store identity/checksum verification.
    """
    config.validate()
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise ValueError("A future finite monotonic deadline is required.")
    if config.mode == "smoke" and deadline - time.monotonic() > 601:
        raise ValueError("Local/smoke runs require a deadline of at most ten minutes.")
    if config.mode != "refit" and validation is None:
        raise ValueError("Inner/smoke fits require a separate validation partition.")
    if config.mode == "refit" and validation is not None:
        raise ValueError("Outer refit cannot early-stop against an external validation/test set.")
    if config.mode != "smoke":
        for dataset in (train, validation):
            if dataset is not None and dataset.identity.get("mode") == "smoke":
                raise ValueError("Smoke fixtures cannot be used for production training.")
    verify_partition_isolation(train, validation, require_validation_classes=config.mode != "smoke")
    if config.device == "cuda" and config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; refusing silent CPU fallback.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.num_threads)
    seed_everything(config.seed, config.deterministic)
    identity = {
        "fit": fit_identity, "config": dataclasses.asdict(config), "train": train.identity,
        "validation": validation.identity if validation is not None else None,
        "train_cases": train.case_ids, "validation_cases": validation.case_ids if validation else [],
        "environment": environment(), "source_fingerprint": source_fingerprint(),
    }
    identity["identity_id"] = fingerprint(identity)
    existing_config = output / "resolved_config.json"
    if existing_config.exists() and json.loads(existing_config.read_text()) != identity:
        raise CheckpointIdentityError("Fit output belongs to another config/data/fold/environment; preserving it.")
    store = CheckpointStore(output / "checkpoints", identity)
    state: dict[str, Any] = {}

    def report(phase: str, **values: Any) -> dict[str, Any]:
        result = {
            "phase": phase, "mode": config.mode, "identity_id": identity["identity_id"],
            "epoch": int(state.get("epoch", 0)), "global_step": int(state.get("global_step", 0)),
            "next_epoch": int(state.get("epoch", 0)) + 1, **values,
        }
        atomic_json(output / "status.json", result)
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), **result}, allow_nan=False) + "\n")
        return result

    with FitLock(output, recover_stale=recover_stale_lock, recovery_evidence=lock_recovery_evidence):
        # Recheck while owning the writer lease: another fit may have committed
        # after the read-only preflight and before this lock was acquired.
        if existing_config.exists() and json.loads(existing_config.read_text()) != identity:
            raise CheckpointIdentityError("Fit identity changed before acquiring its writer lease.")
        atomic_json(output / "resolved_config.json", identity)
        device = torch.device(config.device)
        model = MILModel(config.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        loaded = store.load_latest(map_location="cpu")
        if loaded is not None:
            state = loaded
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(device)
            restore_rng(state["rng"])
        else:
            state = {
                "epoch": 0, "global_step": 0, "best_epoch": None, "best_loss": None,
                "patience_count": 0, "history": [], "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": capture_rng(), "finished": False,
            }
            store.publish(state, epoch=0)
        if state["finished"]:
            return report("complete", resumed=True, best_epoch=state["best_epoch"], history=state["history"])
        report("running", resumed=loaded is not None)
        try:
            while state["epoch"] < config.max_epochs:
                estimate = state["history"][-1]["duration_seconds"] if state["history"] else 0.0
                if time.monotonic() + estimate + 1 >= deadline or (stop_requested and stop_requested()):
                    return report("paused_budget", history=state["history"], best_epoch=state["best_epoch"])
                if config.max_steps is not None and state["global_step"] + len(train.case_ids) > config.max_steps:
                    return report("paused_steps", history=state["history"], best_epoch=state["best_epoch"])
                started = time.monotonic()
                model.train()
                order = random.sample(train.case_ids, len(train.case_ids))
                epoch_losses = []
                for case_id in order:
                    if time.monotonic() >= deadline:
                        # The mutated in-memory epoch is discarded; resume reloads the previous committed snapshot.
                        return report("paused_budget", interrupted_epoch=True)
                    _check_resources(config)
                    case = train.load_case(case_id)
                    features, target = _case_tensors(case, device)
                    optimizer.zero_grad(set_to_none=True)
                    logit = model(features, presence_mask=case.presence_mask)["logit"].reshape(())
                    loss = criterion(logit, target)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss; no optimizer step or commit allowed.")
                    loss.backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise ValueError("Non-finite gradient; refusing optimizer step.")
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise ValueError("Non-finite model weights; refusing checkpoint.")
                    epoch_losses.append(float(loss.detach()))
                    if event_hook:
                        event_hook("after_step", {"epoch": state["epoch"] + 1, "case_id": case_id})
                validation_result = evaluate(model, validation, config.device, config=config,
                                             deadline=deadline) if validation else None
                val_loss = validation_result["loss"] if validation_result else None
                best = val_loss is not None and (state["best_loss"] is None or val_loss < state["best_loss"])
                completed_epoch = state["epoch"] + 1
                history_row = {
                    "epoch": completed_epoch, "train_loss": float(np.mean(epoch_losses)),
                    "validation_loss": val_loss, "duration_seconds": time.monotonic() - started,
                    "case_order": order,
                }
                next_state = {
                    "epoch": completed_epoch, "global_step": state["global_step"] + len(order),
                    "best_epoch": completed_epoch if best else state["best_epoch"],
                    "best_loss": val_loss if best else state["best_loss"],
                    "patience_count": 0 if best else state["patience_count"] + 1,
                    "history": [*state["history"], history_row],
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng(),
                }
                early_stop = validation is not None and next_state["patience_count"] >= config.patience
                next_state["finished"] = early_stop or completed_epoch == config.max_epochs
                if event_hook:
                    event_hook("before_checkpoint", {"epoch": completed_epoch})
                report("checkpointing", pending_epoch=completed_epoch)
                store.publish(next_state, epoch=completed_epoch, is_best=best)
                state = next_state
                report("running", history=state["history"], best_epoch=state["best_epoch"])
                if state["finished"]:
                    break
            return report("complete", history=state["history"], best_epoch=state["best_epoch"])
        except Exception as exc:
            # A commit can succeed before a pointer write fails. Reconcile durable
            # progress read-only, preserving the original I/O error for the operator.
            try:
                committed = store.load_latest(map_location="cpu", reconcile_pointers=False)
            except CheckpointError as integrity_error:
                report("error", error_type=type(exc).__name__, error=str(exc),
                       reconciliation_error=str(integrity_error), durable_progress_verified=False)
                raise exc from integrity_error
            if committed is not None:
                state = committed
            if isinstance(exc, TimeoutError) and time.monotonic() >= deadline:
                return report("paused_budget", interrupted_epoch=True, history=state["history"],
                              best_epoch=state["best_epoch"])
            report("error", error_type=type(exc).__name__, error=str(exc), last_committed_epoch=state["epoch"])
            raise
