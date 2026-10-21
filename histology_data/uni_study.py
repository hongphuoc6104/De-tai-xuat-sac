"""One frozen-UNI image classifier using the archived patient assignments.

The bounded fit dispatcher and selection implementation are shared with the
verified head studies. This is a new scientific protocol, never an R2 result.
"""
from __future__ import annotations

import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from . import image_classification as core, image_classification_r0_r3 as heads
from .image_classification_improvement import validate_source_lineage
from .io import atomic_json, file_hash, fingerprint
from .uni_cache import load_cache
from .uni_views import IMAGE_DIM, PROTOCOL, ViewEncoder, embed_image


def source_from_cache(cache_root: Path, baseline: heads.Source) -> heads.Source:
    """Bind new vectors to unchanged original image labels and baseline folds."""
    features, rows, descriptor = load_cache(cache_root)
    lineage = validate_source_lineage(baseline.rows, rows)
    by_id = {row["image_id"]: i for i, row in enumerate(rows)}
    order = [by_id[row["image_id"]] for row in baseline.rows]
    receipt = {"protocol": PROTOCOL, "baseline": baseline.receipt, "lineage_sha256": lineage,
               "uni_cache_id": cache_root.name, "cache_commit_sha256": file_hash(cache_root / "commit.json"),
               "encoder": descriptor["encoder"], "feature_shape": [len(rows), IMAGE_DIM]}
    source = heads.Source(features[order], [rows[i] for i in order], baseline.baseline,
                          baseline.oof, receipt, cache_root, baseline.baseline_root)
    heads.preflight(source)
    return source


def _verify_report(root: Path, descriptor: dict[str, Any]) -> dict[str, Any] | None:
    if not heads._verify(root / "report", {"descriptor": descriptor}):
        return None
    heads._verify_unit_manifest(root)
    return json.loads((root / "report/comparison.json").read_text())


def run_study(source: heads.Source, output: Path, config: heads.Config = heads.Config(), *,
              max_new_fits: int | None = None, refit: bool = False) -> dict[str, Any]:
    """Fit only UNI/global L2 LR; optional all-data refit is a research artifact.

    No backbone weights change. Normalizer on concatenation follows the fixed
    per-block normalization, then a training-only scaler. R0 is read-only.
    """
    config = replace(config, lbfgs_tol=config.normalized_lbfgs_tol)
    config.validate()
    if source.receipt.get("protocol") != PROTOCOL or source.features.shape[1] != IMAGE_DIM:
        raise ValueError("UNI head requires a verified UNI3072 feature source.")
    if max_new_fits is not None and (type(max_new_fits) is not int or max_new_fits < 0):
        raise ValueError("max_new_fits must be a nonnegative integer.")
    if any(root is not None and Path(output).resolve().is_relative_to(root.resolve())
           for root in (source.release_root, source.baseline_root)):
        raise ValueError("Study output cannot modify immutable source artifacts.")
    heads.preflight(source)
    descriptor = {"protocol": PROTOCOL, "source": source.receipt,
                  "settings": {k: v for k, v in config.scientific().items() if k not in {"ratios", "condition_saga"}},
                  "implementation_sha256": file_hash(Path(__file__)),
                  "worker_sha256": file_hash(Path(heads.__file__)),
                  "core_sha256": file_hash(Path(core.__file__)),
                  "versions": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn")},
                  "assignments_sha256": fingerprint([scope[3] for scope in heads.scopes(source)])}
    descriptor = json.loads(json.dumps(descriptor))
    root = Path(output) / fingerprint(descriptor)
    root.mkdir(parents=True, exist_ok=True)
    runner = heads.Runner(source, root, config, max_new_fits)
    status: dict[str, Any] = {"state": "running", "study_root": str(root), "protocol": PROTOCOL,
                              "real_encoder_checked": source.receipt["encoder"].get("kind") == "UNI"}
    with core._RunLock(root, recover=False):
        atomic_json(root / "config.json", descriptor)
        previous = _verify_report(root, descriptor)
        if previous is not None and (not refit or "deployment" in previous["results"]):
            return {**status, "state": "complete", "new_fits": 0, "report": previous}
        try:
            results = {}
            for name, train, test, inner in heads.scopes(source):
                if name == "deployment" and not refit:
                    continue
                results[name] = {"all": heads._run_head(runner, "UNI", name, train, test, inner, "all")}
                atomic_json(root / "status.json", {**status, "scope": name, "new_fits": runner.new_fits})
            records = heads._prediction_rows(source, "UNI", results)
            old, new = heads.report_metrics(source.oof), heads.report_metrics(records)
            gate = (new["all"]["AUROC"] - old["all"]["AUROC"] >= .02
                    and new["all"]["AP"] >= old["all"]["AP"]
                    and all(new[str(lens)]["AUROC"] - old[str(lens)]["AUROC"] >= -.02 for lens in (4, 10, 40)))
            report = {"interpretation": "Exploratory patient-disjoint internal evaluation; no independent test.",
                      "metrics": {"R0": old, "UNI": new}, "priority_passed": gate,
                      "selected": "UNI" if gate else "R0", "results": results,
                      "external_test_completed": False, "deployment_refit_is_validation": False,
                      "paired": heads.paired_intervals(source.oof, records, config)}
            folder = root / "report"
            folder.mkdir(exist_ok=True)
            atomic_json(folder / "comparison.json", report)
            atomic_json(folder / "oof.json", records)
            atomic_json(folder / "unit_manifest.json", {str(p.relative_to(root)): file_hash(p)
                                                        for p in (root / "arms").rglob("commit.json")})
            heads._publish(folder, {"descriptor": descriptor}, ["comparison.json", "oof.json", "unit_manifest.json"])
            status.update(state="complete", new_fits=runner.new_fits)
        except TimeoutError as error:
            status.update(state="partial", new_fits=runner.new_fits, reason=str(error))
            report = None
        except Exception:
            atomic_json(root / "status.json", {**status, "state": "attention_required"})
            raise
        atomic_json(root / "status.json", status)
    return {**status, "report": report}


def predict_vector(root: Path, vector: np.ndarray, *, encoder_descriptor: dict[str, Any]) -> dict[str, Any]:
    """Verify a trusted local deployment graph before deserializing its model."""
    import joblib

    descriptor = json.loads((root / "config.json").read_text())
    if descriptor.get("protocol") != PROTOCOL or fingerprint(descriptor) != root.name:
        raise ValueError("UNI study identity mismatch.")
    report = _verify_report(root, descriptor)
    if report is None or "deployment" not in report["results"]:
        raise ValueError("Complete UNI deployment refit required for single-image inference.")
    if encoder_descriptor != descriptor["source"]["encoder"]:
        raise ValueError("Inference encoder identity differs from training cache.")
    selected = report["results"]["deployment"]["all"]
    model_path = root / selected["model_root"] / "model.joblib"
    if not model_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Deployment model path escapes study.")
    x = np.asarray(vector, dtype=np.float32)
    if x.shape != (IMAGE_DIM,) or not np.isfinite(x).all():
        raise ValueError("Expected one finite UNI3072 vector.")
    model = joblib.load(model_path)
    if list(model.named_steps) != ["normalizer", "scaler", "classifier"] or not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Invalid UNI head pipeline.")
    score = float(model.predict_proba(x[None, :])[0, 1])
    threshold = selected["selected"]["threshold"]
    return {"score": score, "threshold": threshold, "image_target": int(score >= threshold),
            "prediction_unit": "original_image", "priority_passed": report["priority_passed"],
            "clinical_probability_calibrated": False}


def predict_image(root: Path, image: Any, encoder: ViewEncoder, *, batch_size: int = 4) -> dict[str, Any]:
    """Use exactly the extraction geometry; no labels are accepted for inference."""
    descriptor = json.loads((root / "config.json").read_text())
    if descriptor["source"]["encoder"] != encoder.descriptor():
        raise ValueError("Inference encoder identity differs from training cache.")
    vector, _, _ = embed_image(image, encoder, batch_size=batch_size)
    return predict_vector(root, vector, encoder_descriptor=encoder.descriptor())
