"""Bounded cached-head lab and source-verified original-image error review."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from . import image_classification as core, image_classification_studies as studies
from .io import atomic_json, file_hash, fingerprint

VARIANTS = {
    "patient_lr": {"weight_scheme": "patient_class"},
    "class_lr": {"weight_scheme": "class"},
    "uniform_lr": {"weight_scheme": "none"},
    "patient_pca32": {"weight_scheme": "patient_class", "pca_components": 32},
    "patient_pca64": {"weight_scheme": "patient_class", "pca_components": 64},
    "per_lens_lr": {"weight_scheme": "patient_class", "head_mode": "per_lens"},
    "per_lens_pca32": {"weight_scheme": "patient_class", "head_mode": "per_lens", "pca_components": 32},
}


def _parse_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        for key in ("image_target", "objective_lens", "outer_fold", "prediction_at_0_5", "prediction_at_selected_threshold"):
            record[key] = int(record[key])
        for key in ("predicted_probability", "selected_threshold"):
            record[key] = float(record[key])
    return records


def _validate_baseline_rows(data: studies.StudyData, rows: list[dict[str, Any]]) -> None:
    expected = {r["image_id"]: r for r in data.rows}
    assignments = {data.rows[i]["image_id"]: number
                   for number, (_, test) in enumerate(data.outer, start=1) for i in test}
    if len(rows) != len(expected) or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Baseline CSV is not one prediction per source image.")
    for row in rows:
        original = expected.get(row["image_id"])
        if original is None or any(row[k] != original[k] for k in ("patient_id", "objective_lens", "image_target")):
            raise ValueError("Baseline prediction lineage differs from verified feature rows.")
        if row["outer_fold"] != assignments[row["image_id"]]:
            raise ValueError("Baseline CSV/fixed assignments differ.")


def diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate image-level errors; never publish source or patient identifiers."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    groups = {"all": rows, **{str(k): [r for r in rows if r["objective_lens"] == k] for k in (4, 10, 40)}}
    output = {}
    for name, subset in groups.items():
        y = np.array([r["image_target"] for r in subset])
        p = np.array([r["predicted_probability"] for r in subset])
        prediction = np.array([r["prediction_at_selected_threshold"] for r in subset])
        counts = Counter(r["patient_id"] for r in subset)
        weights = np.array([1.0 / counts[r["patient_id"]] for r in subset])
        output[name] = {"images": len(subset), "patients": len(counts),
                        "AUROC": float(roc_auc_score(y, p)), "AP": float(average_precision_score(y, p)),
                        "patient_balanced_image_AUROC": float(roc_auc_score(y, p, sample_weight=weights)),
                        "FN": int(np.sum((y == 1) & (prediction == 0))),
                        "FP": int(np.sum((y == 0) & (prediction == 1)))}
    return output


def paired_deltas(first: list[dict[str, Any]], second: list[dict[str, Any]], *,
                  repeats: int = 500, seed: int = 17) -> dict[str, Any]:
    """Conditional paired patient bootstrap; uses existing OOF, never refits."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    by_id = {r["image_id"]: r for r in first}
    if len(by_id) != len(first) or len(second) != len(first) or {r["image_id"] for r in second} != set(by_id):
        raise ValueError("Paired comparison requires identical unique source images.")
    aligned = [by_id[r["image_id"]] for r in second]
    keys = ("patient_id", "image_target", "objective_lens", "outer_fold")
    if any(any(a[k] != b[k] for k in keys) for a, b in zip(aligned, second, strict=True)):
        raise ValueError("Paired comparison cohorts/outer assignments differ.")
    y = np.array([r["image_target"] for r in aligned])
    p = np.array([r["predicted_probability"] for r in aligned])
    q = np.array([r["predicted_probability"] for r in second])
    patients = sorted({r["patient_id"] for r in aligned})
    groups = {patient: np.array([i for i, r in enumerate(aligned) if r["patient_id"] == patient]) for patient in patients}
    rng = np.random.default_rng(seed)
    auc, ap = [], []
    for _ in range(repeats):
        indices = np.concatenate([groups[k] for k in rng.choice(patients, len(patients), replace=True)])
        if set(y[indices].tolist()) != {0, 1}:
            continue
        auc.append(float(roc_auc_score(y[indices], q[indices]) - roc_auc_score(y[indices], p[indices])))
        ap.append(float(average_precision_score(y[indices], q[indices]) - average_precision_score(y[indices], p[indices])))
    return {"AUROC_delta": float(roc_auc_score(y, q) - roc_auc_score(y, p)),
            "AP_delta": float(average_precision_score(y, q) - average_precision_score(y, p)),
            "AUROC_delta_95_CI": np.quantile(auc, [0.025, 0.975]).tolist() if auc else None,
            "AP_delta_95_CI": np.quantile(ap, [0.025, 0.975]).tolist() if ap else None,
            "bootstrap_valid": len(auc), "bootstrap_requested": repeats,
            "interpretation": "Conditional on saved paired OOF; resample patients; no model refit."}


def _hash_budget(path: Path, deadline: float | None = None) -> str:
    """Cooperative deadline between bounded reads; cannot interrupt a blocked OS read."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if deadline is not None:
                core._check_deadline(deadline)
            block = stream.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _archive_signature(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "inode": info.st_ino, "device": info.st_dev}


def _archive_preflight(data: studies.StudyData, archives: Mapping[int, Path],
                       deadline: float | None = None) -> dict[int, dict[str, Any]]:
    """Validate every requested archive before any new model is trained."""
    if set(archives) != {4, 10, 40}:
        raise ValueError("Source review requires all three archives or none.")
    bindings = {}
    for lens, supplied in archives.items():
        path = Path(supplied)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Source archive must be a regular readable file.")
        before = _archive_signature(path)
        sha = _hash_budget(path, deadline)
        if before != _archive_signature(path) or sha != data.release["descriptor"]["source_archive_sha256"][str(lens)]:
            raise ValueError(f"Review {lens}X archive differs from the frozen feature source.")
        bindings[lens] = {**before, "sha256": sha}
    return bindings


def _verify_file_manifest(root: Path, commit_name: str) -> dict[str, Any]:
    commit = json.loads((root / commit_name).read_text())
    body = {k: v for k, v in commit.items() if k != "commit_id"}
    if commit.get("commit_id") != fingerprint(body) or commit.get("complete") is not True or not commit.get("files"):
        raise ValueError("Invalid or missing artifact receipt.")
    for name, sha in commit["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe artifact manifest path.")
        path = root / relative
        if path.is_symlink() or not path.is_file() or file_hash(path) != sha:
            raise ValueError(f"Artifact checksum failed: {name}")
    return commit


def review_originals(data: studies.StudyData, baseline_rows: list[dict[str, Any]], output: Path,
                     archives: Mapping[int, Path], *, per_category: int = 2,
                     deadline: float | None = None,
                     bindings: Mapping[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Publish a complete source review atomically; never follow old private links."""
    output = Path(output)
    if output.is_symlink() or (output / "private").is_symlink():
        raise ValueError("Refusing a symlinked source-review directory.")
    if output.exists():
        _verify_file_manifest(output, "review.commit.json")
        receipt = json.loads((output / "review_receipt.json").read_text())
        if receipt["source_release"] != data.release["release_id"] or receipt["baseline_study_id"] != data.baseline["study_id"]:
            raise ValueError("Existing review receipt belongs to another baseline.")
        return receipt
    verified = dict(bindings) if bindings is not None else _archive_preflight(data, archives, deadline)
    if set(verified) != {4, 10, 40} or set(archives) != {4, 10, 40}:
        raise ValueError("Source review requires all three validated archives.")
    for lens in (4, 10, 40):
        path = Path(archives[lens])
        signature = {k: v for k, v in verified[lens].items() if k != "sha256"}
        if signature != _archive_signature(path) or verified[lens]["sha256"] != data.release["descriptor"]["source_archive_sha256"][str(lens)]:
            raise ValueError("Source changed after review preflight.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-", dir=output.parent) as tmp:
        staged = Path(tmp)
        receipt = _render_original_review(data, baseline_rows, staged, archives,
                                          per_category=per_category, deadline=deadline)
        files = {p.relative_to(staged).as_posix(): _hash_budget(p, deadline)
                 for p in staged.rglob("*") if p.is_file()}
        commit = {"complete": True, "files": files, "source_release": data.release["release_id"]}
        commit["commit_id"] = fingerprint(commit)
        atomic_json(staged / "review.commit.json", commit)
        _verify_file_manifest(staged, "review.commit.json")
        os.replace(staged, output)
    return receipt


def verify_lab(root: Path) -> dict[str, Any]:
    """Verify stable diagnostics/review evidence and every linked study commit."""
    root = Path(root)
    commit = _verify_file_manifest(root, "lab.commit.json")
    manifest = json.loads((root / "manifest.json").read_text())
    comparison = json.loads((root / "comparison.json").read_text())
    if fingerprint(manifest) != commit["lab_id"] or set(commit["study_commits"]) != set(manifest["variants"]):
        raise ValueError("Lab identity/study coverage differs from manifest.")
    required = {"manifest.json", "comparison.json", "report.md", "baseline_diagnostics.json"}
    if manifest["source_review"]:
        required |= {"review/review_receipt.json", "review/review.commit.json", "review/private/references.json"}
        required |= {f"review/{lens}x-review.png" for lens in (4, 10, 40)}
        _verify_file_manifest(root / "review", "review.commit.json")
    if not required <= set(commit["files"]) or set(comparison) != set(commit["study_commits"]):
        raise ValueError("Lab evidence/variant coverage is incomplete.")
    for name, reference in commit["study_commits"].items():
        study_root = Path(comparison[name]["study_root"])
        study, _ = studies.load_study(study_root)
        if study["descriptor"] != manifest["variants"][name]:
            raise ValueError("Sub-study differs from preregistered lab configuration.")
        if file_hash(study_root / "study.commit.json") != reference["sha256"] or reference["study_id"] != comparison[name]["study_id"]:
            raise ValueError("Lab links a changed sub-study commit.")
    return commit


def _render_original_review(data: studies.StudyData, baseline_rows: list[dict[str, Any]], output: Path,
                     archives: Mapping[int, Path], *, per_category: int = 2, deadline: float | None = None) -> dict[str, Any]:
    """Review a bounded sample; bind ZIP/image hashes and retain exact TIFF bytes."""
    import zipfile

    from PIL import Image, ImageDraw, ImageOps

    if set(archives) != {4, 10, 40} or not 1 <= per_category <= 4:
        raise ValueError("Review requires three source ZIPs and 1..4 images per error/control category.")
    source_by_id = {r["image_id"]: r for r in data.rows}
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    aliases = {p: f"P{i:02d}" for i, p in enumerate(sorted({r["patient_id"] for r in baseline_rows}), start=1)}
    references, archive_receipts = [], {}
    categories = ("FN", "FP", "TP", "TN")
    for lens in (4, 10, 40):
        archive = Path(archives[lens])
        expected = data.release["descriptor"]["source_archive_sha256"][str(lens)]
        archive_receipts[str(lens)] = expected
        canvas = Image.new("RGB", (per_category * 410, len(categories) * 265), "white")
        draw = ImageDraw.Draw(canvas)
        with zipfile.ZipFile(archive) as zipped:
            for group_index, category in enumerate(categories):
                draw.text((5, group_index * 265 + 2), f"{lens}x {category}", fill="black")
                wanted = {"FN": (1, 0), "FP": (0, 1), "TP": (1, 1), "TN": (0, 0)}[category]
                candidates = [r for r in baseline_rows if r["objective_lens"] == lens
                              and (r["image_target"], r["prediction_at_selected_threshold"]) == wanted]
                # One confident example and one near-threshold example, prefer distinct patients.
                candidates.sort(key=lambda r: r["predicted_probability"], reverse=category in {"FP", "TP"})
                ordered = candidates[:1] + sorted(candidates[1:], key=lambda r: abs(r["predicted_probability"] - r["selected_threshold"]))
                chosen, used = [], set()
                for row in ordered:
                    if row["patient_id"] not in used:
                        chosen.append(row)
                        used.add(row["patient_id"])
                    if len(chosen) == per_category:
                        break
                for column, row in enumerate(chosen):
                    if deadline is not None:
                        core._check_deadline(deadline)
                    src = source_by_id[row["image_id"]]
                    info = zipped.getinfo(src["source_member"])
                    if info.file_size > 512 * 1024**2:
                        raise ValueError("Review source image exceeds its byte safety cap.")
                    payload = zipped.read(info)
                    if hashlib.sha256(payload).hexdigest() != src["original_image_sha256"]:
                        raise ValueError("Review image bytes differ from the extracted original.")
                    name = f"{lens}x-{category}-{column + 1:02d}"
                    original = private / (name + ".tif")
                    with original.open("wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    original.chmod(0o600)
                    if file_hash(original) != src["original_image_sha256"]:
                        raise ValueError("Published review original failed read-back.")
                    with Image.open(io.BytesIO(payload)) as opened:
                        if opened.width * opened.height > 178_000_000:
                            raise ValueError("Review source exceeds pixel safety cap.")
                        preview = ImageOps.exif_transpose(opened).convert("RGB")
                    preview.thumbnail((390, 215))
                    x, y = column * 410 + 5, group_index * 265 + 45
                    canvas.paste(preview, (x, y))
                    draw.text((x, y - 40), f"{lens}x {category} {aliases[row['patient_id']]} Glade={row['raw_glade']}", fill="black")
                    draw.text((x, y - 20), f"OOF p={row['predicted_probability']:.3f} t={row['selected_threshold']:.3f}", fill="black")
                    preview.close()
                    references.append({"review_id": name, "image_id": row["image_id"],
                                       "patient_alias": aliases[row["patient_id"]],
                                       "category": category, "objective_lens": lens,
                                       "source_sha256": src["original_image_sha256"],
                                       "review_file": original.relative_to(output).as_posix(), "raw_glade": row["raw_glade"]})
        canvas.save(output / f"{lens}x-review.png")
        canvas.close()
    atomic_json(private / "references.json", references)
    (private / "references.json").chmod(0o600)
    receipt = {"source_release": data.release["release_id"], "baseline_study_id": data.baseline["study_id"], "image_count": len(references),
               "archive_sha256": archive_receipts,
               "selection": "Confident and near-threshold OOF examples; distinct patients within lens/category; exploratory review.",
               "note": "Montages are previews, not lesion annotations. Inspect saved full originals for diagnostic detail."}
    atomic_json(output / "review_receipt.json", receipt)
    return receipt


def run_lab(run_root: Path, baseline_root: Path, output_root: Path, *,
            variants: tuple[str, ...] = tuple(VARIANTS), budget_minutes: float = 120,
            reserve_minutes: float = 15, pca_override: int | None = None,
            archives: Mapping[int, Path] | None = None,
            notify: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """One Colab job: verify once, train predeclared cache studies, compare OOF."""
    if (not variants or len(set(variants)) != len(variants) or any(k not in VARIANTS for k in variants)
            or not 0 <= reserve_minutes < budget_minutes):
        raise ValueError("Invalid lab variants or budget.")
    if archives is not None and set(archives) != {4, 10, 40}:
        raise ValueError("Source review requires all three archives or none.")
    deadline = time.monotonic() + 60 * (budget_minutes - reserve_minutes)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data = studies.load_study_data(run_root, baseline_root)
    baseline_rows = _parse_rows(Path(baseline_root) / "oof_predictions.csv")
    _validate_baseline_rows(data, baseline_rows)
    configs = {}
    for name in variants:
        options = dict(VARIANTS[name])
        if pca_override is not None and options.get("pca_components"):
            options["pca_components"] = pca_override
        config = studies.StudyConfig(**options)
        studies.preflight(data, config)
        configs[name] = config
    try:
        bindings = _archive_preflight(data, archives, deadline) if archives is not None else None
    except core.BudgetExhausted:
        atomic_json(output_root / "preflight_status.json", {"state": "partial", "reason": "source_preflight_budget"})
        return {"state": "partial", "output_root": str(output_root), "reason": "source_preflight_budget"}
    manifest = {"baseline_study_id": data.baseline["study_id"], "release_id": data.release["release_id"],
                "assignment_sha256": data.assignment_sha256,
                "lab_implementation_sha256": file_hash(Path(__file__)),
                "source_review": bool(archives), "review_samples_per_lens_category": 2,
                "variants": {k: studies._descriptor(data, v) for k, v in configs.items()}}
    identity = fingerprint(manifest)
    path = output_root / identity
    path.mkdir(parents=True, exist_ok=True)
    if (path / "manifest.json").exists() and json.loads((path / "manifest.json").read_text()) != manifest:
        raise ValueError("Lab output belongs to another manifest.")
    atomic_json(path / "manifest.json", manifest)
    summaries = {}

    def record(phase: str, **details: Any) -> None:
        value = {"phase": phase, "lab_id": identity, "at": core._utc(), **details}
        atomic_json(path / "status.json", value)
        with (path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        if notify:
            notify(value)

    if (path / "lab.commit.json").is_file():
        verify_lab(path)
        return {"state": "complete", "lab_root": str(path), "lab_id": identity,
                "image_count": len(data.rows), "studies": json.loads((path / "comparison.json").read_text())}
    with core._RunLock(path, recover=False):
        record("cache_verified", image_count=len(data.rows), variants=list(variants))
        atomic_json(path / "baseline_diagnostics.json", diagnostic_summary(baseline_rows))
        for name, config in configs.items():
            remaining = (deadline - time.monotonic()) / 60
            if remaining <= 0:
                record("partial", completed=list(summaries), reason="lab_budget")
                return {"state": "partial", "lab_root": str(path), "completed": list(summaries)}
            record("study_started", variant=name)
            kwargs = {**config.__dict__, "budget_minutes": remaining, "reserve_minutes": 0.0}
            result = studies.run_study(data, studies.StudyConfig(**kwargs), absolute_deadline=deadline)
            if result["state"] != "complete":
                record("partial", variant=name, study_id=result["study_id"])
                return {"state": "partial", "lab_root": str(path), "variant": name}
            root = Path(result["study_root"])
            rows = _parse_rows(root / "oof_predictions.csv")
            summaries[name] = {"study_id": result["study_id"], "study_root": str(root),
                               "metrics": result["metrics"], "diagnostics": diagnostic_summary(rows),
                               "paired_vs_baseline": paired_deltas(baseline_rows, rows)}
            atomic_json(path / "comparison.json", summaries)
            record("study_complete", variant=name, study_id=result["study_id"],
                   AUROC=result["metrics"]["pooled"]["validation_selected_threshold"]["roc_auc"])
        if archives:
            record("source_review_started")
            try:
                review_originals(data, baseline_rows, path / "review", archives, deadline=deadline, bindings=bindings)
            except core.BudgetExhausted:
                record("partial", completed=list(summaries), reason="review_budget")
                return {"state": "partial", "lab_root": str(path), "completed": list(summaries)}
        report = ["# Cached-head image studies", "", "Same verified images and baseline outer patient assignments.",
                  "Exploratory internal comparisons; model choices use inner validation.", "",
                  "| Variant | AUROC | AP | Sensitivity | Specificity | Paired AUROC delta 95% CI |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, summary in summaries.items():
            metrics = summary["metrics"]["pooled"]["validation_selected_threshold"]
            interval = summary["paired_vs_baseline"]["AUROC_delta_95_CI"]
            report.append(f"| {name} | {metrics['roc_auc']:.3f} | {metrics['auprc']:.3f} | {metrics['sensitivity']:.3f} | {metrics['specificity']:.3f} | {interval} |")
        (path / "report.md").write_text("\n".join(report) + "\n")
        files = {p.name: file_hash(p) for p in (path / "manifest.json", path / "comparison.json", path / "report.md", path / "baseline_diagnostics.json")}
        if archives:
            files.update({p.relative_to(path).as_posix(): file_hash(p) for p in (path / "review").rglob("*") if p.is_file()})
        commit = {"lab_id": identity, "complete": True, "files": files, "study_ids": {k: v["study_id"] for k, v in summaries.items()},
                  "study_commits": {k: {"study_id": v["study_id"], "sha256": file_hash(Path(v["study_root"]) / "study.commit.json")} for k, v in summaries.items()}}
        commit["commit_id"] = fingerprint(commit)
        atomic_json(path / "lab.commit.json", commit)
        verify_lab(path)
        record("complete", image_count=len(data.rows), completed=list(summaries))
        return {"state": "complete", "lab_root": str(path), "lab_id": identity,
                "image_count": len(data.rows), "studies": summaries}
