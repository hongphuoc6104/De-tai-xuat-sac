"""Prepare and persist one bank-owned negative-anchor member state."""
from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from .dual_common import ArtifactStore, check_deadline, photo_sample_indices, photos_for_case, project_case
from .dual_mil import NoveltyLensState
from .dual_novelty import (
    NegativePrototypeBank,
    fit_negative_prototype_bank,
    fit_robust_novelty_scale,
    query_novelty,
    validate_anchor_head_membership,
)
from .io import fingerprint

_LENSES = ("4", "10", "40")
_SCALE_SAMPLE_CAP = 256


def _scope_path(scope: str) -> str:
    if (not isinstance(scope, str) or not scope or scope.startswith("/") or "\\" in scope
            or any(part in {"", ".", ".."} for part in PurePosixPath(scope).parts)
            or PurePosixPath(scope).as_posix() != scope):
        raise ValueError("scope must be a canonical relative artifact path.")
    return scope


def _array_sha256(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite float32 array.")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _dataset_identity(dataset: Any) -> dict[str, Any]:
    identity = getattr(dataset, "identity", None)
    if not isinstance(identity, Mapping):
        raise TypeError("Public datasets must expose a mapping identity.")
    keys = ("feature_id", "bundle_id", "split_id", "governance_id", "cache_id", "source_manifest_sha256", "mode")
    return {key: identity[key] for key in keys if key in identity}


def _row_identity(ref: Mapping[str, Any], *, expected_case: str, lens: str) -> dict[str, Any]:
    case_id = ref.get("case_id")
    if case_id != expected_case or str(ref.get("lens")) != lens:
        raise ValueError("Sample ref case/lens differs from its training-scope identity.")
    label = ref.get("source_case_label")
    if type(label) is not int or label not in (0, 1):
        raise ValueError("Sample source provenance needs the reviewed case-level label.")
    tile_id = ref.get("tile_id")
    photo_id = ref.get("photo_id")
    if not isinstance(tile_id, str) or not tile_id or not isinstance(photo_id, str) or not photo_id:
        raise ValueError("Sample refs require governed tile IDs and verified source-photo IDs.")
    return {"case_id": case_id, "target": label, "photo_id": photo_id, "row_id": tile_id}


def _verify_reused_member(
    artifact: Mapping[str, Any],
    *,
    base_identity: Mapping[str, Any],
    member_id: str,
    head_train_ids: tuple[str, ...],
) -> dict[str, Any]:
    if artifact.get("schema") != 1 or artifact.get("member_id") != member_id:
        raise ValueError("Committed member state has a different protocol or member token.")
    if artifact.get("base_identity") != dict(base_identity):
        raise ValueError("Committed member state belongs to another scope/source/transform/membership.")
    if tuple(artifact.get("head_train_ids", ())) != head_train_ids:
        raise ValueError("Committed member state has different anchor-excluded head membership.")
    anchor_case_id = artifact.get("anchor_case_id")
    anchor_token = fingerprint({"anchor_case_id": anchor_case_id})
    if artifact.get("anchor_token") != anchor_token:
        raise ValueError("Committed member anchor token does not match its case identity.")
    if artifact.get("member_id") != fingerprint({"scope": base_identity["scope"], "anchor_token": anchor_token}):
        raise ValueError("Committed member token does not match its scope and anchor.")
    lens_states = artifact.get("lens_states")
    if not isinstance(lens_states, dict) or set(lens_states) != set(_LENSES):
        raise ValueError("Committed member state is missing a lens bank or scale.")
    for lens, state in lens_states.items():
        if not isinstance(state, NoveltyLensState) or state.bank.lens != lens:
            raise ValueError("Committed member state contains an invalid lens state.")
        if state.bank.anchor_case_id != anchor_case_id:
            raise ValueError("Committed per-lens banks disagree on their anchor.")
        transform = base_identity["transforms"][lens]
        if (_array_sha256(state.mean, f"{lens}x cached mean") != transform["mean_sha256"]
                or _array_sha256(state.components, f"{lens}x cached components") != transform["components_sha256"]):
            raise ValueError("Committed member transform arrays differ from their input identity.")
    identity = artifact.get("member_identity")
    if not isinstance(identity, dict) or fingerprint(identity) != artifact.get("member_identity_id"):
        raise ValueError("Committed member identity hash is invalid.")
    if identity.get("base_identity_id") != fingerprint(base_identity):
        raise ValueError("Committed member identity does not bind its input scope.")
    if {lens: state.bank.bank_id for lens, state in lens_states.items()} != identity.get("bank_ids"):
        raise ValueError("Committed member identity does not bind the loaded lens banks.")
    expected_scales = identity.get("scales")
    if not isinstance(expected_scales, dict) or set(expected_scales) != set(lens_states):
        raise ValueError("Committed member identity does not bind every lens scale.")
    for lens, state in lens_states.items():
        scale = state.scale
        expected = expected_scales[lens]
        if any(expected.get(key) != value for key, value in {
            "median": scale.median, "iqr": scale.iqr, "epsilon": scale.epsilon,
            "degenerate": scale.degenerate,
        }.items()):
            raise ValueError("Committed member scale differs from its identity.")
    return dict(artifact)


def prepare_member_states(
    dataset_all: Any,
    dataset_factory: Callable[[Sequence[str]], Any],
    transforms: Mapping[str, Any],
    source_map: Mapping[str, Any],
    train_ids: Sequence[str],
    anchor: str,
    store: ArtifactStore,
    scope: str,
    deadline: float,
) -> dict[str, Any]:
    """Fit or load an immutable anchor bank and train-only novelty scale.

    The transform must already be committed for exactly ``train_ids``. A
    member uses one negative anchor for every lens, excludes it from head and
    scale samples, and stores the committed state at
    ``scopes/<scope>/members/<anchor_token>/state.pkl``. Repeated calls verify
    the receipt and all input identities before returning the same state.
    """
    check_deadline(deadline)
    scope_name = _scope_path(scope)
    if not isinstance(store, ArtifactStore):
        raise TypeError("store must be the branch B ArtifactStore.")
    if not isinstance(anchor, str) or not anchor:
        raise ValueError("anchor must be a nonempty case ID.")
    case_ids = tuple(train_ids)
    if not case_ids or len(set(case_ids)) != len(case_ids) or any(not isinstance(case, str) or not case for case in case_ids):
        raise ValueError("train_ids must be a nonempty sequence of unique case IDs.")
    if anchor not in case_ids or set(case_ids) - set(dataset_all.case_ids):
        raise ValueError("Anchor and all transform training cases must belong to dataset_all.")
    if set(transforms.get("members", {})) != set(case_ids):
        raise ValueError("Committed transform membership must exactly match train_ids.")
    if transforms.get("dimensions") != 32 or not isinstance(transforms.get("lenses"), Mapping):
        raise ValueError("B member states require the committed per-lens PCA32 transform.")
    source_identity = source_map.get("source_identity")
    source_rows = source_map.get("rows")
    source_order = source_map.get("case_tile_order")
    if not isinstance(source_identity, Mapping) or not isinstance(source_rows, Mapping) or not isinstance(source_order, Mapping):
        raise ValueError("Verified source-photo map must include identity, rows, and case tile order.")
    source_identity = {
        key: source_identity[key]
        for key in ("feature_id", "bundle_id", "split_id", "governance_id", "cache_id", "source_manifest_sha256", "mode")
        if key in source_identity
    }
    if source_identity != _dataset_identity(dataset_all):
        raise ValueError("Source-photo map and public dataset identities differ.")
    anchor_record = dataset_all.records[anchor]
    if type(anchor_record.get("target")) is not int or anchor_record["target"] != 0:
        raise ValueError("Each member anchor must be a reviewed negative case.")
    head_train_ids = tuple(case_id for case_id in case_ids if case_id != anchor)
    head_targets = tuple(dataset_all.records[case_id]["target"] for case_id in head_train_ids)
    smoke = _dataset_identity(dataset_all).get("mode") == "smoke"
    validate_anchor_head_membership(
        head_train_ids,
        head_targets,
        anchor_case_id=anchor,
        production=not smoke,
    )

    anchor_token = fingerprint({"anchor_case_id": anchor})
    member_id = fingerprint({"scope": scope_name, "anchor_token": anchor_token})
    state_name = f"scopes/{scope_name}/members/{anchor_token}/state.pkl"
    transform_identity: dict[str, Any] = {}
    for lens in _LENSES:
        check_deadline(deadline)
        lens_data = transforms["lenses"].get(lens)
        if not isinstance(lens_data, Mapping):
            raise ValueError(f"Committed transform is missing {lens}x data.")
        pca = lens_data.get("pca")
        if pca is None or getattr(pca, "n_components_", None) != 32:
            raise ValueError(f"Committed {lens}x transform does not expose PCA32 state.")
        mean = np.asarray(pca.mean_)
        components = np.asarray(pca.components_)
        if mean.dtype != np.float32 or mean.ndim != 1 or components.dtype != np.float32:
            raise ValueError(f"Committed {lens}x PCA state must use float32 mean and components.")
        if components.shape != (32, mean.size):
            raise ValueError(f"Committed {lens}x PCA components have an invalid shape.")
        projected = np.asarray(lens_data.get("sample_projected"))
        refs = lens_data.get("sample_refs")
        if projected.dtype != np.float32 or projected.ndim != 2 or projected.shape != (len(refs), 32):
            raise ValueError(f"Committed {lens}x sample vectors and refs differ or are not PCA32 float32.")
        for ref in refs:
            tile_id = ref.get("tile_id")
            source = source_rows.get(tile_id)
            if (source is None or source.get("case_id") != ref.get("case_id")
                    or source.get("lens") != lens or source.get("photo_id") != ref.get("photo_id")):
                raise ValueError("PCA sample refs differ from the verified source-photo map.")
        transform_identity[lens] = {
            "mean_sha256": _array_sha256(mean, f"{lens}x mean"),
            "components_sha256": _array_sha256(components, f"{lens}x components"),
            "sample_sha256": _array_sha256(projected, f"{lens}x sample_projected"),
            "sample_ref_ids_sha256": fingerprint([
                {"case_id": ref.get("case_id"), "tile_id": ref.get("tile_id"), "photo_id": ref.get("photo_id")}
                for ref in refs
            ]),
        }
    base_identity = {
        "schema": 1,
        "scope": scope_name,
        "dataset_identity": _dataset_identity(dataset_all),
        "source_map_identity": source_identity,
        "train_case_ids": case_ids,
        "head_train_case_ids": head_train_ids,
        "anchor_token": anchor_token,
        "transform_dimensions": 32,
        "transform_seed": transforms.get("seed"),
        "transforms": transform_identity,
    }
    base_identity_id = fingerprint(base_identity)
    existing = store.read(state_name)
    if existing is not None:
        artifact = pickle.loads(existing)
        if not isinstance(artifact, dict):
            raise ValueError("Committed member payload is not a state mapping.")
        return _verify_reused_member(
            artifact,
            base_identity=base_identity,
            member_id=member_id,
            head_train_ids=head_train_ids,
        )

    train_dataset = dataset_factory(head_train_ids)
    if set(train_dataset.case_ids) != set(head_train_ids) or len(train_dataset.case_ids) != len(head_train_ids):
        raise ValueError("dataset_factory must return exactly the anchor-excluded training membership.")
    for key, value in _dataset_identity(dataset_all).items():
        other = _dataset_identity(train_dataset).get(key)
        if other is not None and other != value:
            raise ValueError(f"dataset_factory changed the verified {key} source identity.")
    loaded_tile_order = getattr(train_dataset, "loaded_tile_order", None)
    if not callable(loaded_tile_order):
        raise TypeError("dataset_factory output must expose verified loaded_tile_order(case_id).")

    banks: dict[str, NegativePrototypeBank] = {}
    for lens in _LENSES:
        check_deadline(deadline)
        lens_data = transforms["lenses"][lens]
        projected = np.asarray(lens_data["sample_projected"])
        refs = lens_data["sample_refs"]
        anchor_indices = [index for index, ref in enumerate(refs) if ref.get("case_id") == anchor]
        if not anchor_indices:
            raise ValueError(f"Anchor has no train-only {lens}x prototype reservoir.")
        anchor_vectors = projected[np.asarray(anchor_indices, dtype=np.int64)]
        bank_rows = [
            _row_identity(refs[index], expected_case=anchor, lens=lens)
            for index in anchor_indices
        ]
        if any(row["target"] != 0 for row in bank_rows):
            raise ValueError("Negative prototype reservoir contains a positive case label.")
        banks[lens] = fit_negative_prototype_bank(
            anchor_vectors,
            bank_rows,
            lens=lens,
            anchor_case_id=anchor,
            random_state=int(transforms.get("seed", 42)),
        )

    all_scale_scores: dict[str, list[np.ndarray]] = {lens: [] for lens in _LENSES}
    all_scale_photos: dict[str, list[str]] = {lens: [] for lens in _LENSES}
    all_scale_metadata: dict[str, list[dict[str, str]]] = {lens: [] for lens in _LENSES}
    scale_sample_ids: dict[str, list[dict[str, Any]]] = {lens: [] for lens in _LENSES}
    for case_id in head_train_ids:
        check_deadline(deadline)
        projected_by_lens = project_case(train_dataset, case_id, transforms)
        if loaded_tile_order(case_id) != source_order[case_id]:
            raise ValueError("dataset_factory feature order differs from the verified source-photo map.")
        for lens in _LENSES:
            check_deadline(deadline)
            photo_ids = photos_for_case(source_map, case_id, lens)
            vectors = projected_by_lens[lens]
            if len(vectors) != len(photo_ids):
                raise ValueError("Projected query rows and source-photo lineage differ in count/order.")
            if not len(vectors):
                continue
            selected = photo_sample_indices(photo_ids, _SCALE_SAMPLE_CAP, int(transforms.get("seed", 42)))
            if not len(selected):
                raise ValueError("No train rows were selected for novelty scale fitting.")
            selected_photos = [photo_ids[int(index)] for index in selected]
            scores = query_novelty(vectors[selected], banks[lens]).distances
            selected_ids = [source_map["case_tile_order"][case_id][lens][int(index)] for index in selected]
            all_scale_scores[lens].append(scores)
            all_scale_photos[lens].extend(selected_photos)
            all_scale_metadata[lens].extend(
                {"case_id": case_id, "partition": "train"} for _ in selected
            )
            scale_sample_ids[lens].append({"case_id": case_id, "tile_ids": selected_ids,
                                           "photo_ids": selected_photos})

    lens_states: dict[str, NoveltyLensState] = {}
    scale_identities: dict[str, dict[str, Any]] = {}
    for lens in _LENSES:
        check_deadline(deadline)
        if not all_scale_scores[lens]:
            raise ValueError(f"T-minus-anchor head training has no observed {lens}x rows for scale fitting.")
        values = np.concatenate(all_scale_scores[lens]).astype(np.float32, copy=False)
        scale = fit_robust_novelty_scale(
            values,
            all_scale_photos[lens],
            all_scale_metadata[lens],
            anchor_case_id=anchor,
            allowed_train_case_ids=head_train_ids,
        )
        pca = transforms["lenses"][lens]["pca"]
        lens_states[lens] = NoveltyLensState(
            mean=np.asarray(pca.mean_, dtype=np.float32),
            components=np.asarray(pca.components_, dtype=np.float32),
            bank=banks[lens],
            scale=scale,
        )
        scale_identities[lens] = {
            "median": scale.median,
            "iqr": scale.iqr,
            "epsilon": scale.epsilon,
            "degenerate": scale.degenerate,
            "sample_ids_sha256": fingerprint(scale_sample_ids[lens]),
        }

    member_identity = {
        "protocol": "negative_anchor_member_v1",
        "base_identity_id": base_identity_id,
        "member_id": member_id,
        "anchor_token": anchor_token,
        "bank_ids": {lens: banks[lens].bank_id for lens in _LENSES},
        "scales": scale_identities,
        "head_train_case_ids": head_train_ids,
    }
    artifact = {
        "schema": 1,
        "scope": scope_name,
        "anchor_case_id": anchor,
        "anchor_token": anchor_token,
        "member_id": member_id,
        "base_identity": base_identity,
        "head_train_ids": head_train_ids,
        "lens_states": lens_states,
        "member_identity": member_identity,
        "member_identity_id": fingerprint(member_identity),
    }
    check_deadline(deadline)
    store.publish(state_name, pickle.dumps(artifact, protocol=5))
    return artifact


__all__ = ["prepare_member_states"]
