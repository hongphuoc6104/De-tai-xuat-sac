"""Producer-to-CLI/cache/model E2E on tiny synthetic whole-image features."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from histology_data import image_classification_r0_r3 as ablation, image_classification_studies as studies
from histology_data.io import file_hash
from tests.test_image_classification_studies import source as producer_fixture


@pytest.fixture(scope="module")
def frozen_source(tmp_path_factory: pytest.TempPathFactory) -> ablation.Source:
    """Run the actual original-image/cache/v1 study producers before v2."""
    root = tmp_path_factory.mktemp("r0-r3-source")
    data, _ = producer_fixture.__wrapped__(root)
    result = studies.run_study(data, studies.StudyConfig(lambda_values=(.5,), budget_minutes=5, reserve_minutes=0))
    return ablation.load_source(data.run_root, Path(result["study_root"]))


@pytest.fixture
def config() -> ablation.Config:
    return ablation.Config(lambdas=(.5,), ratios=(.3,), bootstrap_repeats=20, budget_minutes=10, reserve_minutes=0)


def test_actual_producer_shared_assignments_and_weights(frozen_source: ablation.Source) -> None:
    checks = ablation.preflight(frozen_source)
    assert len(checks) == 72
    assert all(set(c["class_patient_support"]) == {"0", "1"} for c in checks)
    bad = replace(frozen_source, features=frozen_source.features * np.nan)
    with pytest.raises(ValueError, match="finite"):
        ablation.preflight(bad)


def test_cli_worker_executes_as_real_module(tmp_path: Path) -> None:
    """Production worker entry point works under python -m, not only imports."""
    folder = tmp_path / "worker"
    folder.mkdir()
    request = {"arm": "R2", "lambda": .5, "ratio": 0, "iterations": [2000], "tol": 1e-4, "seed": 17}
    (folder / "request.json").write_text(json.dumps(request))
    x = np.array([[-2, 1], [-1, .1], [1, -.1], [2, -1]], dtype=np.float32)
    np.savez(folder / "input.npz", x=x, y=[0, 0, 1, 1], weights=np.ones(4), evaluate=x)
    completed = subprocess.run([sys.executable, "-m", "histology_data.image_classification_r0_r3", "--worker", str(folder)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (folder / "model.joblib").is_file()
    assert json.loads((folder / "fit_receipt.json").read_text())["converged"]


def test_fit_partial_resume_then_complete_and_tamper(frozen_source: ablation.Source, config: ablation.Config, tmp_path: Path) -> None:
    first = ablation.run_suite(frozen_source, tmp_path, config, max_new_fits=1)
    assert first["state"] == "partial" and first["new_fits"] == 1
    root = Path(first["study_root"])
    original = {p: file_hash(p) for p in root.rglob("commit.json")}
    result = ablation.run_suite(frozen_source, tmp_path, config, refit=True)
    assert result["state"] == "complete"
    assert all(file_hash(p) == sha for p, sha in original.items())
    assert len(result["report"]["metrics"]) == 5
    assert (root / "report/BAO_CAO.md").is_file()
    assert (root / "report/auroc_comparison.png").stat().st_size > 1000
    assert len(list(root.rglob("fit_receipt.json"))) == 72
    assert result["new_fits"] == 71
    assert file_hash(frozen_source.baseline_root / "study.commit.json") == frozen_source.receipt["baseline_commit_sha256"]
    for arm in ("R1", "R2", "R3"):
        prediction = ablation.predict_vector(root, arm, frozen_source.features[0], 4)
        assert 0 <= prediction["score"] <= 1 and prediction["image_target"] in (0, 1)
    with pytest.raises(ValueError, match="known objective"):
        ablation.predict_vector(root, "R1", frozen_source.features[0])
    again = ablation.run_suite(frozen_source, tmp_path, config)
    assert again["state"] == "complete"
    assert not (root / "run.lock").exists()
    model = next((root / "arms/R2").rglob("model.joblib"))
    original_bytes = model.read_bytes()
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ablation.run_suite(frozen_source, tmp_path, config)
    model.write_bytes(original_bytes)
    path = root / "report/R2_oof.csv"
    path.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        ablation.run_suite(frozen_source, tmp_path, config)


def test_bounded_worker_reaped_after_real_timeout(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        ablation._bounded([sys.executable, "-c", "import time; time.sleep(5)"], .05, tmp_path / "log")
    assert (tmp_path / "log").exists()


def test_normalizer_scaler_model_and_solver_control(config: ablation.Config, tmp_path: Path) -> None:
    import joblib

    rows = [{"image_id": str(i), "patient_id": str(i), "objective_lens": (4, 10, 40)[i % 3], "image_target": i // 3} for i in range(6)]
    x = np.array([[-3, 1], [-2, .5], [-1, .1], [1, -.1], [2, -.5], [3, -1]], dtype=np.float32)
    source = ablation.Source(x, rows, {}, [], {"fixture": True})
    runner = ablation.Runner(source, tmp_path, config)
    p, _, folder = runner.fit("R2", "fixture", "all", "inner-01", np.array([0, 1, 3, 4]), np.arange(6), .5)
    model = joblib.load(folder / "model.joblib")
    assert list(model.named_steps) == ["normalizer", "scaler", "classifier"]
    assert model.predict_proba(x)[:, 1] == pytest.approx(p)
    assert model.named_steps["scaler"].mean_ == pytest.approx(np.mean((x / np.linalg.norm(x, axis=1, keepdims=True))[[0, 1, 3, 4]], axis=0, dtype=np.float64))


def test_invalid_grid_and_normalized_identity(config: ablation.Config) -> None:
    with pytest.raises(ValueError, match="separate L2"):
        replace(config, ratios=(0,)).validate()
    assert ablation.Config(budget_minutes=200).scientific() == ablation.Config().scientific()
    assert json.loads(json.dumps(config.scientific())) == config.scientific()


def test_production_cli_dispatcher_budget_boundary(frozen_source: ablation.Source, tmp_path: Path) -> None:
    completed = subprocess.run([sys.executable, "-m", ablation.WORKER_MODULE, "--release-root", str(frozen_source.release_root), "--baseline-root", str(frozen_source.baseline_root), "--output", str(tmp_path), "--max-new-fits", "1"], capture_output=True, text=True)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    status = json.loads(next(tmp_path.glob("*/status.json")).read_text())
    assert status["state"] == "partial" and status["new_fits"] == 1


def test_eighteen_patient_shared_inner_leakage_rejected() -> None:
    rows = [{"image_id": str(i), "patient_id": f"synthetic-{i // 6}",
             "objective_lens": (4, 10, 40)[(i % 6) // 2], "image_target": i % 2}
            for i in range(108)]
    def splits(indices: list[int]) -> list[dict]:
        patients = sorted({rows[i]["patient_id"] for i in indices})
        first = set(patients[:len(patients) // 2])
        a = [str(i) for i in indices if rows[i]["patient_id"] in first]
        b = [str(i) for i in indices if rows[i]["patient_id"] not in first]
        return [{"split": 1, "train_image_ids": a, "validation_image_ids": b},
                {"split": 2, "train_image_ids": b, "validation_image_ids": a}]
    baseline = {"folds": [], "deployment": {"all": {"tuning": {"inner_splits": splits(list(range(108)))}}}}
    oof = []
    for fold in (1, 2, 3):
        test = [i for i in range(108) if i // 36 == fold - 1]
        train = [i for i in range(108) if i not in test]
        baseline["folds"].append({"outer_fold": fold, "train_image_ids": [str(i) for i in train],
                                  "test_image_ids": [str(i) for i in test],
                                  "heads": {"all": {"tuning": {"inner_splits": splits(train)}}}})
        oof.extend({**rows[i], "outer_fold": fold, "predicted_probability": .9 if i % 2 else .1,
                    "selected_threshold": .5, "prediction_at_0_5": i % 2, "prediction_at_selected_threshold": i % 2} for i in test)
    source = ablation.Source(np.ones((108, 8)), rows, baseline, oof, {})
    assert len(ablation.preflight(source)) == 72
    inner = baseline["folds"][0]["heads"]["all"]["tuning"]["inner_splits"][0]
    inner["train_image_ids"].append(inner["validation_image_ids"].pop())
    with pytest.raises(ValueError, match="leaks patients"):
        ablation.preflight(source)


def test_repeated_real_worker_timeout_requires_attention(config: ablation.Config, tmp_path: Path) -> None:
    rows = [{"image_id": str(i), "patient_id": str(i), "objective_lens": 4, "image_target": i // 2} for i in range(4)]
    source = ablation.Source(np.ones((4, 2)), rows, {}, [], {})
    runner = ablation.Runner(source, tmp_path, replace(config, fit_timeout_seconds=.001))
    with pytest.raises(TimeoutError):
        runner.fit("R2", "fixture", "all", "inner-01", np.arange(4), np.arange(4), .5)
    with pytest.raises(RuntimeError, match="Repeated"):
        runner.fit("R2", "fixture", "all", "inner-01", np.arange(4), np.arange(4), .5)
    assert not list(tmp_path.rglob("commit.json"))


def test_patient_macro_keeps_bootstrap_copies() -> None:
    rows = [{"image_id": str(i), "patient_id": "same" if i < 2 else "other", "bootstrap_patient": str(i),
             "objective_lens": 4, "image_target": 1, "predicted_probability": .9 if i < 2 else .1,
             "prediction_at_0_5": int(i < 2), "prediction_at_selected_threshold": int(i < 2)} for i in range(3)]
    result = ablation.report_metrics(rows)
    assert result["all"]["fixed_0_5"]["patient_macro"]["sensitivity"] == pytest.approx(2 / 3)
    assert result["all"]["AUROC"] is None


def test_reader_does_not_write_completed_source(frozen_source: ablation.Source) -> None:
    root = frozen_source.release_root
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob('*') if p.is_file()}
    reread = ablation.load_source(root, frozen_source.baseline_root)
    after = {str(p.relative_to(root)): file_hash(p) for p in root.rglob('*') if p.is_file()}
    assert reread.receipt == frozen_source.receipt
    assert after == before


def test_normalized_solvers_use_high_accuracy_and_double_precision(tmp_path: Path) -> None:
    import joblib

    rng = np.random.default_rng(37)
    x = (5 + rng.normal(size=(64, 12))).astype(np.float32)
    x[48:, :2] += 3
    rows = [{"image_id": str(i), "patient_id": str(i // 2), "objective_lens": 4, "image_target": i % 2} for i in range(64)]
    runner = ablation.Runner(ablation.Source(x, rows, {}, [], {}), tmp_path, ablation.Config())
    predictions = []
    for arm in ("R2", "R2_SOLVER_CONTROL"):
        p, receipt, path = runner.fit(arm, "numeric_fixture", "all", "inner-01", np.arange(48), np.arange(48, 64), .5)
        model = joblib.load(path / "model.joblib")
        assert model[-1].coef_.dtype == np.float64
        assert model[-1].tol <= 1e-8
        assert not receipt["conditioning"]
        assert receipt["solver_weight_divisor"] == 1
        if arm == "R2_SOLVER_CONTROL":
            request = json.loads((path / "commit.json").read_text())["binding"]["request"]
            assert request["iterations"] == [2000] and receipt["options"]["maxiter"] == 2000
            assert request["tol"] == receipt["options"]["gtol"] == 1e-9
        predictions.append(p)
    assert np.max(np.abs(predictions[0] - predictions[1])) < 1e-3


def test_conditioned_worker_preserves_logical_objective_and_pipeline(tmp_path: Path) -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    rng = np.random.default_rng(719)
    x = (rng.normal(size=(48, 9)) + 2).astype(np.float32)
    y = np.arange(48) % 2
    weights = np.ones(48)
    weights[0] = 32
    folder = tmp_path / 'conditioned'
    folder.mkdir()
    request = {'arm': 'R2_SAGA_CONTROL', 'lambda': .5, 'ratio': 0,
               'iterations': [1, 10000], 'tol': 1e-8, 'seed': 17, 'condition_saga': True}
    (folder / 'request.json').write_text(json.dumps(request))
    np.savez(folder / 'input.npz', x=x, y=y, weights=weights, evaluate=x[32:])
    result = subprocess.run([sys.executable, '-m', ablation.WORKER_MODULE, '--worker', str(folder)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    fitted = joblib.load(folder / 'model.joblib')
    receipt = json.loads((folder / 'fit_receipt.json').read_text())
    assert receipt['conditioning'] and receipt['solver_weight_divisor'] > 1
    assert [a['conditioning'] for a in receipt['attempts']] == [False, True]
    assert not receipt['attempts'][0]['converged']
    assert receipt['solver_C'] * receipt['solver_weight_sum'] == pytest.approx(receipt['C'] * weights.sum())
    reference = Pipeline([('normalizer', Normalizer()), ('scaler', StandardScaler())])
    z = reference.fit_transform(x).astype(np.float64)
    model = LogisticRegression(C=receipt['C'], solver='lbfgs', tol=1e-10, max_iter=10000)
    model.fit(z, y, sample_weight=weights)
    assert np.max(np.abs(fitted.predict_proba(x)[:, 1] - model.predict_proba(z)[:, 1])) < 1e-5
    assert np.array_equal(fitted[:-1].transform(x), reference.transform(x))
    logits = fitted.decision_function(x)
    objective = np.average(np.logaddexp(0, logits) - y * logits, weights=weights) + .25 * np.square(fitted[-1].coef_).sum()
    assert receipt['objective'] == pytest.approx(objective, abs=1e-12)
    with np.load(folder / 'predictions.npz') as predictions:
        assert np.max(np.abs(predictions['evaluate'] - fitted.predict_proba(x[32:])[:, 1])) < 1e-12


def test_reuse_identical_lbfgs_units_only_and_source_unchanged(frozen_source: ablation.Source, config: ablation.Config, tmp_path: Path) -> None:
    old = ablation.run_suite(frozen_source, tmp_path / 'old', replace(config, condition_saga=False), max_new_fits=1)
    old_root = Path(old['study_root'])
    before = {str(p.relative_to(old_root)): file_hash(p) for p in old_root.rglob('*') if p.is_file()}
    new = ablation.run_suite(frozen_source, tmp_path / 'new', config, max_new_fits=0, reuse_lbfgs_root=old_root)
    new_root = Path(new['study_root'])
    assert new['state'] == 'partial' and new['new_fits'] == 0
    reused = json.loads((new_root / 'reused_units.json').read_text())
    assert len(reused) == 1 and all(p.startswith('arms/R2/') for p in reused)
    for relative in reused:
        assert (old_root / relative / 'commit.json').read_bytes() == (new_root / relative / 'commit.json').read_bytes()
    assert before == {str(p.relative_to(old_root)): file_hash(p) for p in old_root.rglob('*') if p.is_file()}
    # R1 can import identical units, while SAGA must execute its new recipe.
    scope, _, _, inner = ablation.scopes(frozen_source)[0]
    split = inner[0]
    train = frozen_source.positions(split['train_image_ids'])
    evaluate = frozen_source.positions(split['validation_image_ids'])
    old_runner = ablation.Runner(frozen_source, old_root, config)
    p, _, unit = old_runner.fit('R1', scope, '4', 'inner-01', train, evaluate, .5)
    new_runner = ablation.Runner(frozen_source, new_root, config, max_new_fits=0, reuse_lbfgs_root=old_root)
    reused_p, _, reused_unit = new_runner.fit('R1', scope, '4', 'inner-01', train, evaluate, .5)
    assert np.array_equal(p, reused_p) and file_hash(unit / 'model.joblib') == file_hash(reused_unit / 'model.joblib')
    with pytest.raises(TimeoutError):
        new_runner.fit('R3', scope, 'all', 'inner-01', train, evaluate, .5, .3)
    descriptor_path = old_root / 'config.json'
    descriptor = json.loads(descriptor_path.read_text())
    for key, value in [('implementation_sha256', 'unknown'), ('source', {'changed': True}),
                       ('baseline_inner_sha256', 'different'), ('versions', {}), ('settings', {})]:
        descriptor_path.write_text(json.dumps({**descriptor, key: value}))
        with pytest.raises(ValueError, match='Reusable L-BFGS'):
            ablation.run_suite(frozen_source, tmp_path / 'reject', config, max_new_fits=0, reuse_lbfgs_root=old_root)
    descriptor_path.write_text(json.dumps(descriptor))
    (old_root / 'run.lock').write_text('{}')
    with pytest.raises(ValueError, match='Reusable L-BFGS'):
        ablation.run_suite(frozen_source, tmp_path / 'reject', config, max_new_fits=0, reuse_lbfgs_root=old_root)
    (old_root / 'run.lock').unlink()
    (unit / 'model.joblib').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        ablation.Runner(frozen_source, tmp_path / 'tamper', config, 0, old_root).fit('R1', scope, '4', 'inner-01', train, evaluate, .5)


def test_second_sigint_during_cleanup_does_not_orphan_worker(tmp_path: Path) -> None:
    import os
    import signal
    import time

    pid_path, marker = tmp_path / 'child.pid', tmp_path / 'cleanup.started'
    child_code = f"import os,time;from pathlib import Path;Path({str(pid_path)!r}).write_text(str(os.getpid()));time.sleep(30)"
    parent_code = f'''import subprocess,time
from pathlib import Path
from histology_data.image_classification_r0_r3 import _bounded
original=subprocess.Popen
class Observed(original):
 def poll(self):
  Path({str(marker)!r}).write_text('cleanup')
  time.sleep(.3)
  return super().poll()
subprocess.Popen=Observed
_bounded([{sys.executable!r},'-c',{child_code!r}],60,Path({str(tmp_path / 'child.log')!r}))
'''
    with (tmp_path / 'parent.log').open('w') as log:
        parent = subprocess.Popen([sys.executable, '-c', parent_code], stdout=log, stderr=log, start_new_session=True)
        child_pid = None
        try:
            deadline = time.monotonic() + 10
            while not pid_path.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert pid_path.exists()
            child_pid = int(pid_path.read_text())
            parent.send_signal(signal.SIGINT)
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists()
            parent.send_signal(signal.SIGINT)
            parent.wait(timeout=10)
            assert not Path(f'/proc/{child_pid}/cmdline').exists(), 'Interrupted cleanup orphaned the real worker'
        finally:
            if child_pid is not None:
                try:os.kill(child_pid, signal.SIGTERM)
                except ProcessLookupError:pass
            if parent.poll() is None:
                parent.terminate()
                parent.wait(timeout=5)


def test_retry_preserves_predictions_when_coordinates_and_weights_change(tmp_path: Path) -> None:
    rng = np.random.default_rng(812)
    x = (rng.normal(size=(48, 9)) + 2).astype(np.float32)
    weights = np.ones(48)
    weights[0] = 32
    folder = tmp_path / 'warm-worker'
    folder.mkdir()
    (folder / 'request.json').write_text(json.dumps({'arm': 'R2_SAGA_CONTROL', 'lambda': .5, 'ratio': 0,
        'iterations': [1, 2, 10000], 'tol': 1e-8, 'seed': 17, 'condition_saga': True}))
    np.savez(folder / 'input.npz', x=x, y=np.arange(48) % 2, weights=weights, evaluate=x[32:])
    trace = folder / 'continuity.json'
    code = f'''import json,runpy,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
original=LogisticRegression.fit
previous={{}}
marks=[]
def observed(self,X,y,*args,**kwargs):
 weights=np.asarray(kwargs['sample_weight'])
 if previous:
  assert hasattr(self,'coef_'), 'Retry discarded the previous real model'
  assert np.max(np.abs(self.predict_proba(X)-previous['probabilities']))<1e-12, 'Coordinate translation changed predictions before fitting'
  assert np.allclose(self.C*weights,previous['C_weight'],rtol=1e-12,atol=1e-12), 'Retry changed the logical objective'
  marks.append('continuity-preserved')
 result=original(self,X,y,*args,**kwargs)
 previous.update(probabilities=self.predict_proba(X),C_weight=self.C*weights)
 Path({str(trace)!r}).write_text(json.dumps(marks))
 return result
LogisticRegression.fit=observed
sys.argv=[{ablation.WORKER_MODULE!r},'--worker',{str(folder)!r}]
runpy.run_module({ablation.WORKER_MODULE!r},run_name='__main__',alter_sys=True)
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(trace.read_text()) == ['continuity-preserved', 'continuity-preserved']
    assert json.loads((folder / 'fit_receipt.json').read_text())['converged']
