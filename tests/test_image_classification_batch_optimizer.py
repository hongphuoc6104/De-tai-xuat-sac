"""Numerical equivalence, independent certificates and serialized inference."""
import json
import subprocess
import sys

import joblib
import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from histology_data.image_classification_batch_optimizer import (
    CertifiedElasticNetClassifier,
    certificate,
    loss_gradient,
)


def test_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(24, 5))
    y = np.arange(24) % 2
    rho = rng.uniform(.1, 2, 24)
    rho /= rho.sum()
    theta = rng.uniform(.1, 1, 11)
    _, gradient = loss_gradient(theta, x, y, rho, .5, .3)
    numeric = []
    for i in range(len(theta)):
        step = np.eye(len(theta))[i] * 1e-6
        numeric.append((loss_gradient(theta + step, x, y, rho, .5, .3)[0]
                        - loss_gradient(theta - step, x, y, rho, .5, .3)[0]) / 2e-6)
    assert np.max(np.abs(gradient - numeric)) < 1e-7


@pytest.mark.parametrize('ratio', [0., .3, .7])
def test_same_weighted_objective_as_sklearn(ratio):
    rng = np.random.default_rng(72)
    x = rng.normal(size=(80, 12))
    y = (rng.random(80) < expit(x[:, 0] - x[:, 1])).astype(int)
    weights = rng.uniform(.2, 3, 80)
    model = CertifiedElasticNetClassifier(.05, ratio).fit(x, y, weights)
    reference = LogisticRegression(C=1 / (.05 * weights.sum()), solver='saga', l1_ratio=ratio,
                                   tol=1e-10, max_iter=50000).fit(x, y, sample_weight=weights)
    proof = model.fit_receipt_['certificate']
    assert proof['accepted'] and proof['gap'] <= 1e-10 and proof['dual_feasibility_max'] <= 1e-14
    assert np.max(np.abs(model.predict_proba(x) - reference.predict_proba(x))) < 1e-5
    bad = certificate(x, y, weights, model.coef_[0] + .1, model.intercept_[0], .05, ratio)
    assert not bad['accepted'] and bad['gap'] > proof['gap']


@pytest.mark.parametrize('penalty', [50., 5., .5, .05])
@pytest.mark.parametrize('ratio', [0., .3, .7])
def test_2048_correlated_features_extreme_weights(penalty, ratio):
    rng = np.random.default_rng(73)
    latent = rng.normal(size=(48, 8))
    x = latent @ rng.normal(size=(8, 2048)) / np.sqrt(8)
    x[:, -2:] = 0
    x[:, 0] += 100
    y = np.arange(48) % 2
    weights = np.ones(48)
    weights[0] = 100
    model = CertifiedElasticNetClassifier(penalty, ratio).fit(x, y, weights)
    assert model.fit_receipt_['certificate']['accepted']
    assert np.isfinite(model.predict_proba(x)).all() and model.coef_.shape == (1, 2048)
    if penalty == 50 and ratio:
        assert np.count_nonzero(model.coef_) == 0
        assert model.predict_proba(x)[:, 1] == pytest.approx(np.average(y, weights=weights))


def test_nonconvergence_validation_and_fresh_process_restore(tmp_path):
    rng = np.random.default_rng(74)
    x = rng.normal(size=(64, 12))
    y = np.arange(64) % 2
    with pytest.raises(RuntimeError, match='certification'):
        CertifiedElasticNetClassifier(.05, .3, max_iter=1).fit(x, y)
    with pytest.raises(ValueError, match='both binary'):
        CertifiedElasticNetClassifier().fit(x, np.zeros(64))
    with pytest.raises(ValueError, match='positive finite'):
        CertifiedElasticNetClassifier().fit(x, y, np.zeros(64))
    model = CertifiedElasticNetClassifier(.05, .3).fit(x, y)
    joblib.dump(model, tmp_path / 'model.joblib')
    np.save(tmp_path / 'x.npy', x)
    code = "import joblib,numpy as np,json,sys;from pathlib import Path;p=Path(sys.argv[1]);m=joblib.load(p/'model.joblib');print(json.dumps(m.predict_proba(np.load(p/'x.npy')).tolist()))"
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, check=True)
    assert np.array_equal(json.loads(result.stdout), model.predict_proba(x))
    assert np.isfinite(model.predict_proba(x * 1e6)).all()
    with pytest.raises(ValueError, match='dimension'):
        model.predict_proba(x[:, :3])


def test_failed_refit_cannot_reuse_previous_fitted_state():
    rng = np.random.default_rng(75)
    x = rng.normal(size=(64, 12))
    y = np.arange(64) % 2
    model = CertifiedElasticNetClassifier(.05, .3).fit(x, y)
    model.set_params(max_iter=1)
    with pytest.raises(RuntimeError, match='certification'):
        model.fit(x, y)
    assert not hasattr(model, 'coef_'), 'Failed refit retained stale fitted parameters'


def test_dual_equality_correction_is_conservative_lower_bound():
    rng = np.random.default_rng(76)
    x = rng.normal(size=(57, 11)) + 4
    y = np.arange(57) % 2
    weights = rng.uniform(.01, 5, 57)
    model = CertifiedElasticNetClassifier(.05, .3).fit(x, y, weights)
    proof = certificate(x, y, weights, model.coef_[0] + .2, model.intercept_[0] + .3, .05, .3)
    correction = proof['optimal_intercept_abs_bound'] * abs(proof['dual_equality_residual'])
    assert proof['dual_equality_correction'] == pytest.approx(correction, abs=1e-30)
    assert proof['dual'] == pytest.approx(proof['uncorrected_dual'] - correction, abs=1e-15)
    assert proof['dual'] <= model.fit_receipt_['certificate']['primal'] + 1e-12
    assert proof['optimal_intercept_abs_bound'] >= abs(model.intercept_[0])
    assert proof['gap'] > 0 and not proof['accepted']
