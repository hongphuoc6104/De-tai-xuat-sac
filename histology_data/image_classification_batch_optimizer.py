"""Certified full-batch weighted logistic ElasticNet, with a free intercept.

The box-constrained split beta=u-v solves the same convex objective as sklearn
logistic regression with C=1/(lambda*sum(sample_weight)). Numerical acceptance
uses an independent Fenchel dual certificate, never the optimizer flag alone.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, xlogy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

BACKEND = "certified-split-lbfgsb-v1"
GAP_TOLERANCE = 1e-10
KKT_TOLERANCE = 1e-6


def _residual(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(y == 0, expit(logits), -expit(-logits))


def loss_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray, rho: np.ndarray,
                  penalty: float, ratio: float) -> tuple[float, np.ndarray]:
    """Smooth split objective and analytic gradient; intercept is unpenalized."""
    p = x.shape[1]
    beta = theta[:p] - theta[p:2 * p]
    logits = x @ beta + theta[-1]
    residual = rho * _residual(logits, y)
    gradient = x.T @ residual + penalty * (1 - ratio) * beta
    value = (rho @ np.logaddexp(0, (1 - 2 * y) * logits)
             + penalty * (1 - ratio) / 2 * (beta @ beta)
             + penalty * ratio * theta[:2 * p].sum())
    return float(value), np.r_[gradient + penalty * ratio, -gradient + penalty * ratio, residual.sum()]


def certificate(x: np.ndarray, y: np.ndarray, weights: np.ndarray, beta: np.ndarray,
                bias: float, penalty: float, ratio: float) -> dict[str, Any]:
    """Independently evaluate the primal and a domain-feasible Fenchel dual.

    Dual residual mass is balanced by shrinking the larger sign, preserving all
    coordinate bounds. This function does not call the optimizer's loss/gradient.
    """
    rho = weights / weights.sum()
    logits = x @ beta + bias
    mu, tau = penalty * (1 - ratio), penalty * ratio
    primal = float(np.sum(rho * np.logaddexp(0, (1 - 2 * y) * logits))
                   + mu * np.sum(beta * beta) / 2 + tau * np.sum(np.abs(beta)))
    residual = rho * _residual(logits, y)
    derivative = x.T @ residual + mu * beta
    kkt_beta = np.where(beta != 0, np.abs(derivative + tau * np.sign(beta)),
                        np.maximum(np.abs(derivative) - tau, 0))
    kkt = float(max(np.max(kkt_beta), abs(residual.sum())))
    dual = residual.copy()
    positive, negative = dual > 0, dual < 0
    pos_mass, neg_mass = float(dual[positive].sum()), float(-dual[negative].sum())
    common = min(pos_mass, neg_mass)
    if pos_mass:
        dual[positive] *= common / pos_mass
    if neg_mass:
        dual[negative] *= common / neg_mass
    lower, upper = -rho * y, rho * (1 - y)
    feasibility = float(max(abs(dual.sum()), np.max(lower - dual), np.max(dual - upper), 0))
    raw_t = y + dual / rho
    domain_roundoff = float(max(0., np.max(-raw_t), np.max(raw_t - 1)))
    # Clipping repairs only bounded floating-point roundoff, never a materially
    # infeasible candidate. Signed equality residual stays visible in receipts.
    t = np.clip(raw_t, 0, 1)
    entropy = -float(rho @ (xlogy(t, t) + xlogy(1 - t, 1 - t)))
    v = x.T @ dual
    conjugate = float(np.square(np.maximum(np.abs(v) - tau, 0)).sum() / (2 * mu))
    dual_value = entropy - conjugate
    # A bounded free intercept converts residual floating-point dual equality
    # error into a conservative lower-bound correction (Fenchel inequality).
    prevalence = float(rho @ y)
    p0 = -float(xlogy(prevalence, prevalence) + xlogy(1 - prevalence, 1 - prevalence))
    beta_bound = np.sqrt(2 * p0 / mu)
    bias_bound = abs(np.log(prevalence) - np.log1p(-prevalence)) + float(np.linalg.norm(x, axis=1).max()) * beta_bound
    equality_correction = float(bias_bound * abs(dual.sum()))
    safe_dual = dual_value - equality_correction
    raw_gap = primal - safe_dual
    accepted = bool(np.isfinite([primal, dual_value, raw_gap, kkt, feasibility]).all()
                    and feasibility <= 1e-14 and domain_roundoff <= 4 * np.finfo(float).eps
                    and -1e-12 <= raw_gap <= GAP_TOLERANCE
                    and kkt <= KKT_TOLERANCE)
    return {"primal": primal, "dual": safe_dual, "uncorrected_dual": dual_value, "raw_gap": raw_gap,
            "dual_equality_correction": equality_correction, "optimal_intercept_abs_bound": float(bias_bound),
            "gap": max(0., raw_gap), "kkt_max": kkt, "dual_feasibility_max": feasibility,
            "dual_equality_residual": float(dual.sum()), "dual_domain_roundoff": domain_roundoff,
            "arithmetic": "float64; feasibility allowance1e-14; weak-duality allowance1e-12",
            "gap_tolerance": GAP_TOLERANCE, "kkt_tolerance": KKT_TOLERANCE,
            "accepted": accepted}


class CertifiedElasticNetClassifier(ClassifierMixin, BaseEstimator):
    """Named serializable classifier; only certified solutions are published."""

    def __init__(self, penalty: float = .5, l1_ratio: float = .3, max_iter: int = 2000,
                 tol: float = 1e-9):
        self.penalty, self.l1_ratio, self.max_iter, self.tol = penalty, l1_ratio, max_iter, tol

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> CertifiedElasticNetClassifier:
        """Optimize in float64, then certify the original weighted objective."""
        for name in ("coef_", "intercept_", "classes_", "n_features_in_", "n_iter_", "C", "fit_receipt_"):
            self.__dict__.pop(name, None)
        x = check_array(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64)
        weights = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if (y.shape != (len(x),) or not np.array_equal(np.unique(y), [0, 1])
                or weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0)
                or not np.isfinite(weights.sum()) or not np.isfinite(self.penalty) or self.penalty <= 0
                or not 0 <= self.l1_ratio < 1 or type(self.max_iter) is not int or self.max_iter < 1
                or not np.isfinite(self.tol) or self.tol <= 0):
            raise ValueError("Require both binary labels, positive finite weights/penalty and valid optimizer settings.")
        started = time.monotonic()
        rho = weights / weights.sum()
        prior = float(rho @ y)
        p = x.shape[1]
        theta = np.r_[np.zeros(2 * p), np.log(prior) - np.log1p(-prior)]
        result = minimize(loss_gradient, theta, args=(x, y, rho, self.penalty, self.l1_ratio),
                          method="L-BFGS-B", jac=True, bounds=[(0, None)] * (2 * p) + [(None, None)],
                          options={"gtol": self.tol, "ftol": 1e-15, "maxiter": self.max_iter,
                                   "maxls": 80, "maxcor": 30})
        beta = result.x[:p] - result.x[p:2 * p]
        base_logits = x @ beta
        # Solve the one-dimensional free-intercept optimum accurately. Brackets
        # keep all probabilities below/above the weighted class prevalence.
        bias = brentq(lambda b: float(rho @ _residual(base_logits + b, y)),
                      float(-np.max(base_logits) - 100), float(-np.min(base_logits) + 100), xtol=1e-13)
        proof = certificate(x, y, weights, beta, bias, self.penalty, self.l1_ratio)
        surplus = float(result.x[:2 * p].sum() - np.abs(beta).sum())
        accepted = bool(result.success and proof["accepted"]
                        and (self.l1_ratio == 0 or self.penalty * self.l1_ratio * surplus <= GAP_TOLERANCE))
        self.fit_receipt_ = {"backend": BACKEND, "certificate": proof, "split_surplus": surplus,
                             "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
                             "optimizer_message": str(result.message), "iterations": int(result.nit),
                             "seconds": time.monotonic() - started, "converged": accepted,
                             "options": {"gtol": self.tol, "ftol": 1e-15, "maxiter": self.max_iter, "maxls": 80, "maxcor": 30}}
        if not accepted:
            raise RuntimeError("Full-batch optimizer failed independent convergence certification.")
        self.coef_, self.intercept_ = beta[None, :], np.asarray([bias])
        self.classes_, self.n_features_in_, self.n_iter_ = np.asarray([0, 1]), p, np.asarray([result.nit])
        self.C = 1 / (self.penalty * weights.sum())
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return logits in the unchanged preprocessing coordinate system."""
        check_is_fitted(self, "coef_")
        x = check_array(X, dtype=np.float64)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from the fitted classifier.")
        return x @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities ordered [0, 1]."""
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the fixed probability threshold 0.5 for sklearn compatibility."""
        return (self.decision_function(X) >= 0).astype(int)
