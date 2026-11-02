"""
Unit tests for Fixed-Point Iteration in numerical/fixed_point.py.
Validates contraction mapping, study hours model, and scipy.optimize.fixed_point.
"""

import pytest
import numpy as np
from scipy import optimize
from numerical.fixed_point import (
    fixed_point_iteration,
    study_hours_model_g,
    benchmark_fixed_point
)


def test_fixed_point_cosine():
    # x = cos(x), known Dottie number ~= 0.7390851332
    g = lambda x: float(np.cos(x))
    res = fixed_point_iteration(g, x0=0.5, tol=1e-7)
    assert res["converged"] is True
    scipy_root = float(optimize.fixed_point(g, 0.5, xtol=1e-7))
    assert pytest.approx(res["root"], abs=1e-5) == scipy_root
    assert pytest.approx(res["root"], abs=1e-5) == 0.739085


def test_study_hours_fixed_point_model():
    # Target 80 marks out of 100 with alpha=0.25
    g = study_hours_model_g(target_marks=80.0, alpha=0.25, max_marks=100.0)
    res = fixed_point_iteration(g, x0=3.0, tol=1e-5)
    assert res["converged"] is True
    # Verify M(x) = 100*(1 - exp(-0.25*x)) ~= 80
    score = 100.0 * (1.0 - np.exp(-0.25 * res["root"]))
    assert pytest.approx(score, abs=0.5) == 80.0


def test_study_hours_target_exceeds_max():
    with pytest.raises(ValueError, match="strictly less than"):
        study_hours_model_g(target_marks=105.0, max_marks=100.0)


def test_divergence_detection():
    # g(x) = 2x + 1 is divergent (|g'(x)| = 2 > 1)
    g = lambda x: 2.0 * x + 1.0
    res = fixed_point_iteration(g, x0=1.0, max_iter=25)
    assert res["converged"] is False
    assert "Divergence" in res.get("error_reason", "") or "Failed to converge" in res.get("error_reason", "")


def test_benchmark_fixed_point_agreement():
    g = lambda x: (x + 2.0 / x) / 2.0  # fixed point is sqrt(2) ~= 1.41421356
    bench = benchmark_fixed_point(g, x0=1.0, tol=1e-6)
    assert bench["difference"] < 1e-5
    assert pytest.approx(bench["scratch_result"]["root"], abs=1e-5) == np.sqrt(2.0)
