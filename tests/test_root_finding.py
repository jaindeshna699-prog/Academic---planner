"""
Unit tests for Bisection and Newton-Raphson root-finding methods in numerical/root_finding.py.
Validates against scipy.optimize.bisect and scipy.optimize.newton.
"""

import pytest
import numpy as np
from scipy import optimize
from numerical.root_finding import (
    bisection_method,
    newton_raphson_method,
    solve_required_sgpa_bisection,
    benchmark_root_finding
)


def test_bisection_polynomial():
    # f(x) = x^2 - 4 on [1, 3] -> root = 2.0
    f = lambda x: x**2 - 4.0
    res = bisection_method(f, 1.0, 3.0, tol=1e-7)
    assert res["converged"] is True
    assert pytest.approx(res["root"], abs=1e-6) == 2.0
    assert len(res["history"]) > 0


def test_bisection_cgpa_target():
    # Current CGPA 8.0 with 80 credits, 20 upcoming credits, target 8.2
    # Total credits = 100. Target points = 820. Current points = 640. Needed = 180 / 20 = 9.0
    res = solve_required_sgpa_bisection(
        current_cgpa=8.0,
        completed_credits=80.0,
        upcoming_credits=20.0,
        target_cgpa=8.2,
        min_sgpa=0.0,
        max_sgpa=10.0,
        tol=1e-6
    )
    assert res["converged"] is True
    assert pytest.approx(res["root"], abs=1e-5) == 9.0


def test_bisection_invalid_bracket():
    f = lambda x: x**2 + 1.0  # strictly positive everywhere
    with pytest.raises(ValueError, match="bracket condition violated"):
        bisection_method(f, -2.0, 2.0)


def test_bisection_endpoints_order():
    f = lambda x: x - 2.0
    with pytest.raises(ValueError, match="strictly less than"):
        bisection_method(f, 3.0, 1.0)


def test_newton_raphson_cubic():
    # f(x) = x^3 - x - 2 -> root ~= 1.5213797
    f = lambda x: x**3 - x - 2.0
    df = lambda x: 3.0 * x**2 - 1.0
    res = newton_raphson_method(f, df=df, x0=2.0, tol=1e-7)
    assert res["converged"] is True
    scipy_root = optimize.newton(f, 2.0, fprime=df, tol=1e-7)
    assert pytest.approx(res["root"], abs=1e-6) == scipy_root


def test_newton_raphson_zero_derivative():
    f = lambda x: x**2 - 4.0
    df = lambda x: 2.0 * x
    # At x0 = 0, df(0) = 0
    with pytest.raises(ZeroDivisionError):
        newton_raphson_method(f, df=df, x0=0.0)


def test_benchmark_root_finding_agreement():
    f = lambda x: np.exp(x) - 3.0 * x  # root on [0, 1]
    res = benchmark_root_finding(f, 0.0, 1.0, tol=1e-6)
    assert res["bisect_difference"] < 1e-5
    assert res["newton_difference"] < 1e-5
