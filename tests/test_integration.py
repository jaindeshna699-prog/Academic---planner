"""
Unit tests for Composite Trapezoidal Rule in numerical/numerical_integration.py.
Validates exact polynomial integration, discrete study hours integration, and scipy.integrate.trapezoid.
"""

import pytest
import numpy as np
from scipy import integrate
from numerical.numerical_integration import (
    trapezoidal_rule,
    compute_cumulative_study_effort,
    benchmark_trapezoid
)


def test_trapezoid_linear_exact():
    # Integral of f(x) = 2x + 1 from 0 to 4:
    # [x^2 + x]_0^4 = 16 + 4 = 20. Trapezoidal rule is exact for linear polynomials!
    f = lambda x: 2.0 * x + 1.0
    res = trapezoidal_rule(f, a=0.0, b=4.0, n_intervals=10)
    assert pytest.approx(res["total_integral"], abs=1e-6) == 20.0
    assert pytest.approx(res["average_height"], abs=1e-6) == 5.0


def test_trapezoid_discrete_study_effort():
    # Daily study hours across a week
    daily_hours = [3.0, 4.0, 2.5, 5.0, 4.5, 6.0, 5.0]
    days = [1, 2, 3, 4, 5, 6, 7]
    res = compute_cumulative_study_effort(daily_hours, days)

    scipy_val = float(integrate.trapezoid(daily_hours, x=days))
    assert pytest.approx(res["total_integral"], abs=1e-6) == scipy_val
    assert len(res["interval_table"]) == 6


def test_trapezoid_invalid_limits():
    f = lambda x: x**2
    with pytest.raises(ValueError, match="strictly less than"):
        trapezoidal_rule(f, a=5.0, b=2.0)


def test_trapezoid_non_monotonic_x():
    with pytest.raises(ValueError, match="strictly increasing"):
        trapezoidal_rule([2.0, 3.0, 4.0], x_nodes=[1.0, 3.0, 2.0])


def test_benchmark_trapezoid_agreement():
    y = [1.5, 2.0, 3.5, 4.0, 3.0, 5.0]
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    bench = benchmark_trapezoid(y, x_values=x)
    assert bench["scipy_status"] == "Success"
    assert bench["difference"] < 1e-12
