"""
Unit tests for Lagrange, Newton Divided Difference, and Hermite Interpolation in:
  - numerical/interpolation.py
  - numerical/lagrange.py
  - numerical/hermite.py
Validates against scipy.interpolate.lagrange and scipy.interpolate.KroghInterpolator.
"""

import pytest
import numpy as np
from scipy import interpolate
from numerical.lagrange import (
    lagrange_interpolation,
    predict_trajectory_lagrange,
    benchmark_lagrange
)
from numerical.hermite import (
    hermite_interpolation,
    compare_lagrange_vs_hermite,
    benchmark_hermite
)
from numerical.interpolation import (
    newton_polynomial_interpolation,
    benchmark_interpolation
)


def test_lagrange_quadratic():
    # Points on parabola y = x^2 - x + 2
    # x = [1, 2, 3] => y = [2, 4, 8]
    x_nodes = [1.0, 2.0, 3.0]
    y_nodes = [2.0, 4.0, 8.0]
    # At x = 4, y should be 4^2 - 4 + 2 = 14
    res = lagrange_interpolation(x_nodes, y_nodes, x_eval=4.0)
    assert pytest.approx(res["y_pred"], abs=1e-6) == 14.0
    assert res["polynomial_degree"] == 2


def test_lagrange_duplicate_node_error():
    with pytest.raises(ValueError, match="Duplicate nodes detected"):
        lagrange_interpolation([1.0, 2.0, 2.0], [5.0, 6.0, 7.0], x_eval=3.0)


def test_lagrange_scipy_agreement():
    x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_nodes = [8.2, 8.5, 8.1, 8.8, 9.0]
    bench = benchmark_lagrange(x_nodes, y_nodes, x_eval=6.0)
    assert bench["scipy_status"] == "Success"
    assert bench["difference"] < 1e-6


def test_newton_divided_difference_quadratic():
    # y = x^2 - x + 2, at x = 4 => 14
    x_nodes = [1.0, 2.0, 3.0]
    y_nodes = [2.0, 4.0, 8.0]
    res = newton_polynomial_interpolation(x_nodes, y_nodes, x_eval=4.0)
    assert pytest.approx(res["y_pred"], abs=1e-6) == 14.0
    assert len(res["coefficients"]) == 3
    # Check tableau length
    assert len(res["divided_diff_table"]) == 3


def test_newton_duplicate_nodes():
    with pytest.raises(ValueError, match="Duplicate nodes"):
        newton_polynomial_interpolation([1.0, 1.0, 2.0], [3.0, 4.0, 5.0], x_eval=3.0)


def test_newton_and_lagrange_exact_agreement():
    # Both algorithms must yield identical polynomial predictions
    x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_nodes = [7.8, 8.2, 8.5, 8.1, 8.9]
    bench = benchmark_interpolation(x_nodes, y_nodes, x_eval=6.0)
    assert bench["internal_discrepancy"] < 1e-12
    assert bench["diff_lagrange_vs_scipy"] < 1e-6
    assert bench["diff_newton_vs_scipy"] < 1e-6


def test_hermite_cubic():
    # y = x^3 => y' = 3x^2
    # Nodes x = [0, 1] => y = [0, 1], slopes = [0, 3]
    # At x = 0.5: y = 0.125
    x_nodes = [0.0, 1.0]
    y_nodes = [0.0, 1.0]
    d_nodes = [0.0, 3.0]
    res = hermite_interpolation(x_nodes, y_nodes, d_nodes=d_nodes, x_eval=0.5)
    assert pytest.approx(res["y_pred"], abs=1e-6) == 0.125
    assert res["polynomial_degree"] == 3


def test_hermite_scipy_spline_agreement():
    x_nodes = [1.0, 2.0, 3.0, 4.0]
    y_nodes = [7.5, 8.0, 8.4, 8.9]
    d_nodes = [0.5, 0.45, 0.5, 0.4]
    bench = benchmark_hermite(x_nodes, y_nodes, x_eval=2.5, d_nodes=d_nodes)
    assert bench["scipy_status"] == "Success"
    assert bench["difference"] < 1e-5


def test_lagrange_vs_hermite_comparison():
    x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_nodes = [8.0, 8.2, 8.5, 8.6, 9.0]
    comp = compare_lagrange_vs_hermite(x_nodes, y_nodes, x_eval=6.0)
    assert "lagrange_prediction" in comp
    assert "hermite_prediction" in comp
    assert comp["absolute_difference"] >= 0.0
