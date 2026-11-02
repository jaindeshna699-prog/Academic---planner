"""
Unit tests for Linear and Polynomial Least Squares Regression in numerical/regression.py.
Validates normal equations, residuals, R^2 score, RMSE, and numpy.polyfit benchmarks.
"""

import pytest
import numpy as np
from numerical.regression import (
    linear_regression,
    polynomial_regression,
    predict_regression,
    benchmark_regression
)


def test_linear_regression_exact_line():
    # y = 3x - 1
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 5.0, 8.0, 11.0, 14.0]
    res = linear_regression(x, y)
    assert pytest.approx(res["slope"], abs=1e-6) == 3.0
    assert pytest.approx(res["intercept"], abs=1e-6) == -1.0
    assert pytest.approx(res["r_squared"], abs=1e-6) == 1.0
    assert pytest.approx(res["sse"], abs=1e-6) == 0.0


def test_polynomial_regression_quadratic():
    # y = 2x^2 + 5x + 3
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [3.0, 10.0, 21.0, 36.0, 55.0]
    res = polynomial_regression(x, y, degree=2)
    assert pytest.approx(res["coefficients"][0], abs=1e-5) == 2.0
    assert pytest.approx(res["coefficients"][1], abs=1e-5) == 5.0
    assert pytest.approx(res["coefficients"][2], abs=1e-5) == 3.0
    assert pytest.approx(res["r_squared"], abs=1e-5) == 1.0


def test_regression_prediction_and_residuals():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [4.9, 7.1, 9.2, 10.9, 13.1]
    res = linear_regression(x, y)
    assert len(res["residual_table"]) == 5
    # Predict for x = 6
    pred_6 = predict_regression(res, 6.0)
    assert 14.5 <= pred_6 <= 15.5


def test_regression_insufficient_points():
    with pytest.raises(ValueError, match="Insufficient data points"):
        polynomial_regression([1.0, 2.0], [3.0, 4.0], degree=2)


def test_regression_zero_variance():
    with pytest.raises(ValueError, match="Zero variance in x_data"):
        linear_regression([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])


def test_benchmark_regression_numpy_agreement():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [2.2, 3.8, 6.1, 8.0, 10.1, 11.9]
    bench = benchmark_regression(x, y, degree=1)
    assert bench["numpy_status"] == "Success"
    assert bench["max_difference"] < 1e-6
