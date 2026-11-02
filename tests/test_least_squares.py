"""
Unit tests for Least Squares Regression in numerical/least_squares.py.
Validates linear/polynomial fits, normal equations, residuals, R^2, and numpy.polyfit.
"""

import pytest
import numpy as np
from numerical.least_squares import (
    least_squares_fit,
    predict_from_least_squares,
    benchmark_least_squares
)


def test_least_squares_exact_line():
    # y = 2x + 3
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [5.0, 7.0, 9.0, 11.0, 13.0]
    res = least_squares_fit(x, y, degree=1)
    assert pytest.approx(res["slope"], abs=1e-6) == 2.0
    assert pytest.approx(res["intercept"], abs=1e-6) == 3.0
    assert pytest.approx(res["r_squared"], abs=1e-6) == 1.0
    assert pytest.approx(res["sse"], abs=1e-6) == 0.0


def test_least_squares_noisy_study_hours_vs_marks():
    hours = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    marks = [50.0, 62.0, 71.0, 78.0, 88.0, 94.0]
    res = least_squares_fit(hours, marks, degree=1)
    assert res["slope"] > 0  # positive correlation
    assert res["r_squared"] > 0.95
    assert len(res["residual_table"]) == len(hours)

    # Prediction for 4.5 hours
    pred_4_5 = predict_from_least_squares(res, 4.5)
    assert 80.0 <= pred_4_5 <= 86.0


def test_least_squares_quadratic_fit():
    # y = 3x^2 - 2x + 1
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 9.0, 22.0, 41.0]
    res = least_squares_fit(x, y, degree=2)
    assert pytest.approx(res["coefficients"][0], abs=1e-5) == 3.0
    assert pytest.approx(res["coefficients"][1], abs=1e-5) == -2.0
    assert pytest.approx(res["coefficients"][2], abs=1e-5) == 1.0
    assert pytest.approx(res["r_squared"], abs=1e-5) == 1.0


def test_least_squares_insufficient_points():
    with pytest.raises(ValueError, match="Insufficient data points"):
        least_squares_fit([1.0, 2.0], [3.0, 4.0], degree=2)


def test_least_squares_zero_variance():
    with pytest.raises(ValueError, match="Zero variance in x_data"):
        least_squares_fit([2.0, 2.0, 2.0], [5.0, 6.0, 7.0], degree=1)


def test_benchmark_least_squares_agreement():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [2.1, 3.9, 6.2, 8.1, 9.8, 12.1]
    bench = benchmark_least_squares(x, y, degree=1)
    assert bench["numpy_status"] == "Success"
    assert bench["max_difference"] < 1e-6
