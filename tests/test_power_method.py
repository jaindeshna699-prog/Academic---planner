"""
Unit tests for Power Method in numerical/power_method.py.
Validates dominant eigenvalue/vector computation, subject priority matrix, and numpy.linalg.eig.
"""

import pytest
import numpy as np
from numerical.power_method import (
    power_method,
    build_subject_priority_matrix,
    benchmark_power_method
)


def test_power_method_symmetric_matrix():
    # Matrix with known eigenvalues
    # A = [[2, 1], [1, 2]], eigenvalues: 3 and 1. Dominant = 3, eigenvector = [1/sqrt(2), 1/sqrt(2)]
    A = [
        [2.0, 1.0],
        [1.0, 2.0]
    ]
    res = power_method(A, tol=1e-7)
    assert res["converged"] is True
    assert pytest.approx(res["eigenvalue"], abs=1e-5) == 3.0
    # Both components of dominant eigenvector should be equal
    assert pytest.approx(res["eigenvector"][0], abs=1e-5) == res["eigenvector"][1]


def test_subject_priority_ranking():
    # Subject A: 4 credits, strength 1 (weakest = highest diff), exam in 3 days, 80% pending
    # Subject B: 2 credits, strength 5 (mastered = lowest diff), exam in 30 days, 20% pending
    res = build_subject_priority_matrix(
        subject_names=["Hard Course", "Easy Course"],
        credits_list=[4.0, 2.0],
        strength_ratings=[1, 5],
        days_to_exam=[3, 30],
        syllabus_pending_pct=[80.0, 20.0]
    )
    ranking = res["ranking"]
    assert ranking[0][0] == "Hard Course"
    assert ranking[0][1] > ranking[1][1]


def test_power_method_zero_vector_error():
    A = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="cannot be the zero vector"):
        power_method(A, v0=[0.0, 0.0])


def test_benchmark_power_method_numpy_agreement():
    A = [
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 2.0],
        [1.0, 2.0, 5.0]
    ]
    bench = benchmark_power_method(A)
    assert bench["numpy_status"] == "Success"
    assert bench["eigenvalue_difference"] < 1e-5
    assert bench["max_weight_difference"] < 1e-4
