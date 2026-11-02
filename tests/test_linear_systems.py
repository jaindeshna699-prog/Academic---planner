"""
Unit tests for Gauss Elimination and Gauss-Jordan in numerical/linear_systems.py.
Validates partial pivoting, singular matrix exceptions, resource allocation, and numpy.linalg.solve.
"""

import pytest
import numpy as np
from numerical.linear_systems import (
    gauss_elimination,
    gauss_jordan_elimination,
    solve_daily_study_hours_allocation,
    solve_mark_requirements_system,
    benchmark_linear_system,
    SingularMatrixError
)


def test_gauss_elimination_3x3():
    # 2x + y - z = 8
    # -3x - y + 2z = -11
    # -2x + y + 2z = -3
    # Solution: x = 2, y = 3, z = -1
    A = [
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0]
    ]
    b = [8.0, -11.0, -3.0]
    res = gauss_elimination(A, b)
    assert pytest.approx(res["solution"], abs=1e-6) == [2.0, 3.0, -1.0]


def test_gauss_jordan_elimination_3x3():
    A = [
        [1.0, 2.0, 1.0],
        [3.0, 8.0, 1.0],
        [0.0, 4.0, 1.0]
    ]
    b = [2.0, 12.0, 2.0]
    res = gauss_jordan_elimination(A, b)
    np_sol = np.linalg.solve(A, b)
    assert pytest.approx(res["solution"], abs=1e-6) == [float(x) for x in np_sol]


def test_singular_matrix_detection():
    # Linearly dependent rows (Row 2 = 2 * Row 1)
    A = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 1.0]
    ]
    b = [1.0, 2.0, 3.0]
    with pytest.raises(SingularMatrixError):
        gauss_elimination(A, b)


def test_daily_study_hours_allocation():
    subjects = ["Numerical Methods", "Cloud Computing", "Cryptography"]
    credits_list = [4.0, 3.0, 3.0]
    diff = [5.0, 3.0, 4.0]  # weights: 20, 9, 12. Sum = 41
    total_hours = 6.0
    res = solve_daily_study_hours_allocation(subjects, credits_list, diff, total_hours)
    sol = res["solution"]
    assert pytest.approx(sum(sol), abs=1e-5) == total_hours
    # Numerical Methods should get the most hours
    assert sol[0] > sol[1]
    assert sol[0] > sol[2]


def test_mark_requirements_system():
    # 30% Internal (known 28/30), 20% Midterm (known 16/20), 50% Endsem for target 85
    # Total so far = 28 + 16 = 44. Remaining for 85 = 41. 41 / 0.5 = 82 / 100
    res = solve_mark_requirements_system(
        internal_weight=0.30,
        midterm_weight=0.20,
        endsem_weight=0.50,
        target_overall=85.0,
        known_internal=28.0 / 0.30, # marks out of 100
        known_midterm=16.0 / 0.20
    )
    assert res["feasible"] is True
    assert pytest.approx(res["endsem_needed"], abs=0.1) == 82.0


def test_benchmark_linear_system_agreement():
    A = [
        [4.0, 1.0, 2.0],
        [2.0, 5.0, 1.0],
        [1.0, 1.0, 6.0]
    ]
    b = [16.0, 19.0, 25.0]
    bench = benchmark_linear_system(A, b)
    assert bench["numpy_status"] == "Success"
    assert bench["max_difference"] < 1e-10
