"""
Module: linear_systems.py
Implements Gauss Elimination and Gauss-Jordan Elimination from scratch with
partial pivoting and singular-matrix detection.
Solves resource allocation (daily study hours distribution) and target marks systems.
Includes step-by-step augmented matrix logging and NumPy/SciPy benchmarks.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from scipy import linalg


class SingularMatrixError(ValueError):
    """Raised when the matrix is singular or near-singular within tolerance."""
    pass


def gauss_elimination(
    A: Union[List[List[float]], np.ndarray],
    b: Union[List[float], np.ndarray],
    tol: float = 1e-12
) -> Dict[str, Any]:
    """
    Solves Ax = b using Gaussian Elimination with Partial Pivoting and Back Substitution from scratch.

    Method:
        Forward row elimination with row interchanges to upper triangular form,
        followed by backward substitution.
    Complexity:
        Time Complexity: O(n^3 / 3) flops.
        Space Complexity: O(n^2) for augmented matrix and step history.

    Parameters:
        A: Square coefficient matrix (n x n).
        b: Right-hand side vector (n).
        tol: Pivot singularity threshold.

    Returns:
        dict containing:
            - solution: Solution vector x.
            - augmented_matrix_final: Upper triangular augmented matrix.
            - step_history: Step-by-step state of augmented matrix.
    """
    A_mat = np.array(A, dtype=float, copy=True)
    b_vec = np.array(b, dtype=float, copy=True)
    n = A_mat.shape[0]

    if A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError(f"Matrix A must be square, received dimensions: {A_mat.shape}.")
    if len(b_vec) != n:
        raise ValueError(f"Dimension mismatch: A is {n}x{n} but b has length {len(b_vec)}.")

    # Form augmented matrix [A | b]
    Aug = np.hstack([A_mat, b_vec.reshape(-1, 1)])
    step_history: List[Dict[str, Any]] = []

    step_history.append({
        "stage": "Initial Augmented Matrix",
        "k": 0,
        "matrix": Aug.copy().tolist(),
        "description": "System formulation [A | b]"
    })

    # Forward Elimination with Partial Pivoting
    for k in range(n - 1):
        # Find pivot in column k at or below row k
        pivot_row = k + int(np.argmax(np.abs(Aug[k:, k])))
        max_val = abs(Aug[pivot_row, k])

        if max_val < tol:
            raise SingularMatrixError(
                f"Singular matrix detected at column {k}: maximum pivot |{max_val:.2e}| < tol ({tol}). "
                f"No unique solution exists."
            )

        # Swap rows if necessary
        if pivot_row != k:
            Aug[[k, pivot_row]] = Aug[[pivot_row, k]]
            step_history.append({
                "stage": f"Pivot Swap (Col {k+1})",
                "k": k + 1,
                "matrix": Aug.copy().tolist(),
                "description": f"Swapped Row {k+1} with Row {pivot_row+1} (pivot = {Aug[k, k]:.4f})"
            })

        # Eliminate rows below pivot
        for i in range(k + 1, n):
            factor = Aug[i, k] / Aug[k, k]
            Aug[i, k:] -= factor * Aug[k, k:]
            Aug[i, k] = 0.0  # Force exact zero

        step_history.append({
            "stage": f"Elimination below Col {k+1}",
            "k": k + 1,
            "matrix": Aug.copy().tolist(),
            "description": f"Eliminated column {k+1} below pivot"
        })

    # Check last pivot
    if abs(Aug[n - 1, n - 1]) < tol:
        raise SingularMatrixError(
            f"Singular matrix detected at final pivot: |A[{n-1},{n-1}]| = {abs(Aug[n-1, n-1]):.2e} < {tol}."
        )

    # Back Substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        sum_ax = np.dot(Aug[i, i + 1:n], x[i + 1:n])
        x[i] = (Aug[i, n] - sum_ax) / Aug[i, i]

    return {
        "solution": [float(val) for val in x],
        "augmented_matrix_final": Aug.tolist(),
        "step_history": step_history,
        "method": "Gauss Elimination with Partial Pivoting"
    }


def gauss_jordan_elimination(
    A: Union[List[List[float]], np.ndarray],
    b: Union[List[float], np.ndarray],
    tol: float = 1e-12
) -> Dict[str, Any]:
    """
    Solves Ax = b using Gauss-Jordan Elimination with Partial Pivoting from scratch.

    Method:
        Eliminates both below and above each pivot row to transform [A | b]
        directly into Reduced Row Echelon Form (RREF) [I | x].
    Complexity:
        Time Complexity: O(n^3 / 2) flops.
        Space Complexity: O(n^2).
    """
    A_mat = np.array(A, dtype=float, copy=True)
    b_vec = np.array(b, dtype=float, copy=True)
    n = A_mat.shape[0]

    if A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError(f"Matrix A must be square, received {A_mat.shape}.")
    if len(b_vec) != n:
        raise ValueError(f"Length mismatch: A is {n}x{n}, b has length {len(b_vec)}.")

    Aug = np.hstack([A_mat, b_vec.reshape(-1, 1)])
    step_history: List[Dict[str, Any]] = []

    for k in range(n):
        # Partial pivoting
        pivot_row = k + int(np.argmax(np.abs(Aug[k:, k])))
        max_val = abs(Aug[pivot_row, k])

        if max_val < tol:
            raise SingularMatrixError(f"Singular matrix detected: column {k} pivot magnitude < {tol}.")

        if pivot_row != k:
            Aug[[k, pivot_row]] = Aug[[pivot_row, k]]

        # Normalize pivot row
        pivot = Aug[k, k]
        Aug[k, :] = Aug[k, :] / pivot

        # Eliminate all other rows (both above and below)
        for i in range(n):
            if i != k:
                factor = Aug[i, k]
                Aug[i, :] -= factor * Aug[k, :]

        step_history.append({
            "stage": f"Jordan Step (Pivot {k+1})",
            "k": k + 1,
            "matrix": Aug.copy().tolist(),
            "description": f"Normalized row {k+1} and eliminated column {k+1} across all rows"
        })

    x = Aug[:, n]
    return {
        "solution": [float(val) for val in x],
        "augmented_matrix_rref": Aug.tolist(),
        "step_history": step_history,
        "method": "Gauss-Jordan Elimination"
    }


def solve_daily_study_hours_allocation(
    subject_names: List[str],
    credits_list: List[float],
    difficulty_weights: List[float],
    total_daily_hours: float
) -> Dict[str, Any]:
    """
    Constructs and solves a linear system to optimally allocate daily study hours across subjects.

    Constraints Formulation:
        1. \sum h_i = total_daily_hours (Total available time budget)
        2. Pairwise ratios proportional to (Credit * Difficulty):
           (Credit_i * Diff_i) * h_{i+1} - (Credit_{i+1} * Diff_{i+1}) * h_i = 0
    """
    n = len(subject_names)
    if n < 1:
        raise ValueError("At least 1 subject is required.")
    if n == 1:
        return {
            "allocation": {subject_names[0]: float(total_daily_hours)},
            "solution": [float(total_daily_hours)],
            "system_A": [[1.0]],
            "system_b": [float(total_daily_hours)]
        }

    # Weight per subject
    w = [c * d for c, d in zip(credits_list, difficulty_weights)]

    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    # Row 0: Sum of hours = total_daily_hours
    A[0, :] = 1.0
    b[0] = total_daily_hours

    # Rows 1 to n-1: w[i] * h_{i+1} - w[i+1] * h_i = 0  =>  h_i / w_i = h_{i+1} / w_{i+1}
    for i in range(n - 1):
        A[i + 1, i] = -w[i + 1]
        A[i + 1, i + 1] = w[i]
        b[i + 1] = 0.0

    res = gauss_elimination(A, b)
    sol = res["solution"]

    alloc_dict = {name: round(max(0.1, hours), 2) for name, hours in zip(subject_names, sol)}
    return {
        "allocation": alloc_dict,
        "solution": sol,
        "system_A": A.tolist(),
        "system_b": b.tolist(),
        "details": res
    }


def solve_mark_requirements_system(
    internal_weight: float = 0.30,
    midterm_weight: float = 0.20,
    endsem_weight: float = 0.50,
    target_overall: float = 85.0,
    known_internal: Optional[float] = None,
    known_midterm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Solves linear constraints to determine required marks across components.
    w_I * I + w_M * M + w_E * E = target_overall
    """
    if known_internal is not None and known_midterm is not None:
        rem = target_overall - (internal_weight * known_internal + midterm_weight * known_midterm)
        endsem_needed = rem / endsem_weight
        return {
            "internal": known_internal,
            "midterm": known_midterm,
            "endsem_needed": float(endsem_needed),
            "feasible": 0.0 <= endsem_needed <= 100.0
        }

    # Assume proportional ratio M = 0.8 * E, I = 0.9 * E if unknown
    A = [
        [internal_weight, midterm_weight, endsem_weight],
        [1.0, -1.125, 0.0],  # I / M ratio
        [0.0, 1.0, -0.80]    # M / E ratio
    ]
    b = [target_overall, 0.0, 0.0]
    res = gauss_jordan_elimination(A, b)
    sol = res["solution"]

    return {
        "internal_needed": round(sol[0], 2),
        "midterm_needed": round(sol[1], 2),
        "endsem_needed": round(sol[2], 2),
        "feasible": all(0.0 <= val <= 100.0 for val in sol),
        "details": res
    }


def benchmark_linear_system(
    A: List[List[float]],
    b: List[float]
) -> Dict[str, Any]:
    """
    Compares from-scratch Gauss and Gauss-Jordan against numpy.linalg.solve.
    """
    scratch_gauss = gauss_elimination(A, b)
    scratch_jordan = gauss_jordan_elimination(A, b)

    try:
        np_sol = np.linalg.solve(A, b)
        np_sol_list = [float(val) for val in np_sol]
        np_status = "Success"
        diff_gauss = [abs(s - n_s) for s, n_s in zip(scratch_gauss["solution"], np_sol_list)]
        diff_jordan = [abs(s - n_s) for s, n_s in zip(scratch_jordan["solution"], np_sol_list)]
        max_diff = max(max(diff_gauss), max(diff_jordan))
    except Exception as e:
        np_sol_list = None
        np_status = f"NumPy Error: {str(e)}"
        max_diff = None

    return {
        "scratch_gauss_solution": scratch_gauss["solution"],
        "scratch_jordan_solution": scratch_jordan["solution"],
        "numpy_solution": np_sol_list,
        "max_difference": max_diff,
        "numpy_status": np_status,
        "step_history": scratch_gauss["step_history"]
    }
