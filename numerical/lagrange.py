"""
Module: lagrange.py
Implements Lagrange Polynomial Interpolation from scratch for predicting
missing or future semester SGPAs from historical academic trajectories.
Includes basis polynomial diagnostics, step tables, and SciPy benchmarking.
"""

from typing import List, Dict, Any, Tuple, Union
import numpy as np
from scipy import interpolate


def lagrange_interpolation(
    x_nodes: Union[List[float], np.ndarray],
    y_nodes: Union[List[float], np.ndarray],
    x_eval: float
) -> Dict[str, Any]:
    """
    Evaluates the unique degree-n Lagrange interpolating polynomial P_n(x_eval) from scratch.

    Method:
        Lagrange Form of Polynomial Interpolation.
    Formula:
        P_n(x) = \sum_{i=0}^n y_i \cdot \ell_i(x)
        where \ell_i(x) = \prod_{j \neq i}^n \frac{x - x_j}{x_i - x_j}
    Complexity:
        Time Complexity: O(n^2) where n is the polynomial degree.
        Space Complexity: O(n) for basis polynomial evaluation.

    Parameters:
        x_nodes: Known semester indices (e.g. [1, 2, 3, 4, 5]).
        y_nodes: Observed SGPAs at corresponding semesters.
        x_eval: Target semester number to predict (e.g. 6.0).

    Returns:
        dict containing:
            - y_pred: Evaluated SGPA estimate at x_eval.
            - polynomial_degree: Degree n = len(x_nodes) - 1.
            - basis_weights: List of dicts detailing each basis \ell_i(x_eval) and its contribution.
    """
    x = [float(val) for val in x_nodes]
    y = [float(val) for val in y_nodes]
    n = len(x)

    if n < 2:
        raise ValueError(f"Lagrange interpolation requires at least 2 distinct data points, received {n}.")
    if len(y) != n:
        raise ValueError(f"Length mismatch: x_nodes ({n}) and y_nodes ({len(y)}) must have identical dimensions.")

    # Check for duplicate nodes
    for i in range(n):
        for j in range(i + 1, n):
            if abs(x[i] - x[j]) < 1e-12:
                raise ValueError(f"Duplicate nodes detected: x[{i}] = {x[i]} and x[{j}] = {x[j]}. Division by zero.")

    basis_details: List[Dict[str, Any]] = []
    y_pred = 0.0

    for i in range(n):
        ell_i = 1.0
        numerator_terms = []
        denominator_terms = []

        for j in range(n):
            if j != i:
                num = x_eval - x[j]
                den = x[i] - x[j]
                ell_i *= (num / den)
                numerator_terms.append(f"({x_eval} - {x[j]})")
                denominator_terms.append(f"({x[i]} - {x[j]})")

        contribution = y[i] * ell_i
        y_pred += contribution

        basis_details.append({
            "i": i,
            "x_i": x[i],
            "y_i": y[i],
            "basis_value_ell_i": float(ell_i),
            "contribution": float(contribution),
            "expression": f"L_{i}(x) = [{' * '.join(numerator_terms)}] / [{' * '.join(denominator_terms)}]"
        })

    return {
        "y_pred": float(y_pred),
        "polynomial_degree": n - 1,
        "x_eval": float(x_eval),
        "basis_details": basis_details
    }


def predict_trajectory_lagrange(
    semesters: List[int],
    sgpas: List[float],
    future_semesters: List[int]
) -> List[Dict[str, Any]]:
    """Generates predictions across multiple upcoming semesters using Lagrange polynomial."""
    results = []
    for sem in future_semesters:
        res = lagrange_interpolation(semesters, sgpas, float(sem))
        results.append({
            "semester": sem,
            "predicted_sgpa": round(res["y_pred"], 3),
            "details": res
        })
    return results


def benchmark_lagrange(
    x_nodes: List[float],
    y_nodes: List[float],
    x_eval: float
) -> Dict[str, Any]:
    """
    Compares from-scratch Lagrange interpolation against scipy.interpolate.lagrange.
    """
    scratch_res = lagrange_interpolation(x_nodes, y_nodes, x_eval)

    try:
        scipy_poly = interpolate.lagrange(np.array(x_nodes), np.array(y_nodes))
        scipy_val = float(scipy_poly(x_eval))
        scipy_status = "Success"
        difference = abs(scratch_res["y_pred"] - scipy_val)
    except Exception as e:
        scipy_val = None
        scipy_status = f"SciPy Error: {str(e)}"
        difference = None

    return {
        "scratch_result": scratch_res["y_pred"],
        "scipy_result": scipy_val,
        "difference": difference,
        "scipy_status": scipy_status,
        "polynomial_degree": scratch_res["polynomial_degree"],
        "basis_details": scratch_res["basis_details"]
    }
