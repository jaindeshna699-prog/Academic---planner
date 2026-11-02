"""
Module: interpolation.py
Implements Lagrange and Newton Divided Difference Polynomial Interpolation from scratch,
along with Hermite Interpolation using confluent divided differences.
Includes basis polynomial diagnostics, divided difference tableaux, and SciPy benchmarks.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from scipy import interpolate


# ---------------------------------------------------------------------------
# 1. Lagrange Polynomial Interpolation
# ---------------------------------------------------------------------------
def lagrange_interpolation(
    x_nodes: Union[List[float], np.ndarray],
    y_nodes: Union[List[float], np.ndarray],
    x_eval: float
) -> Dict[str, Any]:
    r"""
    Evaluates the unique degree-n Lagrange interpolating polynomial P_n(x_eval) from scratch.

    Method:
        Lagrange Form of Polynomial Interpolation.
    Formula:
        P_n(x) = \sum_{i=0}^n y_i \cdot \ell_i(x)
        where \ell_i(x) = \prod_{j \neq i}^n \frac{x - x_j}{x_i - x_j}
    Complexity:
        Time Complexity: O(n^2) where n is polynomial degree (number of nodes - 1).
        Space Complexity: O(n) for basis terms.

    Parameters:
        x_nodes: Known semester or time nodes [x_0, ..., x_n].
        y_nodes: Observed values [y_0, ..., y_n] (e.g. SGPAs).
        x_eval: Target evaluation point.

    Returns:
        dict containing y_pred, polynomial_degree, and basis_details.
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
        "basis_details": basis_details,
        "method": "Lagrange Polynomial Interpolation"
    }


# ---------------------------------------------------------------------------
# 2. Newton Divided Difference Polynomial Interpolation
# ---------------------------------------------------------------------------
def newton_polynomial_interpolation(
    x_nodes: Union[List[float], np.ndarray],
    y_nodes: Union[List[float], np.ndarray],
    x_eval: float
) -> Dict[str, Any]:
    r"""
    Evaluates the interpolating polynomial using Newton's Divided Difference formula from scratch.

    Method:
        Newton's Divided Difference Form.
    Formula:
        P_n(x) = f[x_0] + f[x_0, x_1](x - x_0) + f[x_0, x_1, x_2](x - x_0)(x - x_1) + ...
        where f[x_i, ..., x_{i+k}] = \frac{f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}]}{x_{i+k} - x_i}
    Complexity:
        Time Complexity: O(n^2) for divided difference table construction and evaluation.
        Space Complexity: O(n^2) for the full difference tableau.

    Parameters:
        x_nodes: Known nodes [x_0, ..., x_n].
        y_nodes: Function values [y_0, ..., y_n].
        x_eval: Target evaluation point.

    Returns:
        dict containing:
            - y_pred: Evaluated polynomial value at x_eval.
            - divided_diff_table: Step-by-step divided difference tableau.
            - coefficients: Newton polynomial coefficients [a_0, a_1, ..., a_n].
            - basis_terms: Evaluated Newton basis products.
    """
    x = [float(val) for val in x_nodes]
    y = [float(val) for val in y_nodes]
    n = len(x)

    if n < 2:
        raise ValueError(f"Newton interpolation requires at least 2 distinct points, received {n}.")
    if len(y) != n:
        raise ValueError(f"Mismatch: len(x_nodes)={n} != len(y_nodes)={len(y)}.")

    # Distinct check
    for i in range(n):
        for j in range(i + 1, n):
            if abs(x[i] - x[j]) < 1e-12:
                raise ValueError(f"Duplicate nodes: x[{i}]={x[i]} and x[{j}]={x[j]}. Division by zero.")

    # Construct divided difference table
    F = np.zeros((n, n), dtype=float)
    for i in range(n):
        F[i, 0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            denom = x[i + j] - x[i]
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / denom

    # Coefficients a_k = F[0, k]
    coeffs = [float(F[0, k]) for k in range(n)]

    # Evaluate at x_eval using Horner-like product: P(x) = a_0 + \sum_{k=1}^n a_k \prod_{j=0}^{k-1} (x - x_j)
    y_pred = coeffs[0]
    basis_product = 1.0
    basis_terms = [{"k": 0, "coeff": coeffs[0], "product": 1.0, "term_value": coeffs[0]}]

    for k in range(1, n):
        basis_product *= (x_eval - x[k - 1])
        term_val = coeffs[k] * basis_product
        y_pred += term_val
        basis_terms.append({
            "k": k,
            "coeff": coeffs[k],
            "product": float(basis_product),
            "term_value": float(term_val)
        })

    # Prepare readable tableau
    tableau_rows = []
    for i in range(n):
        row_dict: Dict[str, Any] = {"i": i, "x_i": x[i], "f[x_i]": float(F[i, 0])}
        for j in range(1, n - i):
            row_dict[f"Order_{j}"] = float(F[i, j])
        tableau_rows.append(row_dict)

    return {
        "y_pred": float(y_pred),
        "polynomial_degree": n - 1,
        "x_eval": float(x_eval),
        "coefficients": coeffs,
        "divided_diff_table": tableau_rows,
        "basis_terms": basis_terms,
        "method": "Newton Divided Difference Polynomial Interpolation"
    }


# ---------------------------------------------------------------------------
# 3. Hermite Confluent Divided Difference Interpolation
# ---------------------------------------------------------------------------
def hermite_interpolation(
    x_nodes: Union[List[float], np.ndarray],
    y_nodes: Union[List[float], np.ndarray],
    d_nodes: Optional[Union[List[float], np.ndarray]] = None,
    x_eval: float = 6.0
) -> Dict[str, Any]:
    r"""
    Evaluates the Hermite interpolating polynomial H(x) at x_eval from scratch
    incorporating trend velocities f'(x_i).
    """
    from .hermite import hermite_interpolation as _hermite_impl
    return _hermite_impl(x_nodes, y_nodes, d_nodes=d_nodes, x_eval=x_eval)


# ---------------------------------------------------------------------------
# 4. SciPy Benchmarking
# ---------------------------------------------------------------------------
def benchmark_interpolation(
    x_nodes: List[float],
    y_nodes: List[float],
    x_eval: float
) -> Dict[str, Any]:
    """
    Benchmarks from-scratch Lagrange and Newton interpolation against scipy.interpolate.
    Since Lagrange and Newton produce the exact same unique interpolating polynomial,
    they must agree to within machine precision.
    """
    scratch_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_eval)
    scratch_newton = newton_polynomial_interpolation(x_nodes, y_nodes, x_eval)

    # SciPy KroghInterpolator / BarycentricInterpolator
    try:
        krogh = interpolate.KroghInterpolator(x_nodes, y_nodes)
        scipy_val = float(krogh(x_eval))
        scipy_status = "Success"
        diff_lagrange = abs(scratch_lagrange["y_pred"] - scipy_val)
        diff_newton = abs(scratch_newton["y_pred"] - scipy_val)
    except Exception as e:
        scipy_val = None
        scipy_status = f"SciPy Error: {str(e)}"
        diff_lagrange = None
        diff_newton = None

    internal_discrepancy = abs(scratch_lagrange["y_pred"] - scratch_newton["y_pred"])

    return {
        "scratch_lagrange": scratch_lagrange["y_pred"],
        "scratch_newton": scratch_newton["y_pred"],
        "scipy_result": scipy_val,
        "internal_discrepancy": internal_discrepancy,
        "diff_lagrange_vs_scipy": diff_lagrange,
        "diff_newton_vs_scipy": diff_newton,
        "scipy_status": scipy_status
    }
