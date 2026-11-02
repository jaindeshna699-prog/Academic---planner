"""
Module: hermite.py
Implements Hermite Interpolation using Newton's Divided Differences on doubled nodes
to predict SGPA trajectories by incorporating improvement rates (derivatives f'(x)).
Includes divided difference tables, comparative analysis against Lagrange, and SciPy benchmarking.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from scipy import interpolate


def hermite_interpolation(
    x_nodes: Union[List[float], np.ndarray],
    y_nodes: Union[List[float], np.ndarray],
    d_nodes: Optional[Union[List[float], np.ndarray]] = None,
    x_eval: float = 6.0
) -> Dict[str, Any]:
    r"""
    Evaluates the Hermite interpolating polynomial H(x) at x_eval from scratch.

    Method:
        Newton's Divided Difference algorithm with doubled confluent nodes.
    Formula:
        Given nodes x_i, values y_i, and slopes y'_i for i = 0, ..., n:
        Doubled array: z = [x_0, x_0, x_1, x_1, ..., x_n, x_n]
        Divided differences:
            Q[2i, 0] = y_i,  Q[2i+1, 0] = y_i
            Q[2i+1, 1] = y'_i (since z_{2i+1} = z_{2i})
            For j >= 1: Q[i, j] = (Q[i, j-1] - Q[i-1, j-1]) / (z_i - z_{i-j})
        Evaluation:
            H(x) = Q[0,0] + \sum_{k=1}^{2n+1} Q[k,k] \prod_{j=0}^{k-1} (x - z_j)
    Complexity:
        Time Complexity: O(N^2) where N = 2(n + 1).
        Space Complexity: O(N^2) for the divided difference tableau.

    Parameters:
        x_nodes: Semester indices [x_0, ..., x_n].
        y_nodes: SGPAs [y_0, ..., y_n].
        d_nodes: Slopes/Improvement rates [y'_0, ..., y'_n].
                 If None, estimated via central/forward/backward differences.
        x_eval: Target evaluation point.

    Returns:
        dict containing:
            - y_pred: Evaluated Hermite prediction at x_eval.
            - polynomial_degree: Degree 2n + 1.
            - divided_diff_table: List of row dictionaries for visualization.
            - basis_terms: List of Newton basis product values.
    """
    x = [float(val) for val in x_nodes]
    y = [float(val) for val in y_nodes]
    n = len(x)

    if n < 2:
        raise ValueError(f"Hermite interpolation requires at least 2 distinct points, received {n}.")
    if len(y) != n:
        raise ValueError(f"Mismatch: len(x_nodes)={n} != len(y_nodes)={len(y)}.")

    # Distinct check
    for i in range(n):
        for j in range(i + 1, n):
            if abs(x[i] - x[j]) < 1e-12:
                raise ValueError(f"Duplicate nodes: x[{i}]={x[i]} and x[{j}]={x[j]}.")

    # If slopes not provided, approximate via finite differences
    if d_nodes is None:
        d = []
        for i in range(n):
            if i == 0:
                slope = (y[1] - y[0]) / (x[1] - x[0])
            elif i == n - 1:
                slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            else:
                # Central difference
                slope = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
            d.append(float(slope))
    else:
        d = [float(val) for val in d_nodes]
        if len(d) != n:
            raise ValueError(f"Mismatch: len(d_nodes)={len(d)} != len(x_nodes)={n}.")

    # Double the nodes
    m = 2 * n
    z = [0.0] * m
    Q = np.zeros((m, m), dtype=float)

    for i in range(n):
        z[2 * i] = x[i]
        z[2 * i + 1] = x[i]
        Q[2 * i, 0] = y[i]
        Q[2 * i + 1, 0] = y[i]
        Q[2 * i + 1, 1] = d[i]
        if i > 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    # Fill higher-order divided differences
    for col in range(2, m):
        for row in range(col, m):
            denom = z[row] - z[row - col]
            if abs(denom) < 1e-14:
                raise ZeroDivisionError(f"Confluent singularity at row {row}, col {col}.")
            Q[row, col] = (Q[row, col - 1] - Q[row - 1, col - 1]) / denom

    # Evaluate polynomial at x_eval
    y_pred = Q[0, 0]
    prod_term = 1.0
    basis_terms = [{"k": 0, "coeff": float(Q[0, 0]), "product": 1.0, "term_val": float(Q[0, 0])}]

    for k in range(1, m):
        prod_term *= (x_eval - z[k - 1])
        term_val = Q[k, k] * prod_term
        y_pred += term_val
        basis_terms.append({
            "k": k,
            "coeff": float(Q[k, k]),
            "product": float(prod_term),
            "term_val": float(term_val)
        })

    # Prepare readable divided difference tableau
    table_rows = []
    for i in range(m):
        row_dict = {
            "index": i,
            "z_i": z[i],
            "f[z_i]": float(Q[i, 0])
        }
        for j in range(1, min(i + 1, 6)):
            row_dict[f"Order_{j}"] = float(Q[i, j])
        table_rows.append(row_dict)

    return {
        "y_pred": float(y_pred),
        "polynomial_degree": m - 1,
        "x_eval": float(x_eval),
        "slopes_used": d,
        "basis_terms": basis_terms,
        "divided_diff_table": table_rows
    }


def compare_lagrange_vs_hermite(
    x_nodes: List[float],
    y_nodes: List[float],
    x_eval: float,
    d_nodes: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Directly compares predictions from Lagrange vs Hermite interpolation at x_eval.
    Highlights how slope information prevents artificial overshoot/undershoot.
    """
    from .lagrange import lagrange_interpolation

    lagrange_res = lagrange_interpolation(x_nodes, y_nodes, x_eval)
    hermite_res = hermite_interpolation(x_nodes, y_nodes, d_nodes=d_nodes, x_eval=x_eval)

    diff = abs(hermite_res["y_pred"] - lagrange_res["y_pred"])

    return {
        "x_eval": x_eval,
        "lagrange_prediction": lagrange_res["y_pred"],
        "hermite_prediction": hermite_res["y_pred"],
        "absolute_difference": diff,
        "lagrange_degree": lagrange_res["polynomial_degree"],
        "hermite_degree": hermite_res["polynomial_degree"],
        "improvement_slopes": hermite_res["slopes_used"]
    }


def benchmark_hermite(
    x_nodes: List[float],
    y_nodes: List[float],
    x_eval: float,
    d_nodes: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Benchmarks from-scratch Hermite interpolation against scipy.interpolate.KroghInterpolator.
    """
    scratch_res = hermite_interpolation(x_nodes, y_nodes, d_nodes=d_nodes, x_eval=x_eval)
    d = scratch_res["slopes_used"]

    try:
        # KroghInterpolator accepts confluent nodes xi with values and derivatives in yi
        xi = []
        yi = []
        for x_val, y_val, d_val in zip(x_nodes, y_nodes, d):
            xi.extend([x_val, x_val])
            yi.extend([y_val, d_val])

        krogh = interpolate.KroghInterpolator(xi, yi)
        scipy_val = float(krogh(x_eval))
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
        "slopes": d
    }
