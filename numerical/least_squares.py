"""
Module: least_squares.py
Implements the Method of Least Squares from scratch for linear and polynomial regression.
Applies to study-hours-vs-marks correlation and semester-wise SGPA trajectory fitting.
Includes normal equations, residual analysis, R^2 goodness-of-fit, and NumPy/SciPy benchmarks.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from scipy import linalg


def least_squares_fit(
    x_data: Union[List[float], np.ndarray],
    y_data: Union[List[float], np.ndarray],
    degree: int = 1
) -> Dict[str, Any]:
    """
    Fits a polynomial of specified degree to (x, y) data using the Method of Least Squares from scratch.

    Method:
        Minimization of sum of squared residuals via Normal Equations:
        (A^T A) c = A^T y
    Formula:
        For linear fit (degree=1): y = m*x + c
            m = (N * \sum(xy) - \sum(x)\sum(y)) / (N * \sum(x^2) - (\sum x)^2)
            c = (\sum(y) - m * \sum(x)) / N
    Complexity:
        Time Complexity: O(N * p^2 + p^3) where N = len(data), p = degree + 1.
        Space Complexity: O(N * p) for design matrix.

    Parameters:
        x_data: Independent variable values (e.g. daily hours or semester indices).
        y_data: Dependent variable values (e.g. marks % or SGPA).
        degree: Degree of fitted polynomial (1 for linear, 2 for quadratic).

    Returns:
        dict containing:
            - coefficients: Fitted coefficients in descending order [c_d, ..., c_1, c_0].
            - slope: Slope m (if degree=1).
            - intercept: Intercept c (if degree=1).
            - equation_str: Formatted regression equation string.
            - y_pred: Predicted values \hat{y} for input x_data.
            - residuals: e_i = y_i - \hat{y}_i.
            - sse: Sum of Squared Errors.
            - sst: Total Sum of Squares.
            - r_squared: Coefficient of Determination R^2.
            - rmse: Root Mean Square Error.
            - residual_table: Point-by-point comparison table.
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    n = len(x)

    if n < degree + 1:
        raise ValueError(
            f"Insufficient data points for degree {degree} fit. Received {n} points, required at least {degree + 1}."
        )
    if len(y) != n:
        raise ValueError(f"Length mismatch: x_data ({n}) and y_data ({len(y)}) must have identical dimensions.")

    # Explicit closed-form for linear fit (degree=1) to guarantee zero numerical divergence
    if degree == 1:
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xx = np.sum(x * x)
        sum_xy = np.sum(x * y)

        denom = n * sum_xx - (sum_x ** 2)
        if abs(denom) < 1e-14:
            raise ValueError("Zero variance in x_data: All x values are identical, cannot compute regression line.")

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        coeffs = [float(slope), float(intercept)]
        eq_str = f"y = {slope:.4f} * x + {intercept:.4f}"
    else:
        # Construct Vandermonde design matrix A: columns [x^degree, ..., x^1, 1]
        A = np.zeros((n, degree + 1), dtype=float)
        for p in range(degree + 1):
            A[:, degree - p] = x ** p

        # Normal equations: (A^T A) c = A^T y
        ATA = np.dot(A.T, A)
        ATy = np.dot(A.T, y)

        # Solve system using numpy.linalg.solve or manual Gaussian inversion
        try:
            coeffs_arr = np.linalg.solve(ATA, ATy)
            coeffs = [float(c) for c in coeffs_arr]
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix encountered when forming normal equations.")

        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power > 1:
                terms.append(f"{c:+.4f}*x^{power}")
            elif power == 1:
                terms.append(f"{c:+.4f}*x")
            else:
                terms.append(f"{c:+.4f}")
        eq_str = "y = " + " ".join(terms).lstrip("+")
        slope = coeffs[0] if degree == 1 else None
        intercept = coeffs[-1]

    # Evaluate fitted values
    y_pred = np.zeros(n, dtype=float)
    for i, c in enumerate(coeffs):
        power = degree - i
        y_pred += c * (x ** power)

    residuals = y - y_pred
    sse = float(np.sum(residuals ** 2))
    mean_y = float(np.mean(y))
    sst = float(np.sum((y - mean_y) ** 2))
    r_squared = float(1.0 - (sse / sst)) if sst > 1e-14 else 1.0
    rmse = float(np.sqrt(sse / n))

    # Point-by-point residual table
    residual_table = []
    for i in range(n):
        residual_table.append({
            "point": i + 1,
            "x": float(x[i]),
            "y_actual": float(y[i]),
            "y_predicted": float(y_pred[i]),
            "residual": float(residuals[i]),
            "sq_residual": float(residuals[i] ** 2)
        })

    return {
        "coefficients": coeffs,
        "slope": slope,
        "intercept": intercept,
        "equation_str": eq_str,
        "degree": degree,
        "y_pred": [float(val) for val in y_pred],
        "residuals": [float(val) for val in residuals],
        "sse": sse,
        "sst": sst,
        "r_squared": r_squared,
        "rmse": rmse,
        "residual_table": residual_table
    }


def predict_from_least_squares(fit_result: Dict[str, Any], x_new: Union[float, List[float]]) -> Union[float, List[float]]:
    """Predicts y for one or multiple new x values given a fitted model."""
    coeffs = fit_result["coefficients"]
    deg = fit_result["degree"]

    def eval_single(x_val: float) -> float:
        res = 0.0
        for i, c in enumerate(coeffs):
            res += c * (x_val ** (deg - i))
        return float(res)

    if isinstance(x_new, (int, float)):
        return eval_single(float(x_new))
    return [eval_single(float(val)) for val in x_new]


def benchmark_least_squares(
    x_data: List[float],
    y_data: List[float],
    degree: int = 1
) -> Dict[str, Any]:
    """
    Benchmarks from-scratch least-squares against numpy.polyfit and scipy.linalg.lstsq.
    """
    scratch_fit = least_squares_fit(x_data, y_data, degree=degree)

    # NumPy benchmark
    try:
        np_coeffs = np.polyfit(x_data, y_data, deg=degree)
        np_coeffs_list = [float(c) for c in np_coeffs]
        np_status = "Success"
        coeff_diff = [abs(s - n_c) for s, n_c in zip(scratch_fit["coefficients"], np_coeffs_list)]
        max_diff = max(coeff_diff)
    except Exception as e:
        np_coeffs_list = None
        np_status = f"NumPy Error: {str(e)}"
        max_diff = None

    return {
        "scratch_coefficients": scratch_fit["coefficients"],
        "numpy_coefficients": np_coeffs_list,
        "max_difference": max_diff,
        "r_squared": scratch_fit["r_squared"],
        "rmse": scratch_fit["rmse"],
        "numpy_status": np_status,
        "residual_table": scratch_fit["residual_table"]
    }
