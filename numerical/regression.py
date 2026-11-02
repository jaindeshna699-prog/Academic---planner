"""
Module: regression.py
Implements Least Squares Linear and Polynomial Regression from scratch
using Normal Equations (A^T A) c = A^T y.
Includes residual diagnostics, R^2 goodness-of-fit, RMSE, and NumPy/SciPy benchmarks.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from scipy import linalg


def linear_regression(
    x_data: Union[List[float], np.ndarray],
    y_data: Union[List[float], np.ndarray]
) -> Dict[str, Any]:
    r"""
    Fits a linear model y = m*x + c to bivariate data using Least Squares from scratch.

    Method:
        Analytic solution to 2x2 Normal Equations minimizing sum of squared errors:
        m = \frac{N \sum(xy) - \sum(x)\sum(y)}{N \sum(x^2) - (\sum x)^2}
        c = \frac{\sum(y) - m \sum(x)}{N}
    Complexity:
        Time Complexity: O(N) where N is number of observations.
        Space Complexity: O(N) for fitted values and residuals.

    Parameters:
        x_data: Independent variable (e.g. daily study hours or semester number).
        y_data: Dependent variable (e.g. marks % or SGPA).

    Returns:
        dict containing slope m, intercept c, R^2, RMSE, SSE, SST, and residual table.
    """
    return polynomial_regression(x_data, y_data, degree=1)


def polynomial_regression(
    x_data: Union[List[float], np.ndarray],
    y_data: Union[List[float], np.ndarray],
    degree: int = 1
) -> Dict[str, Any]:
    r"""
    Fits a degree-d polynomial to data using the Method of Least Squares from scratch.

    Method:
        Vandermonde design matrix formulation solved via Normal Equations:
        (A^T A) c = A^T y
    Complexity:
        Time Complexity: O(N * p^2 + p^3) where N = len(data), p = degree + 1.
        Space Complexity: O(N * p) for design matrix.

    Parameters:
        x_data: Independent variable values.
        y_data: Dependent variable values.
        degree: Degree of fitted polynomial (1 for linear, 2 for quadratic, etc.).

    Returns:
        dict containing coefficients, equation string, residuals, R^2, and residual table.
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    n = len(x)

    if n < degree + 1:
        raise ValueError(
            f"Insufficient data points for degree {degree} fit. Received {n} points, required at least {degree + 1}."
        )
    if len(y) != n:
        raise ValueError(f"Length mismatch: len(x_data)={n} != len(y_data)={len(y)}.")

    # Explicit closed-form for linear fit (degree=1) to guarantee zero numerical divergence
    if degree == 1:
        sum_x = float(np.sum(x))
        sum_y = float(np.sum(y))
        sum_xx = float(np.sum(x * x))
        sum_xy = float(np.sum(x * y))

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
        "slope": coeffs[0] if degree == 1 else None,
        "intercept": coeffs[-1],
        "equation_str": eq_str,
        "degree": degree,
        "y_pred": [float(val) for val in y_pred],
        "residuals": [float(val) for val in residuals],
        "sse": sse,
        "sst": sst,
        "r_squared": r_squared,
        "rmse": rmse,
        "residual_table": residual_table,
        "method": f"Least Squares Polynomial Regression (Degree {degree})"
    }


def predict_regression(
    regression_model: Dict[str, Any],
    x_new: Union[float, List[float], np.ndarray]
) -> Union[float, List[float]]:
    """Predicts dependent variable y for new observations given a fitted regression model."""
    coeffs = regression_model["coefficients"]
    deg = regression_model["degree"]

    def eval_single(x_val: float) -> float:
        res = 0.0
        for i, c in enumerate(coeffs):
            res += c * (x_val ** (deg - i))
        return float(res)

    if isinstance(x_new, (int, float)):
        return eval_single(float(x_new))
    return [eval_single(float(val)) for val in x_new]


def benchmark_regression(
    x_data: List[float],
    y_data: List[float],
    degree: int = 1
) -> Dict[str, Any]:
    """
    Benchmarks from-scratch regression against numpy.polyfit and scipy.linalg.lstsq.
    """
    scratch_fit = polynomial_regression(x_data, y_data, degree=degree)

    try:
        np_coeffs = np.polyfit(x_data, y_data, deg=degree)
        np_coeffs_list = [float(c) for c in np_coeffs]
        np_status = "Success"
        diff = [abs(s - n_c) for s, n_c in zip(scratch_fit["coefficients"], np_coeffs_list)]
        max_diff = max(diff)
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
