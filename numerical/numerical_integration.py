"""
Module: numerical_integration.py
Implements the Composite Trapezoidal Rule from scratch to compute cumulative
study effort (area under daily study hours curve) over a week or semester.
Includes interval-by-interval integration tables and SciPy benchmarking.
"""

from typing import Callable, List, Dict, Any, Union, Optional
import numpy as np
from scipy import integrate


def trapezoidal_rule(
    data_or_func: Union[Callable[[float], float], List[float], np.ndarray],
    x_nodes: Optional[Union[List[float], np.ndarray]] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    n_intervals: int = 100
) -> Dict[str, Any]:
    """
    Computes definite integral using the Composite Trapezoidal Rule from scratch.

    Method:
        Approximates the area under the curve by summing trapezoids across subintervals.
    Formula:
        For discrete data (x_i, y_i):
            Area = \sum_{i=0}^{n-1} \frac{x_{i+1} - x_i}{2} (y_i + y_{i+1})
        For continuous f(x) over [a, b] with uniform h = (b - a)/n:
            Area = \frac{h}{2} [ f(a) + 2 \sum_{i=1}^{n-1} f(x_i) + f(b) ]
    Complexity:
        Time Complexity: O(n) where n is the number of trapezoids.
        Space Complexity: O(n) for the interval area iteration table.

    Parameters:
        data_or_func: Continuous function f(x) OR array of y values (study hours).
        x_nodes: Array of corresponding x coordinates (e.g. day indices 1, 2, ..., N).
        a: Lower integration limit (for continuous function).
        b: Upper integration limit (for continuous function).
        n_intervals: Number of subdivisions if continuous function is provided.

    Returns:
        dict containing:
            - total_integral: Cumulative effort (e.g., total study hours).
            - average_height: Mean value of function/effort over domain = Integral / (b - a).
            - n_trapezoids: Number of trapezoid strips.
            - interval_table: Step-by-step table of each trapezoid.
    """
    if callable(data_or_func):
        if a is None or b is None:
            raise ValueError("Integration limits 'a' and 'b' must be specified for continuous functions.")
        if a >= b:
            raise ValueError(f"Lower limit 'a' ({a}) must be strictly less than upper limit 'b' ({b}).")
        if n_intervals < 1:
            raise ValueError("n_intervals must be at least 1.")

        x = np.linspace(a, b, n_intervals + 1)
        y = np.array([float(data_or_func(val)) for val in x], dtype=float)
    else:
        y = np.asarray(data_or_func, dtype=float)
        if len(y) < 2:
            raise ValueError("Trapezoidal rule requires at least 2 data points.")

        if x_nodes is None:
            # Assume uniform spacing 0, 1, 2, ..., N-1
            x = np.arange(len(y), dtype=float)
        else:
            x = np.asarray(x_nodes, dtype=float)
            if len(x) != len(y):
                raise ValueError(f"Mismatch: len(x_nodes)={len(x)} != len(y)={len(y)}.")
            # Verify monotonic ordering
            if not np.all(np.diff(x) > 0):
                raise ValueError("x coordinates must be strictly increasing for trapezoidal integration.")

    n_strips = len(x) - 1
    total_area = 0.0
    interval_table: List[Dict[str, Any]] = []

    for i in range(n_strips):
        dx = x[i + 1] - x[i]
        strip_area = 0.5 * dx * (y[i] + y[i + 1])
        total_area += strip_area

        interval_table.append({
            "interval": i + 1,
            "x_left": float(x[i]),
            "x_right": float(x[i + 1]),
            "y_left": float(y[i]),
            "y_right": float(y[i + 1]),
            "width_dx": float(dx),
            "trapezoid_area": float(strip_area),
            "cumulative_area": float(total_area)
        })

    domain_width = float(x[-1] - x[0])
    avg_height = (total_area / domain_width) if domain_width > 1e-12 else 0.0

    return {
        "total_integral": float(total_area),
        "average_height": float(avg_height),
        "domain_start": float(x[0]),
        "domain_end": float(x[-1]),
        "n_trapezoids": n_strips,
        "x_values": [float(val) for val in x],
        "y_values": [float(val) for val in y],
        "interval_table": interval_table
    }


def compute_cumulative_study_effort(
    daily_hours: List[float],
    day_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Computes total student effort hours over a semester / test cycle using trapezoidal integration.
    """
    return trapezoidal_rule(daily_hours, x_nodes=day_indices)


def benchmark_trapezoid(
    y_values: List[float],
    x_values: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Benchmarks from-scratch trapezoidal integration against scipy.integrate.trapezoid.
    """
    scratch_res = trapezoidal_rule(y_values, x_nodes=x_values)

    try:
        if x_values is not None:
            scipy_val = float(integrate.trapezoid(y_values, x=x_values))
        else:
            scipy_val = float(integrate.trapezoid(y_values))
        scipy_status = "Success"
        difference = abs(scratch_res["total_integral"] - scipy_val)
    except Exception as e:
        scipy_val = None
        scipy_status = f"SciPy Error: {str(e)}"
        difference = None

    return {
        "scratch_integral": scratch_res["total_integral"],
        "scipy_integral": scipy_val,
        "difference": difference,
        "scipy_status": scipy_status,
        "average_effort": scratch_res["average_height"],
        "interval_table": scratch_res["interval_table"]
    }
