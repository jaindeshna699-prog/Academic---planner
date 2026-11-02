"""
Numerical Analysis Programming (NAP) Core Module.
Contains pure from-scratch mathematical implementations paired with SciPy/NumPy
benchmarking implementations, iteration tables, and error diagnostics.
"""

from .root_finding import bisection_method, newton_raphson_method, benchmark_root_finding
from .fixed_point import fixed_point_iteration, benchmark_fixed_point
from .lagrange import lagrange_interpolation, benchmark_lagrange
from .hermite import hermite_interpolation, benchmark_hermite
from .interpolation import newton_polynomial_interpolation, benchmark_interpolation
from .least_squares import least_squares_fit, benchmark_least_squares
from .regression import linear_regression, polynomial_regression, predict_regression, benchmark_regression
from .numerical_integration import trapezoidal_rule, benchmark_trapezoid
from .runge_kutta import rk4_solve, rk45_adaptive_solve, benchmark_runge_kutta
from .linear_systems import gauss_elimination, gauss_jordan_elimination, benchmark_linear_system
from .power_method import power_method, benchmark_power_method

__all__ = [
    "bisection_method",
    "newton_raphson_method",
    "benchmark_root_finding",
    "fixed_point_iteration",
    "benchmark_fixed_point",
    "lagrange_interpolation",
    "newton_polynomial_interpolation",
    "hermite_interpolation",
    "benchmark_lagrange",
    "benchmark_hermite",
    "benchmark_interpolation",
    "least_squares_fit",
    "linear_regression",
    "polynomial_regression",
    "predict_regression",
    "benchmark_least_squares",
    "benchmark_regression",
    "trapezoidal_rule",
    "benchmark_trapezoid",
    "rk4_solve",
    "rk45_adaptive_solve",
    "benchmark_runge_kutta",
    "gauss_elimination",
    "gauss_jordan_elimination",
    "benchmark_linear_system",
    "power_method",
    "benchmark_power_method",
]
