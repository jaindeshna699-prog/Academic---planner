"""
Module: root_finding.py
Implements Bisection and Newton-Raphson root-finding algorithms from scratch
for target CGPA and required examination marks determination.
Includes SciPy benchmark comparisons and comprehensive iteration diagnostics.
"""

from typing import Callable, Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import optimize


def bisection_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    Finds a root of f(x) = 0 on [a, b] using the Bisection Method from scratch.

    Method:
        Interval halving based on Bolzano's Intermediate Value Theorem.
    Formula:
        c_k = (a_k + b_k) / 2
        If f(a_k) * f(c_k) < 0, then b_{k+1} = c_k, else a_{k+1} = c_k.
    Complexity:
        Time Complexity: O(log2((b - a) / tol))
        Space Complexity: O(N) where N is iteration count (for iteration table).

    Parameters:
        f: Target scalar continuous function.
        a: Left bracket endpoint.
        b: Right bracket endpoint.
        tol: Convergence tolerance for bracket width or |f(c)|.
        max_iter: Maximum permitted iterations.

    Returns:
        dict containing:
            - root: Approximated root value.
            - iterations: Number of iterations performed.
            - converged: Boolean indicating successful convergence.
            - f_root: Value of f(root).
            - history: List of iteration steps (table).
    """
    if a >= b:
        raise ValueError(f"Invalid interval [a, b]: a ({a}) must be strictly less than b ({b}).")

    fa = f(a)
    fb = f(b)

    if abs(fa) < 1e-14:
        return {
            "root": a,
            "iterations": 0,
            "converged": True,
            "f_root": fa,
            "history": [{"iteration": 0, "a": a, "b": b, "c": a, "f_c": fa, "error": 0.0}]
        }
    if abs(fb) < 1e-14:
        return {
            "root": b,
            "iterations": 0,
            "converged": True,
            "f_root": fb,
            "history": [{"iteration": 0, "a": a, "b": b, "c": b, "f_c": fb, "error": 0.0}]
        }

    if fa * fb > 0:
        raise ValueError(
            f"Bisection bracket condition violated: f(a)={fa:.6f} and f(b)={fb:.6f} must have opposite signs. "
            f"f(a)*f(b) = {fa*fb:.6f} > 0."
        )

    history: List[Dict[str, Any]] = []
    c = a
    for k in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        err = (b - a) / 2.0
        rel_err = err / abs(c) if abs(c) > 1e-12 else err

        history.append({
            "iteration": k,
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "f_c": float(fc),
            "bracket_width": float(b - a),
            "error": float(err),
            "rel_error": float(rel_err)
        })

        if abs(fc) < tol or err < tol:
            return {
                "root": float(c),
                "iterations": k,
                "converged": True,
                "f_root": float(fc),
                "history": history
            }

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return {
        "root": float(c),
        "iterations": max_iter,
        "converged": False,
        "f_root": float(f(c)),
        "history": history
    }


def newton_raphson_method(
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]] = None,
    x0: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    Finds a root of f(x) = 0 using the Newton-Raphson Method from scratch.

    Method:
        First-order Taylor expansion tangent line linearization.
    Formula:
        x_{n+1} = x_n - f(x_n) / f'(x_n)
    Complexity:
        Time Complexity: O(k) evaluations with quadratic convergence near simple root.
        Space Complexity: O(k) for iteration table storage.

    Parameters:
        f: Objective function.
        df: Derivative function f'(x). If None, calculated via central differences.
        x0: Initial guess.
        tol: Convergence tolerance for step size |x_{n+1} - x_n| and |f(x)|.
        max_iter: Maximum permitted iterations.

    Returns:
        dict containing:
            - root: Approximated root value.
            - iterations: Number of iterations performed.
            - converged: Boolean indicating successful convergence.
            - f_root: Value of f(root).
            - history: Iteration table list.
    """
    if df is None:
        h = 1e-7
        df = lambda x: (f(x + h) - f(x - h)) / (2.0 * h)

    history: List[Dict[str, Any]] = []
    x_curr = float(x0)

    for k in range(1, max_iter + 1):
        fx = f(x_curr)
        dfx = df(x_curr)

        if abs(dfx) < 1e-14:
            raise ZeroDivisionError(
                f"Newton-Raphson failed: derivative f'(x)={dfx:.2e} is effectively zero at iteration {k}, x={x_curr:.6f}."
            )

        step = fx / dfx
        x_next = x_curr - step
        rel_err = abs(step) / abs(x_next) if abs(x_next) > 1e-12 else abs(step)

        history.append({
            "iteration": k,
            "x_n": float(x_curr),
            "f_x": float(fx),
            "df_x": float(dfx),
            "step": float(step),
            "x_next": float(x_next),
            "rel_error": float(rel_err)
        })

        if abs(step) < tol or abs(fx) < tol:
            return {
                "root": float(x_next),
                "iterations": k,
                "converged": True,
                "f_root": float(f(x_next)),
                "history": history
            }

        x_curr = x_next

    return {
        "root": float(x_curr),
        "iterations": max_iter,
        "converged": False,
        "f_root": float(f(x_curr)),
        "history": history
    }


def solve_required_sgpa_bisection(
    current_cgpa: float,
    completed_credits: float,
    upcoming_credits: float,
    target_cgpa: float,
    min_sgpa: float = 0.0,
    max_sgpa: float = 10.0,
    tol: float = 1e-5
) -> Dict[str, Any]:
    """
    Determines required upcoming semester SGPA to achieve target CGPA using Bisection.

    Formula:
        f(x) = ( (completed_credits * current_cgpa) + (upcoming_credits * x) ) / (completed_credits + upcoming_credits) - target_cgpa = 0
    """
    total_credits = completed_credits + upcoming_credits
    if total_credits <= 0:
        raise ValueError("Total credits must be strictly positive.")

    def obj(sgpa: float) -> float:
        cgpa_val = ((completed_credits * current_cgpa) + (upcoming_credits * sgpa)) / total_credits
        return cgpa_val - target_cgpa

    return bisection_method(obj, min_sgpa, max_sgpa, tol=tol)


def benchmark_root_finding(
    f: Callable[[float], float],
    a: float,
    b: float,
    x0: Optional[float] = None,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """
    Cross-benchmarks from-scratch Bisection and Newton-Raphson against SciPy's
    optimize.bisect and optimize.newton.
    """
    if x0 is None:
        x0 = (a + b) / 2.0

    # From-scratch executions
    scratch_bisect = bisection_method(f, a, b, tol=tol)
    scratch_newton = newton_raphson_method(f, x0=x0, tol=tol)

    # SciPy executions
    try:
        scipy_bisect_root, scipy_bisect_res = optimize.bisect(f, a, b, xtol=tol, full_output=True)
        scipy_bisect_iters = scipy_bisect_res.iterations
    except Exception as e:
        scipy_bisect_root, scipy_bisect_iters = None, str(e)

    try:
        scipy_newton_root, scipy_newton_res = optimize.newton(f, x0, tol=tol, full_output=True)
        scipy_newton_iters = scipy_newton_res.iterations
    except Exception as e:
        scipy_newton_root, scipy_newton_iters = None, str(e)

    # Error differences
    bisect_diff = abs(scratch_bisect["root"] - scipy_bisect_root) if scipy_bisect_root is not None else None
    newton_diff = abs(scratch_newton["root"] - scipy_newton_root) if scipy_newton_root is not None else None

    return {
        "scratch_bisect": scratch_bisect,
        "scipy_bisect_root": scipy_bisect_root,
        "scipy_bisect_iters": scipy_bisect_iters,
        "bisect_difference": bisect_diff,
        "scratch_newton": scratch_newton,
        "scipy_newton_root": scipy_newton_root,
        "scipy_newton_iters": scipy_newton_iters,
        "newton_difference": newton_diff,
    }
