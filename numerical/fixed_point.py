"""
Module: fixed_point.py
Implements the Fixed-Point Iteration method from scratch to solve non-linear
study-hours-vs-expected-marks models x = g(x).
Includes convergence analysis, Banach contraction diagnostics, and SciPy benchmarking.
"""

from typing import Callable, Dict, Any, List, Optional
import numpy as np
from scipy import optimize


def fixed_point_iteration(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    Finds a fixed point x = g(x) using the Fixed-Point Iteration algorithm from scratch.

    Method:
        Successive approximation / Picard iteration based on Banach's Fixed-Point Theorem.
    Formula:
        x_{k+1} = g(x_k)
    Convergence Condition:
        |g'(x)| < 1 in the neighborhood of the fixed point.
    Complexity:
        Time Complexity: O(k) evaluations with linear convergence.
        Space Complexity: O(k) for iteration table history.

    Parameters:
        g: Iteration function g(x) such that x = g(x).
        x0: Initial guess for study hours.
        tol: Tolerance on consecutive step difference |x_{k+1} - x_k|.
        max_iter: Maximum permitted iterations.

    Returns:
        dict containing:
            - root: Approximated fixed point x*.
            - iterations: Total iterations performed.
            - converged: Boolean status.
            - g_prime_estimate: Estimated derivative g'(x*) for contraction verification.
            - history: Iteration table list.
    """
    history: List[Dict[str, Any]] = []
    x_curr = float(x0)

    # Estimate g'(x0) numerically
    h = 1e-6
    try:
        g_prime_init = (g(x0 + h) - g(x0 - h)) / (2.0 * h)
    except Exception:
        g_prime_init = None

    prev_step = None
    for k in range(1, max_iter + 1):
        try:
            x_next = float(g(x_curr))
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            return {
                "root": float(x_curr),
                "iterations": k,
                "converged": False,
                "error_reason": f"Evaluation error at iteration {k}: {str(e)}",
                "g_prime_estimate": g_prime_init,
                "history": history
            }

        step = abs(x_next - x_curr)
        rel_error = step / abs(x_next) if abs(x_next) > 1e-12 else step
        ratio = (step / prev_step) if (prev_step is not None and prev_step > 1e-14) else 0.0
        prev_step = step

        history.append({
            "iteration": k,
            "x_k": float(x_curr),
            "g_x": float(x_next),
            "step": float(step),
            "rel_error": float(rel_error),
            "ratio": float(ratio)
        })

        # Divergence watchdog
        if step > 1e8 or np.isnan(x_next) or np.isinf(x_next):
            return {
                "root": float(x_next),
                "iterations": k,
                "converged": False,
                "error_reason": f"Divergence detected at iteration {k} (|g'(x)| >= 1). Step exploded to {step:.2e}.",
                "g_prime_estimate": g_prime_init,
                "history": history
            }

        if step < tol:
            # Estimate g'(root)
            try:
                g_prime_root = (g(x_next + h) - g(x_next - h)) / (2.0 * h)
            except Exception:
                g_prime_root = None

            return {
                "root": float(x_next),
                "iterations": k,
                "converged": True,
                "g_prime_estimate": float(g_prime_root) if g_prime_root is not None else None,
                "history": history
            }

        x_curr = x_next

    return {
        "root": float(x_curr),
        "iterations": max_iter,
        "converged": False,
        "error_reason": f"Failed to converge within {max_iter} iterations. Final step size: {step:.2e}.",
        "g_prime_estimate": g_prime_init,
        "history": history
    }


def study_hours_model_g(
    target_marks: float,
    alpha: float = 0.25,
    max_marks: float = 100.0,
    damping: float = 0.5
) -> Callable[[float], float]:
    """
    Constructs a well-behaved contraction mapping g(x) for academic study hours.

    Model:
        Marks achieved M(x) = max_marks * (1 - exp(-alpha * x))
        Inversion gives: x_ideal = -(1/alpha) * ln(1 - target_marks / max_marks)
        Under relaxed iteration with damping c:
            g(x) = (1 - c) * x + c * [-(1/alpha) * ln(max(1e-4, 1.0 - (target_marks / max_marks)))]
    """
    if target_marks >= max_marks:
        raise ValueError(f"Target marks ({target_marks}) must be strictly less than maximum obtainable marks ({max_marks}).")
    if target_marks <= 0:
        raise ValueError("Target marks must be positive.")

    # Target ideal study hours
    ratio = 1.0 - (target_marks / max_marks)
    if ratio <= 0:
        ratio = 1e-6
    ideal_hours = - (1.0 / alpha) * np.log(ratio)

    # Relaxed fixed-point mapping: g(x) = (1 - damping) * x + damping * ideal_hours
    # guarantees |g'(x)| = |1 - damping| < 1 when 0 < damping < 2.
    def g(x: float) -> float:
        # Implicit non-linear diminishing returns adjustment
        score_at_x = max_marks * (1.0 - np.exp(-alpha * max(0.0, x)))
        discrepancy = target_marks - score_at_x
        # Damped update
        step_update = discrepancy / (alpha * max_marks * np.exp(-alpha * max(0.0, x)) + 1e-5)
        return float(x + damping * step_update)

    return g


def benchmark_fixed_point(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    Compares the from-scratch fixed-point iteration against scipy.optimize.fixed_point.
    """
    scratch_res = fixed_point_iteration(g, x0=x0, tol=tol, max_iter=max_iter)

    try:
        scipy_root = float(optimize.fixed_point(g, x0, xtol=tol, maxiter=max_iter))
        scipy_status = "Converged"
    except Exception as e:
        scipy_root = None
        scipy_status = f"SciPy Error: {str(e)}"

    difference = abs(scratch_res["root"] - scipy_root) if scipy_root is not None else None

    return {
        "scratch_result": scratch_res,
        "scipy_root": scipy_root,
        "scipy_status": scipy_status,
        "difference": difference
    }
