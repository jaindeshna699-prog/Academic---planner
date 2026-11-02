"""
Module: runge_kutta.py
Implements classical 4th-Order Runge-Kutta (RK4) and Adaptive Runge-Kutta (RK45)
from scratch to simulate syllabus mastery and the Ebbinghaus forgetting curve:
    dy/dt = k*(1 - y) - d*y
Includes automatic revision trigger scheduling and SciPy benchmark comparison.
"""

from typing import Callable, List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import integrate


def rk4_solve(
    dydt: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float = 0.5
) -> Dict[str, Any]:
    """
    Solves an Initial Value Problem dy/dt = f(t, y) using Classical 4th-Order Runge-Kutta (RK4) from scratch.

    Method:
        Classical RK4 with fixed step size h.
    Formula:
        k_1 = f(t_n, y_n)
        k_2 = f(t_n + h/2, y_n + (h/2)*k_1)
        k_3 = f(t_n + h/2, y_n + (h/2)*k_2)
        k_4 = f(t_n + h,   y_n + h*k_3)
        y_{n+1} = y_n + (h / 6) * (k_1 + 2*k_2 + 2*k_3 + k_4)
    Complexity:
        Time Complexity: O(N) where N = (t_end - t_start)/h steps, 4 function evaluations/step.
        Global Truncation Error: O(h^4).

    Parameters:
        dydt: Differential equation derivative function f(t, y).
        y0: Initial condition at t_span[0].
        t_span: (t_start, t_end).
        h: Fixed integration step size.

    Returns:
        dict containing:
            - t_values: Time points array.
            - y_values: Computed state values.
            - steps_table: Iteration table detailing k1, k2, k3, k4 per step.
    """
    t0, tf = t_span
    if tf <= t0:
        raise ValueError(f"Invalid time span: t_end ({tf}) must be strictly greater than t_start ({t0}).")
    if h <= 0:
        raise ValueError(f"Step size h ({h}) must be strictly positive.")

    n_steps = int(np.ceil((tf - t0) / h))
    t_vals = [float(t0)]
    y_vals = [float(y0)]
    steps_table: List[Dict[str, Any]] = []

    t_curr = float(t0)
    y_curr = float(y0)

    for step_idx in range(1, n_steps + 1):
        actual_h = min(h, tf - t_curr)
        if actual_h <= 1e-12:
            break

        k1 = dydt(t_curr, y_curr)
        k2 = dydt(t_curr + 0.5 * actual_h, y_curr + 0.5 * actual_h * k1)
        k3 = dydt(t_curr + 0.5 * actual_h, y_curr + 0.5 * actual_h * k2)
        k4 = dydt(t_curr + actual_h, y_curr + actual_h * k3)

        y_next = y_curr + (actual_h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t_next = t_curr + actual_h

        steps_table.append({
            "step": step_idx,
            "t": float(t_curr),
            "y": float(y_curr),
            "h": float(actual_h),
            "k1": float(k1),
            "k2": float(k2),
            "k3": float(k3),
            "k4": float(k4),
            "y_next": float(y_next)
        })

        t_curr = t_next
        y_curr = y_next
        t_vals.append(float(t_curr))
        y_vals.append(float(y_curr))

    return {
        "t_values": t_vals,
        "y_values": y_vals,
        "steps_table": steps_table,
        "method": "Runge-Kutta 4th Order (RK4)"
    }


def rk45_adaptive_solve(
    dydt: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    tol: float = 1e-5,
    h_init: float = 0.5,
    h_min: float = 1e-4,
    h_max: float = 2.0
) -> Dict[str, Any]:
    """
    Solves dy/dt = f(t, y) using Adaptive Runge-Kutta-Fehlberg (RK45) from scratch.

    Method:
        Embedded pair (Fehlberg 4th and 5th order) with automatic step-size control.
    Local Truncation Error:
        LTE = |y^{(5)} - y^{(4)}|
        Step scale: s = 0.9 * (tol / (LTE + 1e-12)) ** 0.2
        h_{new} = clamp(h * s, h_min, h_max)
    Complexity:
        Adaptive: O(N) evaluations, minimizing steps in smooth regions.
    """
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("Invalid time span: t_end must be greater than t_start.")

    # Fehlberg Butcher Tableau coefficients
    a2, b21 = 1.0/4.0, 1.0/4.0
    a3, b31, b32 = 3.0/8.0, 3.0/32.0, 9.0/32.0
    a4, b41, b42, b43 = 12.0/13.0, 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0
    a5, b51, b52, b53, b54 = 1.0, 439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0
    a6, b61, b62, b63, b64, b65 = 1.0/2.0, -8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0

    # 4th order coefficients
    c1, c3, c4, c5 = 25.0/216.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0
    # 5th order coefficients
    d1, d3, d4, d5, d6 = 16.0/135.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0

    t_curr = float(t0)
    y_curr = float(y0)
    h = min(h_init, tf - t_curr)

    t_vals = [t_curr]
    y_vals = [y_curr]
    steps_table: List[Dict[str, Any]] = []

    step_counter = 0
    max_steps = 2000

    while t_curr < tf and step_counter < max_steps:
        step_counter += 1
        h = min(h, tf - t_curr)

        k1 = dydt(t_curr, y_curr)
        k2 = dydt(t_curr + a2 * h, y_curr + h * (b21 * k1))
        k3 = dydt(t_curr + a3 * h, y_curr + h * (b31 * k1 + b32 * k2))
        k4 = dydt(t_curr + a4 * h, y_curr + h * (b41 * k1 + b42 * k2 + b43 * k3))
        k5 = dydt(t_curr + a5 * h, y_curr + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
        k6 = dydt(t_curr + a6 * h, y_curr + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))

        # 4th and 5th order approximations
        y4 = y_curr + h * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5)
        y5 = y_curr + h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6)

        lte = abs(y5 - y4)
        accepted = lte <= tol or h <= h_min

        steps_table.append({
            "step": step_counter,
            "t": float(t_curr),
            "y": float(y_curr),
            "h_used": float(h),
            "lte": float(lte),
            "accepted": accepted
        })

        if accepted:
            t_curr += h
            y_curr = y5  # Local extrapolation (5th order)
            t_vals.append(float(t_curr))
            y_vals.append(float(y_curr))

        # Adapt step size
        if lte < 1e-15:
            scale = 2.0
        else:
            scale = 0.9 * ((tol / lte) ** 0.2)
        scale = max(0.2, min(2.0, scale))
        h = max(h_min, min(h_max, h * scale))

    return {
        "t_values": t_vals,
        "y_values": y_vals,
        "steps_table": steps_table,
        "method": "Adaptive Runge-Kutta-Fehlberg (RK45)"
    }


def simulate_retention_and_schedule_revisions(
    initial_mastery: float = 0.20,
    learning_rate: float = 0.35,
    decay_rate: float = 0.08,
    days: int = 30,
    revision_threshold: float = 0.70,
    study_active: bool = True
) -> Dict[str, Any]:
    """
    Simulates student retention and triggers scheduled revision days.

    ODE Model:
        dy/dt = k * (1 - y) - d * y
        where y(t) in [0, 1] is retention/syllabus mastery.
        When not studying (idle decay): k = 0, so dy/dt = -d * y.
    """
    def model(t: float, y: float) -> float:
        k = learning_rate if study_active else 0.0
        return k * (1.0 - y) - decay_rate * y

    sim_rk4 = rk4_solve(model, y0=initial_mastery, t_span=(0.0, float(days)), h=0.5)

    # Detect days where retention drops below threshold
    revision_days = []
    t_arr = sim_rk4["t_values"]
    y_arr = sim_rk4["y_values"]

    # Sample daily points
    daily_points = []
    for day in range(days + 1):
        idx = int(np.argmin(np.abs(np.array(t_arr) - day)))
        ret = y_arr[idx]
        is_rev = bool(ret < revision_threshold and day > 0 and (day % 3 == 0 or ret < 0.60))
        if is_rev:
            revision_days.append(day)
        daily_points.append({
            "day": day,
            "retention_pct": round(ret * 100.0, 1),
            "needs_revision": is_rev
        })

    return {
        "simulation": sim_rk4,
        "daily_points": daily_points,
        "revision_days": sorted(list(set(revision_days))),
        "final_retention": y_arr[-1]
    }


def benchmark_runge_kutta(
    dydt: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float = 0.5
) -> Dict[str, Any]:
    """
    Benchmarks from-scratch RK4 and RK45 against scipy.integrate.solve_ivp.
    """
    scratch_rk4 = rk4_solve(dydt, y0, t_span, h=h)
    scratch_rk45 = rk45_adaptive_solve(dydt, y0, t_span, h_init=h)

    # SciPy solve_ivp
    try:
        def ivp_f(t, y):
            return [dydt(t, y[0])]

        scipy_sol = integrate.solve_ivp(
            ivp_f, t_span, [y0],
            method='RK45',
            t_eval=scratch_rk4["t_values"]
        )
        scipy_final = float(scipy_sol.y[0][-1])
        scipy_status = "Success"
        difference = abs(scratch_rk4["y_values"][-1] - scipy_final)
    except Exception as e:
        scipy_final = None
        scipy_status = f"SciPy Error: {str(e)}"
        difference = None

    return {
        "scratch_rk4_final": scratch_rk4["y_values"][-1],
        "scratch_rk45_final": scratch_rk45["y_values"][-1],
        "scipy_final": scipy_final,
        "difference": difference,
        "scipy_status": scipy_status
    }
