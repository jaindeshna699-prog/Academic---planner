"""
Unit tests for Runge-Kutta RK4 and RK45 in numerical/runge_kutta.py.
Validates exponential decay, syllabus forgetting curve ODE, and scipy.integrate.solve_ivp.
"""

import pytest
import numpy as np
from scipy import integrate
from numerical.runge_kutta import (
    rk4_solve,
    rk45_adaptive_solve,
    simulate_retention_and_schedule_revisions,
    benchmark_runge_kutta
)


def test_rk4_exponential_decay():
    # dy/dt = -0.5 * y, y(0) = 1.0 => exact y(t) = exp(-0.5 * t)
    # at t = 2: y(2) = exp(-1) ~= 0.36787944
    dydt = lambda t, y: -0.5 * y
    res = rk4_solve(dydt, y0=1.0, t_span=(0.0, 2.0), h=0.1)
    y_final = res["y_values"][-1]
    exact_final = float(np.exp(-1.0))
    assert pytest.approx(y_final, abs=1e-4) == exact_final
    assert len(res["steps_table"]) == 20


def test_rk45_adaptive_accuracy():
    # dy/dt = -y, y(0) = 1.0 on [0, 3] => y(3) = exp(-3) ~= 0.049787
    dydt = lambda t, y: -y
    res = rk45_adaptive_solve(dydt, y0=1.0, t_span=(0.0, 3.0), tol=1e-6)
    y_final = res["y_values"][-1]
    exact_final = float(np.exp(-3.0))
    assert pytest.approx(y_final, abs=1e-4) == exact_final


def test_simulate_retention_revision_trigger():
    # Simulate 30 days retention with high decay
    res = simulate_retention_and_schedule_revisions(
        initial_mastery=0.10,
        learning_rate=0.40,
        decay_rate=0.12,
        days=20,
        revision_threshold=0.75
    )
    assert len(res["daily_points"]) == 21
    assert "revision_days" in res
    assert isinstance(res["revision_days"], list)


def test_benchmark_runge_kutta_agreement():
    # Logistic growth dy/dt = y * (1 - y)
    dydt = lambda t, y: y * (1.0 - y)
    bench = benchmark_runge_kutta(dydt, y0=0.1, t_span=(0.0, 4.0), h=0.2)
    assert bench["scipy_status"] == "Success"
    assert bench["difference"] < 1e-3
