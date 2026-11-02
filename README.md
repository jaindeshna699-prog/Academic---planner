# Smart Academic Planner & CGPA Predictor
### University Numerical Analysis Programming (NAP) Course Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An intelligent, production-ready offline academic planner and CGPA predictor designed for university students and numerical analysis courses. The system merges practical academic management (semester CGPA calculations, attendance auditing, revision timetable generation, exam countdowns) with **9 numerical analysis algorithms implemented from scratch** and cross-verified against **SciPy and NumPy**.

---

## 🚀 Key Features

1. **Semester CGPA Calculator**:
   - Manage subjects, credits, and letter grades across all semesters.
   - Automatic calculation of SGPA per semester and cumulative CGPA.
   - Configurable university grade-point scale in `config.py` (Default: UGC 10-point scale: O=10, A+=9, A=8, B+=7, B=6, C=5, P=4, F=0).
2. **Target CGPA Planner**:
   - Computes the exact SGPA needed in remaining semesters to achieve a target CGPA.
   - Powered by the **Bisection Method** root solver $f(\text{SGPA}) = \text{CGPA}(\text{SGPA}) - \text{Target} = 0$.
   - Includes multi-credit sensitivity tables and mathematical feasibility warnings.
3. **Study Plan & Timetable Generator**:
   - Uses the **Power Method** (dominant eigenvector) to rank subjects by credit weight, weakness rating, syllabus backlog, and exam urgency.
   - Solves a linear system with **Gauss Elimination** to distribute daily study hours under credit/difficulty constraints.
   - Uses **Runge-Kutta (RK4)** on an Ebbinghaus retention ODE $dy/dt = k(1-y) - d \cdot y$ to automatically trigger revision milestones.
4. **Attendance Manager**:
   - Configurable mandatory threshold (default: 75%).
   - Calculates exact class buffer ("Can safely skip $N$ classes") or recovery requirements ("Must attend next $N$ classes consecutively").
   - Quick present/absent logger with instant persistence.
5. **Exam Countdown & Syllabus Tracker**:
   - Real-time countdown clocks and urgency indicators.
   - Topic completion progress bars and distribution charts.
6. **Numerical Methods Lab**:
   - Interactive sandbox for all 9 numerical methods with convergence tables, iteration logs, and side-by-side SciPy/NumPy benchmark comparisons.
7. **Multi-Format Export**:
   - CSV spreadsheets for CGPA history, study plans, and attendance audits.
   - Formatted academic PDF reports generated via ReportLab.

---

## 🧪 Laboratory Experiments Mapping (1 to 12)

| Lab Exp # | Numerical Method | Implementation File | Academic Feature in Planner | SciPy / NumPy Verification |
|:---:|:---|:---|:---|:---|
| **Lab 1** | **Bisection Method** | [`numerical/root_finding.py`](file:///numerical/root_finding.py) | Target CGPA Planner: Solves required SGPA to bridge cumulative deficit | `scipy.optimize.bisect` |
| **Lab 2** | **Newton-Raphson Method** | [`numerical/root_finding.py`](file:///numerical/root_finding.py) | Marks Solver: Determines required end-semester marks under non-linear curves | `scipy.optimize.newton` |
| **Lab 3** | **Fixed-Point Iteration** | [`numerical/fixed_point.py`](file:///numerical/fixed_point.py) | Effort Model: Inverts diminishing returns study-hours curve $x = g(x)$ | `scipy.optimize.fixed_point` |
| **Lab 4** | **Gauss Elimination** | [`numerical/linear_systems.py`](file:///numerical/linear_systems.py) | Study Time Allocation: Splits daily hours across subjects under credit constraints | `numpy.linalg.solve` |
| **Lab 5** | **Gauss-Jordan Elimination** | [`numerical/linear_systems.py`](file:///numerical/linear_systems.py) | Mark Balancing: Solves internal, midterm, and end-sem component equations | `numpy.linalg.solve` |
| **Lab 6** | **Power Method** | [`numerical/power_method.py`](file:///numerical/power_method.py) | Subject Ranking: Dominant eigenvector of pairwise priority matrix | `numpy.linalg.eig` |
| **Lab 7** | **Lagrange Interpolation** | [`numerical/lagrange.py`](file:///numerical/lagrange.py) | SGPA Trajectory: Estimates future/missing semester performance | `scipy.interpolate.lagrange` |
| **Lab 8** | **Hermite Interpolation** | [`numerical/hermite.py`](file:///numerical/hermite.py) | Slope-Guided Prediction: Incorporates improvement velocity $f'(x_i)$ | `scipy.interpolate.KroghInterpolator` |
| **Lab 9** | **Least Squares Regression** | [`numerical/least_squares.py`](file:///numerical/least_squares.py) | Empirical Fit: Regresses study hours vs. marks with residuals and $R^2$ | `numpy.polyfit` / `scipy.linalg.lstsq` |
| **Lab 10** | **Trapezoidal Rule** | [`numerical/numerical_integration.py`](file:///numerical/numerical_integration.py) | Cumulative Effort: Integrates daily study hours curve across the semester | `scipy.integrate.trapezoid` |
| **Lab 11** | **Runge-Kutta 4th Order (RK4)** | [`numerical/runge_kutta.py`](file:///numerical/runge_kutta.py) | Memory Decay: Simulates syllabus learning/forgetting ODE $dy/dt$ | `scipy.integrate.solve_ivp` |
| **Lab 12** | **Adaptive Runge-Kutta (RK45)**| [`numerical/runge_kutta.py`](file:///numerical/runge_kutta.py) | Revision Timing: Error-controlled adaptive step integration | `scipy.integrate.solve_ivp(method='RK45')` |

---

## 📂 Project Architecture

```
study plan/
├── app.py                          # Streamlit web interface with 8 interactive pages
├── config.py                       # 10-point grade scale, attendance rules, defaults
├── database.py                     # SQLite persistence layer (CRUD for courses, plans, logs)
├── sample_data.py                  # Seed data loader for out-of-the-box operation
├── export_utils.py                 # CSV and ReportLab PDF document compilation engine
├── requirements.txt                # Python package dependencies
├── README.md                       # Complete guide & lab mapping
├── viva_notes.md                   # NAP viva question bank & method derivations
├── numerical/                      # Pure from-scratch mathematical modules
│   ├── __init__.py
│   ├── root_finding.py             # Bisection & Newton-Raphson methods
│   ├── fixed_point.py              # Fixed-Point Iteration (study-hours model)
│   ├── lagrange.py                 # Lagrange polynomial interpolation
│   ├── hermite.py                  # Hermite interpolation (confluent divided differences)
│   ├── least_squares.py            # Normal equations linear/quadratic regression
│   ├── numerical_integration.py    # Composite Trapezoidal rule
│   ├── runge_kutta.py              # Classical RK4 & adaptive RK45 Fehlberg solvers
│   ├── linear_systems.py           # Gauss & Gauss-Jordan with partial pivoting
│   └── power_method.py             # Power method & subject priority matrix
└── tests/                          # Automated Pytest suite (43 passed tests)
    ├── __init__.py
    ├── test_root_finding.py
    ├── test_fixed_point.py
    ├── test_interpolation.py
    ├── test_least_squares.py
    ├── test_integration.py
    ├── test_runge_kutta.py
    ├── test_linear_systems.py
    └── test_power_method.py
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11, 3.12, or higher
- Optional: `uv` package manager (recommended for fast virtual environment management)

### 1. Clone or Navigate to Directory
```bash
cd "study plan"
```

### 2. Create Virtual Environment
Using `uv`:
```bash
uv venv --python 3.12
.\.venv\Scripts\activate
```
Or using standard Python:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```bash
pytest -v
```
*Expected Output: `43 passed in < 2.0s`*

### 5. Launch Application
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 📸 Application Pages & UI Walkthrough

1. **🎓 CGPA Calculator & History**:
   - Real-time SGPA trajectory chart with cumulative CGPA benchmark line.
   - Tabular summary of all registered semesters with expandable subject editors.
   - Quick-add modal for registering courses, credits, and grades.
2. **🎯 Target CGPA Planner**:
   - Interactive slider for desired target CGPA.
   - Root-finding solver displaying exact required SGPA, iterations, and bracket progression.
   - Credit sensitivity analysis table.
3. **📅 Study Plan & Timetable Generator**:
   - Configurable daily hour budget and planning horizon.
   - Power method dominant eigenvector subject ranking weights.
   - Gauss elimination daily hour breakdown.
   - Automatically scheduled revision days flagged by the RK4 Ebbinghaus forgetting model.
4. **📊 Attendance Manager**:
   - Visual progress gauges for each registered subject.
   - Immediate mathematical determination of safe skip limits or deficit recovery requirements.
   - One-click daily attendance logging (+1 Present / +1 Absent).
5. **⏳ Exam Countdown & Syllabus**:
   - Countdown clocks to upcoming exams with urgency badges.
   - Progress bar module trackers and visual completion bar charts.
6. **🧪 Numerical Methods Lab**:
   - Dedicated tab for every numerical method.
   - Real-time calculation, step-by-step augmented matrices / iteration tables, and SciPy benchmark comparison cards.
7. **📥 Export Reports**:
   - One-click CSV downloads for CGPA, study plans, and attendance audits.
   - Downloadable multi-page PDF transcript with university formatting.
8. **⚙️ University Config & Settings**:
   - Custom student name, roll number, and institute.
   - Verification view of the 10-point grade scale.

---

## 🛡️ Mathematical & Engineering Quality Standards

- **Zero External API Dependencies**: Works completely offline with local SQLite database storage.
- **Strict Numerical Stability**:
  - Linear systems employ **row partial pivoting** to eliminate small pivot amplification and singular matrix detection.
  - Root-finding verifies bracket signs ($f(a) \cdot f(b) < 0$) and guards against $f'(x) \approx 0$.
  - Fixed-point iteration monitors Banach contraction condition $|g'(x)| < 1$ with divergence guardrails.
  - Interpolation verifies node distinctness to avoid division by zero.
- **Rigorously Tested**: Every from-scratch numerical method is covered by automated unit tests validating against SciPy/NumPy to within $\epsilon \le 10^{-4}$.
