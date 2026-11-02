"""
Configuration module for Smart Academic Planner & CGPA Predictor.
Contains university grading scales, attendance thresholds, evaluation weights,
and application defaults. All settings are configurable and user-verifiable.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# University Grade Scale (Default: Standard Indian/UGC 10-point scale)
# Can be adjusted to match specific university regulations (e.g., VTU, AKTU, SPPU, Anna Univ)
# ---------------------------------------------------------------------------
DEFAULT_GRADE_SCALE: Dict[str, Tuple[float, int, int]] = {
    # Grade: (Grade Point, Min Marks %, Max Marks %)
    "O":  (10.0, 90, 100),
    "A+": (9.0,  80, 89),
    "A":  (8.0,  70, 79),
    "B+": (7.0,  60, 69),
    "B":  (6.0,  50, 59),
    "C":  (5.0,  45, 49),
    "P":  (4.0,  40, 44),
    "F":  (0.0,  0,  39),
}

# Ordered list of letter grades from highest to lowest
GRADE_CHOICES: List[str] = list(DEFAULT_GRADE_SCALE.keys())

# Default minimum mandatory attendance percentage
DEFAULT_MIN_ATTENDANCE_PCT: float = 75.0

# Academic course structure defaults
TOTAL_SEMESTERS_DEFAULT: int = 8
DEFAULT_DAILY_STUDY_HOURS: float = 4.0

# Mark Distribution Weights (Internal, Mid-Term, End-Semester)
DEFAULT_MARK_WEIGHTS = {
    "internal": 0.30,  # 30% Continuous Internal Assessment (CIA)
    "midterm": 0.20,   # 20% Mid-Semester Examination
    "endsem": 0.50,    # 50% End-Semester Examination
}

# Ebbinghaus forgetting model default parameters
DEFAULT_FORGETTING_RATE: float = 0.08   # Decay parameter d
DEFAULT_LEARNING_RATE: float = 0.35     # Acquisition parameter k
REVISION_THRESHOLD_PCT: float = 70.0    # Trigger revision when retention drops below this %

# Numerical Method Tolerances
DEFAULT_TOLERANCE: float = 1e-6
DEFAULT_MAX_ITERATIONS: int = 100

# UI Theme Configuration
APP_TITLE = "Smart Academic Planner & CGPA Predictor"
APP_SUBTITLE = "Numerical Analysis Programming (NAP) Course Project"
APP_VERSION = "2.0.0"
