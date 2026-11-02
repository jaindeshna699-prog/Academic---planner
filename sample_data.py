"""
Sample data provider for Smart Academic Planner & CGPA Predictor.
Populates realistic academic data for a computer science / engineering student
enrolled in Numerical Analysis Programming (NAP) and core university courses.
Ensures instant out-of-the-box readiness upon first launch.
"""

from datetime import date, timedelta
from database import (
    init_db, reset_database, add_semester, add_subject,
    set_setting, add_study_log, recompute_semester_sgpa
)
from config import DEFAULT_GRADE_SCALE


def seed_sample_data(force_reset: bool = False) -> None:
    """Populate default semesters, subjects, attendance, and study logs."""
    init_db()
    if force_reset:
        reset_database()

    # 1. Base User Settings
    set_setting("target_cgpa", "8.75")
    set_setting("min_attendance_pct", "75.0")
    set_setting("daily_study_hours", "4.0")
    set_setting("student_name", "Arjun Sharma")
    set_setting("roll_number", "23CSB042")
    set_setting("university", "National Institute of Technology / University NAP Dept.")

    # 2. Semester 1 (Completed)
    add_semester(1, is_completed=1)
    sem1_subjects = [
        ("MA101", "Calculus & Linear Algebra", 4.0, "A+", 9.0, 27.0, 18.0, 42.0, 4, None, 10, 10, 42, 38),
        ("PH101", "Engineering Physics", 3.0, "A", 8.0, 25.0, 16.0, 39.0, 3, None, 10, 10, 40, 32),
        ("CS101", "Intro to Programming in C", 4.0, "O", 10.0, 29.0, 19.0, 48.0, 5, None, 12, 12, 45, 43),
        ("EE101", "Basic Electrical Engineering", 3.0, "B+", 7.0, 22.0, 14.0, 35.0, 2, None, 8, 8, 36, 28),
        ("EN101", "Professional Communication", 2.0, "A", 8.0, 26.0, 17.0, 38.0, 4, None, 6, 6, 28, 25),
        ("ME101", "Engineering Workshop Practice", 2.0, "A+", 9.0, 28.0, 18.0, 44.0, 4, None, 6, 6, 30, 29),
    ]
    for sub in sem1_subjects:
        add_subject(1, sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7], sub[8], sub[9], sub[10], sub[11], sub[12], sub[13])
    recompute_semester_sgpa(1)

    # 3. Semester 2 (Completed)
    add_semester(2, is_completed=1)
    sem2_subjects = [
        ("MA102", "Advanced Differential Equations", 4.0, "A", 8.0, 26.0, 16.0, 38.0, 3, None, 10, 10, 44, 35),
        ("CS102", "Data Structures & Algorithms", 4.0, "O", 10.0, 29.0, 20.0, 47.0, 5, None, 12, 12, 46, 44),
        ("EC101", "Digital Electronics & Logic", 3.0, "B+", 7.0, 23.0, 15.0, 34.0, 2, None, 10, 10, 38, 30),
        ("CH101", "Engineering Chemistry", 3.0, "A", 8.0, 25.0, 17.0, 39.0, 3, None, 10, 10, 40, 33),
        ("CS103", "Object Oriented Python Lab", 2.0, "A+", 9.0, 28.0, 19.0, 45.0, 5, None, 8, 8, 32, 31),
    ]
    for sub in sem2_subjects:
        add_subject(2, sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7], sub[8], sub[9], sub[10], sub[11], sub[12], sub[13])
    recompute_semester_sgpa(2)

    # 4. Semester 3 (Completed)
    add_semester(3, is_completed=1)
    sem3_subjects = [
        ("MA201", "Discrete Mathematics", 4.0, "A+", 9.0, 28.0, 18.0, 43.0, 4, None, 10, 10, 40, 36),
        ("CS201", "Computer Organization & Arch", 4.0, "B+", 7.0, 22.0, 14.0, 34.0, 2, None, 10, 10, 42, 30),
        ("CS202", "Design & Analysis of Algorithms", 4.0, "A", 8.0, 26.0, 17.0, 40.0, 4, None, 12, 12, 45, 38),
        ("CS203", "Database Management Systems", 3.0, "A+", 9.0, 27.0, 18.0, 44.0, 4, None, 10, 10, 38, 34),
        ("HS201", "Universal Human Values", 2.0, "A", 8.0, 26.0, 16.0, 39.0, 4, None, 6, 6, 26, 24),
    ]
    for sub in sem3_subjects:
        add_subject(3, sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7], sub[8], sub[9], sub[10], sub[11], sub[12], sub[13])
    recompute_semester_sgpa(3)

    # 5. Semester 4 (Completed)
    add_semester(4, is_completed=1)
    sem4_subjects = [
        ("MA202", "Probability & Statistics", 4.0, "A", 8.0, 25.0, 16.0, 39.0, 3, None, 10, 10, 42, 34),
        ("CS204", "Operating Systems", 4.0, "A+", 9.0, 28.0, 18.0, 44.0, 4, None, 12, 12, 44, 40),
        ("CS205", "Theory of Computation", 3.0, "B+", 7.0, 21.0, 13.0, 33.0, 2, None, 10, 10, 38, 27),
        ("CS206", "Software Engineering", 3.0, "A", 8.0, 26.0, 17.0, 40.0, 4, None, 10, 10, 36, 32),
        ("CS207", "OS & Networks Lab", 2.0, "O", 10.0, 30.0, 20.0, 49.0, 5, None, 8, 8, 30, 29),
    ]
    for sub in sem4_subjects:
        add_subject(4, sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7], sub[8], sub[9], sub[10], sub[11], sub[12], sub[13])
    recompute_semester_sgpa(4)

    # 6. Semester 5 (Completed)
    add_semester(5, is_completed=1)
    sem5_subjects = [
        ("CS301", "Computer Networks", 4.0, "A", 8.0, 26.0, 17.0, 41.0, 4, None, 12, 12, 44, 36),
        ("CS302", "Compiler Design", 4.0, "B+", 7.0, 23.0, 14.0, 34.0, 2, None, 10, 10, 40, 29),
        ("CS303", "Machine Learning Fundamentals", 3.0, "A+", 9.0, 28.0, 18.0, 45.0, 4, None, 10, 10, 38, 34),
        ("CS304", "Artificial Intelligence", 3.0, "A", 8.0, 25.0, 16.0, 39.0, 3, None, 10, 10, 36, 31),
        ("CS305", "Web Technologies Lab", 2.0, "O", 10.0, 29.0, 19.0, 48.0, 5, None, 8, 8, 32, 30),
    ]
    for sub in sem5_subjects:
        add_subject(5, sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7], sub[8], sub[9], sub[10], sub[11], sub[12], sub[13])
    recompute_semester_sgpa(5)

    # 7. Semester 6 (Current / Ongoing Semester - In Progress)
    add_semester(6, is_completed=0)
    today = date.today()
    sem6_subjects = [
        ("MA301", "Numerical Analysis Programming", 4.0, "A+", 9.0, 27.0, 18.0, 44.0, 4,
         (today + timedelta(days=12)).isoformat(), 12, 8, 38, 32), # 84.2% Attendance
        ("CS306", "Distributed Cloud Computing", 4.0, "A", 8.0, 24.0, 15.0, 38.0, 3,
         (today + timedelta(days=16)).isoformat(), 10, 6, 36, 26), # 72.2% Attendance (Below 75%!)
        ("CS307", "Information Security & Cryptography", 3.0, "B+", 7.0, 21.0, 14.0, 33.0, 2,
         (today + timedelta(days=21)).isoformat(), 10, 5, 34, 27), # 79.4% Attendance
        ("CS308", "Big Data Analytics", 3.0, "A", 8.0, 25.0, 16.0, 39.0, 3,
         (today + timedelta(days=26)).isoformat(), 10, 7, 32, 25), # 78.1% Attendance
        ("CS309", "Numerical Analysis Lab", 2.0, "O", 10.0, 29.0, 19.0, 47.0, 5,
         (today + timedelta(days=30)).isoformat(), 8, 6, 24, 21), # 87.5% Attendance
    ]
    for sub in sem6_subjects:
        add_subject(6, sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7], sub[8], sub[9], sub[10], sub[11], sub[12], sub[13])
    recompute_semester_sgpa(6)

    # 8. Historical Daily Study Logs (for integration and regression modeling)
    # Pairs of (hours_studied, marks_percentage)
    sample_study_logs = [
        ((today - timedelta(days=28)).isoformat(), 1.5, 54.0),
        ((today - timedelta(days=27)).isoformat(), 2.0, 62.0),
        ((today - timedelta(days=26)).isoformat(), 2.5, 68.0),
        ((today - timedelta(days=25)).isoformat(), 3.0, 72.0),
        ((today - timedelta(days=24)).isoformat(), 3.2, 75.0),
        ((today - timedelta(days=23)).isoformat(), 3.8, 79.0),
        ((today - timedelta(days=22)).isoformat(), 4.0, 83.0),
        ((today - timedelta(days=21)).isoformat(), 4.5, 86.0),
        ((today - timedelta(days=20)).isoformat(), 4.8, 88.0),
        ((today - timedelta(days=19)).isoformat(), 5.0, 91.0),
        ((today - timedelta(days=18)).isoformat(), 5.5, 93.0),
        ((today - timedelta(days=17)).isoformat(), 5.8, 95.0),
        ((today - timedelta(days=16)).isoformat(), 6.0, 96.0),
        ((today - timedelta(days=15)).isoformat(), 4.2, 85.0),
        ((today - timedelta(days=14)).isoformat(), 3.5, 76.0),
    ]
    for log in sample_study_logs:
        add_study_log(log[0], log[1], log[2])
