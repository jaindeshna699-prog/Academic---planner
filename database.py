"""
Database management module for Smart Academic Planner & CGPA Predictor.
Uses SQLite for robust, offline, zero-dependency local storage.
Handles CRUD operations for semesters, subjects, attendance, study schedules,
and academic configurations.
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "academic_planner.db")


def get_connection() -> sqlite3.Connection:
    """Return an active connection with Row factory enabled."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite database tables if they do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Semesters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semesters (
                semester_num INTEGER PRIMARY KEY,
                credits_registered REAL DEFAULT 0,
                sgpa REAL DEFAULT 0.0,
                is_completed INTEGER DEFAULT 1
            )
        """)

        # Subjects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                semester_num INTEGER NOT NULL,
                subject_code TEXT NOT NULL,
                subject_name TEXT NOT NULL,
                credits REAL NOT NULL DEFAULT 3.0,
                grade TEXT DEFAULT 'A',
                grade_point REAL DEFAULT 8.0,
                internal_marks REAL DEFAULT 0.0,
                midterm_marks REAL DEFAULT 0.0,
                endsem_marks REAL DEFAULT 0.0,
                strength INTEGER DEFAULT 3, -- 1 (Weakest) to 5 (Mastered)
                exam_date TEXT,
                syllabus_total INTEGER DEFAULT 10,
                syllabus_completed INTEGER DEFAULT 0,
                FOREIGN KEY (semester_num) REFERENCES semesters(semester_num)
            )
        """)

        # Attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL UNIQUE,
                total_classes INTEGER NOT NULL DEFAULT 40,
                attended_classes INTEGER NOT NULL DEFAULT 32,
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            )
        """)

        # Study plan daily blocks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_plan (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_date TEXT NOT NULL,
                subject_id INTEGER,
                subject_name TEXT NOT NULL,
                allocated_hours REAL NOT NULL,
                completed_hours REAL DEFAULT 0.0,
                status TEXT DEFAULT 'Pending', -- 'Pending', 'In Progress', 'Completed'
                revision_day INTEGER DEFAULT 0, -- 1 if triggered by forgetting curve
                notes TEXT
            )
        """)

        # User Settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Study log entries for historical integration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_date TEXT NOT NULL,
                hours_studied REAL NOT NULL,
                marks_scored REAL
            )
        """)

        conn.commit()


# ---------------------------------------------------------------------------
# Semester & Subject Queries
# ---------------------------------------------------------------------------
def add_semester(semester_num: int, is_completed: int = 1) -> None:
    """Insert or update a semester entry."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO semesters (semester_num, credits_registered, sgpa, is_completed)
            VALUES (?, 0, 0.0, ?)
            ON CONFLICT(semester_num) DO UPDATE SET is_completed=excluded.is_completed
        """, (semester_num, is_completed))
        conn.commit()


def get_all_semesters() -> List[Dict[str, Any]]:
    """Retrieve all semesters sorted by semester number."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM semesters ORDER BY semester_num ASC")
        return [dict(row) for row in cursor.fetchall()]


def add_subject(
    semester_num: int,
    subject_code: str,
    subject_name: str,
    credits: float,
    grade: str = "A",
    grade_point: float = 8.0,
    internal_marks: float = 25.0,
    midterm_marks: float = 16.0,
    endsem_marks: float = 40.0,
    strength: int = 3,
    exam_date: Optional[str] = None,
    syllabus_total: int = 10,
    syllabus_completed: int = 4,
    total_classes: int = 40,
    attended_classes: int = 32
) -> int:
    """Add a subject and its associated attendance record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Ensure semester exists
        cursor.execute("INSERT OR IGNORE INTO semesters (semester_num) VALUES (?)", (semester_num,))

        cursor.execute("""
            INSERT INTO subjects (
                semester_num, subject_code, subject_name, credits,
                grade, grade_point, internal_marks, midterm_marks, endsem_marks,
                strength, exam_date, syllabus_total, syllabus_completed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            semester_num, subject_code, subject_name, credits,
            grade, grade_point, internal_marks, midterm_marks, endsem_marks,
            strength, exam_date, syllabus_total, syllabus_completed
        ))
        subject_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO attendance (subject_id, total_classes, attended_classes)
            VALUES (?, ?, ?)
        """, (subject_id, total_classes, attended_classes))

        conn.commit()
        recompute_semester_sgpa(semester_num)
        return subject_id


def update_subject(
    subject_id: int,
    subject_name: str,
    credits: float,
    grade: str,
    grade_point: float,
    internal_marks: float,
    midterm_marks: float,
    endsem_marks: float,
    strength: int,
    exam_date: Optional[str],
    syllabus_total: int,
    syllabus_completed: int,
    total_classes: int,
    attended_classes: int
) -> None:
    """Update subject and its attendance details."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE subjects SET
                subject_name=?, credits=?, grade=?, grade_point=?,
                internal_marks=?, midterm_marks=?, endsem_marks=?,
                strength=?, exam_date=?, syllabus_total=?, syllabus_completed=?
            WHERE id=?
        """, (
            subject_name, credits, grade, grade_point,
            internal_marks, midterm_marks, endsem_marks,
            strength, exam_date, syllabus_total, syllabus_completed,
            subject_id
        ))

        cursor.execute("""
            UPDATE attendance SET
                total_classes=?, attended_classes=?
            WHERE subject_id=?
        """, (total_classes, attended_classes, subject_id))

        # Get semester_num to recompute
        cursor.execute("SELECT semester_num FROM subjects WHERE id=?", (subject_id,))
        row = cursor.fetchone()
        conn.commit()

        if row:
            recompute_semester_sgpa(row["semester_num"])


def delete_subject(subject_id: int) -> None:
    """Delete a subject and recalculate semester SGPA."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT semester_num FROM subjects WHERE id=?", (subject_id,))
        row = cursor.fetchone()
        sem_num = row["semester_num"] if row else None

        cursor.execute("DELETE FROM attendance WHERE subject_id=?", (subject_id,))
        cursor.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
        conn.commit()

        if sem_num is not None:
            recompute_semester_sgpa(sem_num)


def get_subjects_by_semester(semester_num: int) -> List[Dict[str, Any]]:
    """Get all subjects for a given semester with attendance information."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, a.total_classes, a.attended_classes
            FROM subjects s
            LEFT JOIN attendance a ON s.id = a.subject_id
            WHERE s.semester_num = ?
            ORDER BY s.id ASC
        """, (semester_num,))
        return [dict(row) for row in cursor.fetchall()]


def get_all_subjects_with_attendance() -> List[Dict[str, Any]]:
    """Retrieve all subjects across all semesters joined with attendance."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, a.total_classes, a.attended_classes
            FROM subjects s
            LEFT JOIN attendance a ON s.id = a.subject_id
            ORDER BY s.semester_num ASC, s.id ASC
        """)
        return [dict(row) for row in cursor.fetchall()]


def recompute_semester_sgpa(semester_num: int) -> Tuple[float, float]:
    """
    Recalculate SGPA and total credits for a semester from its subjects.
    SGPA = Sum(Credits_i * GradePoint_i) / Sum(Credits_i)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT credits, grade_point
            FROM subjects
            WHERE semester_num = ?
        """, (semester_num,))
        rows = cursor.fetchall()

        total_credits = sum(r["credits"] for r in rows)
        if total_credits > 0:
            weighted_points = sum(r["credits"] * r["grade_point"] for r in rows)
            sgpa = round(weighted_points / total_credits, 2)
        else:
            sgpa = 0.0

        cursor.execute("""
            UPDATE semesters
            SET credits_registered = ?, sgpa = ?
            WHERE semester_num = ?
        """, (total_credits, sgpa, semester_num))
        conn.commit()
        return sgpa, total_credits


def calculate_overall_cgpa() -> Dict[str, Any]:
    """
    Compute cumulative GPA across all completed semesters.
    CGPA = Sum(Semester_Credits * SGPA) / Sum(Semester_Credits)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT semester_num, credits_registered, sgpa
            FROM semesters
            WHERE is_completed = 1 AND credits_registered > 0
            ORDER BY semester_num ASC
        """)
        completed_sems = [dict(r) for r in cursor.fetchall()]

        total_credits = sum(s["credits_registered"] for s in completed_sems)
        if total_credits > 0:
            total_points = sum(s["credits_registered"] * s["sgpa"] for s in completed_sems)
            cgpa = round(total_points / total_credits, 2)
        else:
            cgpa = 0.0

        return {
            "cgpa": cgpa,
            "total_credits": total_credits,
            "completed_semesters": len(completed_sems),
            "semester_history": completed_sems
        }


# ---------------------------------------------------------------------------
# Attendance Operations
# ---------------------------------------------------------------------------
def update_attendance_counts(subject_id: int, total_classes: int, attended_classes: int) -> None:
    """Update class attendance count directly."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE attendance
            SET total_classes = ?, attended_classes = ?
            WHERE subject_id = ?
        """, (total_classes, attended_classes, subject_id))
        conn.commit()


# ---------------------------------------------------------------------------
# Study Plan & Log Operations
# ---------------------------------------------------------------------------
def save_study_plan_schedule(schedule_entries: List[Dict[str, Any]]) -> None:
    """Clear existing study plan and insert new generated schedule."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM study_plan")
        for item in schedule_entries:
            cursor.execute("""
                INSERT INTO study_plan (
                    plan_date, subject_id, subject_name,
                    allocated_hours, completed_hours, status,
                    revision_day, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.get("plan_date"),
                item.get("subject_id"),
                item.get("subject_name"),
                item.get("allocated_hours", 1.0),
                item.get("completed_hours", 0.0),
                item.get("status", "Pending"),
                item.get("revision_day", 0),
                item.get("notes", "")
            ))
        conn.commit()


def get_current_study_plan() -> List[Dict[str, Any]]:
    """Retrieve full generated study plan ordered by date."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM study_plan ORDER BY plan_date ASC, id ASC")
        return [dict(row) for row in cursor.fetchall()]


def mark_study_task_completed(task_id: int, completed_hours: float) -> None:
    """Mark a daily study plan block as completed."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE study_plan
            SET completed_hours = ?, status = 'Completed'
            WHERE id = ?
        """, (completed_hours, task_id))
        conn.commit()


def get_study_logs() -> List[Dict[str, Any]]:
    """Retrieve historical daily study hours and mark logs."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM study_logs ORDER BY log_date ASC")
        return [dict(row) for row in cursor.fetchall()]


def add_study_log(log_date: str, hours: float, marks: Optional[float] = None) -> None:
    """Record historical study hours and marks."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO study_logs (log_date, hours_studied, marks_scored)
            VALUES (?, ?, ?)
        """, (log_date, hours, marks))
        conn.commit()


# ---------------------------------------------------------------------------
# Settings Operations
# ---------------------------------------------------------------------------
def get_setting(key: str, default_value: str = "") -> str:
    """Retrieve a setting by key."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM user_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else default_value


def set_setting(key: str, value: str) -> None:
    """Save or update a setting by key."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, value))
        conn.commit()


def reset_database() -> None:
    """Drop and recreate all tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS study_logs")
        cursor.execute("DROP TABLE IF EXISTS study_plan")
        cursor.execute("DROP TABLE IF EXISTS attendance")
        cursor.execute("DROP TABLE IF EXISTS subjects")
        cursor.execute("DROP TABLE IF EXISTS semesters")
        cursor.execute("DROP TABLE IF EXISTS user_settings")
        conn.commit()
    init_db()


# ---------------------------------------------------------------------------
# Dynamic Grade Scale & Attendance Policy Operations
# ---------------------------------------------------------------------------
import json
from config import DEFAULT_GRADE_SCALE, DEFAULT_MIN_ATTENDANCE_PCT


def get_grade_scale() -> Dict[str, Tuple[float, int, int]]:
    """Retrieve custom university grade scale from database or fallback to DEFAULT_GRADE_SCALE."""
    raw = get_setting("grade_scale_json", "")
    if raw:
        try:
            data = json.loads(raw)
            return {str(k): (float(v[0]), int(v[1]), int(v[2])) for k, v in data.items()}
        except Exception:
            pass
    return DEFAULT_GRADE_SCALE.copy()


def save_grade_scale(scale: Dict[str, Tuple[float, int, int]]) -> None:
    """Save custom university grade scale to database."""
    serializable = {k: [float(v[0]), int(v[1]), int(v[2])] for k, v in scale.items()}
    set_setting("grade_scale_json", json.dumps(serializable))


def get_attendance_rules() -> Dict[str, float]:
    """Retrieve configured attendance policy thresholds."""
    return {
        "min_percentage": float(get_setting("min_attendance_pct", str(DEFAULT_MIN_ATTENDANCE_PCT))),
        "warning_threshold": float(get_setting("warn_attendance_pct", "80.0")),
    }


def save_attendance_rules(min_pct: float, warn_pct: float = 80.0) -> None:
    """Save custom attendance rules to database."""
    set_setting("min_attendance_pct", str(min_pct))
    set_setting("warn_attendance_pct", str(warn_pct))

