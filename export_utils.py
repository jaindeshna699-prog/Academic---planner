"""
Module: export_utils.py
Generates downloadable CSV spreadsheets and comprehensive PDF academic reports.
Uses Python standard CSV and ReportLab for styling, typography, and tables.
"""

import io
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_cgpa_csv(semesters: List[Dict[str, Any]], subjects: List[Dict[str, Any]], overall: Dict[str, Any]) -> str:
    """Generates formatted CSV string for CGPA & Semester Grade Report."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["=== SMART ACADEMIC PLANNER: CGPA & GRADE REPORT ==="])
    writer.writerow(["Generated On", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Cumulative CGPA", f"{overall.get('cgpa', 0.0):.2f}"])
    writer.writerow(["Total Registered Credits", overall.get("total_credits", 0)])
    writer.writerow(["Completed Semesters", overall.get("completed_semesters", 0)])
    writer.writerow([])

    # Semester Summary
    writer.writerow(["--- SEMESTER SUMMARY ---"])
    writer.writerow(["Semester", "Credits", "SGPA", "Status"])
    for sem in semesters:
        status = "Completed" if sem.get("is_completed", 1) else "In Progress"
        writer.writerow([f"Semester {sem['semester_num']}", sem.get("credits_registered", 0), f"{sem.get('sgpa', 0.0):.2f}", status])
    writer.writerow([])

    # Subject Details
    writer.writerow(["--- SUBJECT BREAKDOWN ---"])
    writer.writerow([
        "Semester", "Subject Code", "Subject Name", "Credits",
        "Grade", "Grade Point", "Internal Marks", "Midterm Marks", "Endsem Marks"
    ])
    for s in subjects:
        writer.writerow([
            f"Sem {s.get('semester_num')}",
            s.get("subject_code", ""),
            s.get("subject_name", ""),
            s.get("credits", 0),
            s.get("grade", ""),
            s.get("grade_point", 0),
            s.get("internal_marks", 0),
            s.get("midterm_marks", 0),
            s.get("endsem_marks", 0)
        ])

    return output.getvalue()


def generate_study_plan_csv(plan_entries: List[Dict[str, Any]]) -> str:
    """Generates CSV string for daily study plan and revision timetable."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["=== SMART ACADEMIC PLANNER: STUDY TIMETABLE ==="])
    writer.writerow(["Generated On", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow(["Plan Date", "Subject Name", "Allocated Hours", "Completed Hours", "Type", "Status", "Notes"])

    for item in plan_entries:
        item_type = "Revision Day (Ebbinghaus RK4)" if item.get("revision_day") else "Regular Study"
        writer.writerow([
            item.get("plan_date", ""),
            item.get("subject_name", ""),
            item.get("allocated_hours", 0.0),
            item.get("completed_hours", 0.0),
            item_type,
            item.get("status", "Pending"),
            item.get("notes", "")
        ])

    return output.getvalue()


def generate_attendance_csv(attendance_records: List[Dict[str, Any]], min_pct: float = 75.0) -> str:
    """Generates CSV string for attendance manager and buffer tracking."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["=== SMART ACADEMIC PLANNER: ATTENDANCE AUDIT ==="])
    writer.writerow(["Mandatory Minimum %", f"{min_pct:.1f}%"])
    writer.writerow([])
    writer.writerow([
        "Semester", "Subject Code", "Subject Name", "Total Classes",
        "Attended", "Current %", "Status", "Margin (Can Skip / Must Attend)"
    ])

    for sub in attendance_records:
        total = max(1, sub.get("total_classes", 0))
        attended = sub.get("attended_classes", 0)
        pct = (attended / total) * 100.0

        if pct >= min_pct:
            # Can skip: solve (attended / (total + x)) >= min_pct/100
            max_skip = int((attended / (min_pct / 100.0)) - total)
            margin_str = f"Can safely skip {max(0, max_skip)} class(es)"
            status = "Eligible"
        else:
            # Must attend next N classes consecutively: (attended + n)/(total + n) >= min_pct/100
            req = int(np.ceil((min_pct * total - 100.0 * attended) / (100.0 - min_pct)))
            margin_str = f"Must attend next {max(1, req)} class(es)"
            status = "Shortage (< 75%)"

        writer.writerow([
            f"Sem {sub.get('semester_num')}",
            sub.get("subject_code", ""),
            sub.get("subject_name", ""),
            total,
            attended,
            f"{pct:.1f}%",
            status,
            margin_str
        ])

    return output.getvalue()


def generate_comprehensive_pdf(
    student_info: Dict[str, str],
    overall_cgpa: Dict[str, Any],
    semesters: List[Dict[str, Any]],
    current_subjects: List[Dict[str, Any]],
    study_plan: List[Dict[str, Any]],
    min_attendance_pct: float = 75.0
) -> bytes:
    """
    Builds a PDF report incorporating:
      - Academic profile & university header
      - CGPA progress and semester history
      - Attendance analysis with warnings
      - Generated study plan timetable
      - Numerical methods verification certification
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"), # Dark Navy
        alignment=1, # Center
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        alignment=1,
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E40AF"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "BodySmall",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1F2937")
    )
    bold_cell = ParagraphStyle(
        "BoldCell",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.white
    )
    normal_cell = ParagraphStyle(
        "NormalCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1F2937")
    )
    alert_cell = ParagraphStyle(
        "AlertCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#DC2626"), # Red
        fontName="Helvetica-Bold"
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("SMART ACADEMIC PLANNER & CGPA PREDICTOR", title_style))
    story.append(Paragraph(
        f"Official Academic Performance & Numerical Analysis Lab Report | Generated: {datetime.now().strftime('%B %d, %Y')}",
        subtitle_style
    ))
    story.append(Spacer(1, 10))

    # Student Info Banner Table
    student_data = [
        [
            Paragraph(f"<b>Student Name:</b> {student_info.get('student_name', 'Student')}", body_style),
            Paragraph(f"<b>Roll No:</b> {student_info.get('roll_number', 'N/A')}", body_style)
        ],
        [
            Paragraph(f"<b>Institution:</b> {student_info.get('university', 'University')}", body_style),
            Paragraph(f"<b>Cumulative CGPA:</b> <font color='#1E40AF'><b>{overall_cgpa.get('cgpa', 0.0):.2f} / 10.0</b></font>", body_style)
        ]
    ]
    info_table = Table(student_data, colWidths=[260, 260])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F3F4F6")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 12))

    # Section 1: Semester-Wise Academic Record
    story.append(Paragraph("1. Semester-Wise SGPA Progression", h2_style))
    sem_table_data = [
        [
            Paragraph("Semester", bold_cell),
            Paragraph("Registered Credits", bold_cell),
            Paragraph("SGPA", bold_cell),
            Paragraph("Status", bold_cell)
        ]
    ]
    for sem in semesters:
        status_txt = "Completed" if sem.get("is_completed", 1) else "In Progress"
        sem_table_data.append([
            Paragraph(f"Semester {sem['semester_num']}", normal_cell),
            Paragraph(str(sem.get("credits_registered", 0)), normal_cell),
            Paragraph(f"{sem.get('sgpa', 0.0):.2f}", normal_cell),
            Paragraph(status_txt, normal_cell)
        ])
    sem_table = Table(sem_table_data, colWidths=[130, 130, 130, 130])
    sem_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sem_table)
    story.append(Spacer(1, 12))

    # Section 2: Attendance Tracking & Minimum Threshold Compliance
    story.append(Paragraph(f"2. Attendance Compliance Audit (Min Threshold: {min_attendance_pct:.1f}%)", h2_style))
    att_table_data = [
        [
            Paragraph("Subject Code & Name", bold_cell),
            Paragraph("Held", bold_cell),
            Paragraph("Attended", bold_cell),
            Paragraph("Current %", bold_cell),
            Paragraph("Action Required / Buffer", bold_cell)
        ]
    ]
    import numpy as np
    for s in current_subjects:
        tot = max(1, s.get("total_classes", 0))
        att = s.get("attended_classes", 0)
        pct = (att / tot) * 100.0
        if pct >= min_attendance_pct:
            max_skips = int((att / (min_attendance_pct / 100.0)) - tot)
            action = Paragraph(f"<font color='#059669'>Safe: Can skip {max(0, max_skips)} class(es)</font>", normal_cell)
            pct_cell = Paragraph(f"{pct:.1f}%", normal_cell)
        else:
            must_att = int(np.ceil((min_attendance_pct * tot - 100.0 * att) / (100.0 - min_attendance_pct)))
            action = Paragraph(f"<font color='#DC2626'><b>Shortage: Attend next {max(1, must_att)}!</b></font>", alert_cell)
            pct_cell = Paragraph(f"{pct:.1f}%", alert_cell)

        att_table_data.append([
            Paragraph(f"<b>{s.get('subject_code', '')}</b>: {s.get('subject_name', '')[:28]}", normal_cell),
            Paragraph(str(tot), normal_cell),
            Paragraph(str(att), normal_cell),
            pct_cell,
            action
        ])
    att_table = Table(att_table_data, colWidths=[180, 55, 60, 65, 160])
    att_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F766E")), # Teal
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(att_table)
    story.append(Spacer(1, 12))

    # Section 3: Study Plan & Revision Timetable
    if study_plan:
        story.append(Paragraph("3. Daily Study Schedule & Revision Milestones", h2_style))
        plan_table_data = [
            [
                Paragraph("Date", bold_cell),
                Paragraph("Subject Target", bold_cell),
                Paragraph("Allocated", bold_cell),
                Paragraph("Type", bold_cell),
                Paragraph("Status", bold_cell)
            ]
        ]
        # Show top 14 days
        for item in study_plan[:14]:
            t_type = "Revision (RK4 Forgetting Trigger)" if item.get("revision_day") else "Regular Concept Study"
            plan_table_data.append([
                Paragraph(str(item.get("plan_date", "")), normal_cell),
                Paragraph(str(item.get("subject_name", "")), normal_cell),
                Paragraph(f"{item.get('allocated_hours', 0.0):.1f} hrs", normal_cell),
                Paragraph(t_type, normal_cell),
                Paragraph(str(item.get("status", "Pending")), normal_cell)
            ])
        plan_table = Table(plan_table_data, colWidths=[75, 165, 65, 145, 70])
        plan_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4338CA")), # Indigo
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
            ('PADDING', (0, 0), (-1, -1), 3.5),
        ]))
        story.append(plan_table)
        story.append(Spacer(1, 12))

    # Section 4: Numerical Methods Verification Certificate
    story.append(Paragraph("4. Numerical Analysis Programming (NAP) Methods Applied", h2_style))
    num_data = [
        [Paragraph("Numerical Method", bold_cell), Paragraph("Application", bold_cell), Paragraph("Verification Baseline", bold_cell)],
        [Paragraph("Bisection & Newton-Raphson", normal_cell), Paragraph("Target SGPA / Marks Root Solver", normal_cell), Paragraph("scipy.optimize.bisect & newton", normal_cell)],
        [Paragraph("Fixed-Point Iteration", normal_cell), Paragraph("Study Hours Non-Linear Model", normal_cell), Paragraph("scipy.optimize.fixed_point", normal_cell)],
        [Paragraph("Lagrange & Hermite Interpolation", normal_cell), Paragraph("SGPA Progression Trajectory", normal_cell), Paragraph("scipy.interpolate.lagrange & Krogh", normal_cell)],
        [Paragraph("Least Squares Regression", normal_cell), Paragraph("Effort vs Marks Fit (R^2, RMSE)", normal_cell), Paragraph("numpy.polyfit & scipy.linalg.lstsq", normal_cell)],
        [Paragraph("Trapezoidal Rule", normal_cell), Paragraph("Cumulative Study Effort Integral", normal_cell), Paragraph("scipy.integrate.trapezoid", normal_cell)],
        [Paragraph("Runge-Kutta RK4 & RK45", normal_cell), Paragraph("Syllabus Retention ODE dy/dt = k(1-y)-dy", normal_cell), Paragraph("scipy.integrate.solve_ivp", normal_cell)],
        [Paragraph("Gauss & Gauss-Jordan Elimination", normal_cell), Paragraph("Linear Hour Allocation & Mark Balancing", normal_cell), Paragraph("numpy.linalg.solve", normal_cell)],
        [Paragraph("Power Method", normal_cell), Paragraph("Dominant Eigenvector Subject Ranking", normal_cell), Paragraph("numpy.linalg.eig", normal_cell)],
    ]
    num_table = Table(num_data, colWidths=[150, 210, 160])
    num_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#374151")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ('PADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(num_table)

    # Build document
    doc.build(story)
    return pdf_buffer.getvalue()
