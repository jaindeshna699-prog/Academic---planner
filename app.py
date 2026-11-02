"""
Smart Academic Planner & CGPA Predictor
Main Streamlit Application with Multi-Page Navigation and Numerical Methods Lab.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import io

from config import (
    DEFAULT_GRADE_SCALE, GRADE_CHOICES, DEFAULT_MIN_ATTENDANCE_PCT,
    DEFAULT_DAILY_STUDY_HOURS, DEFAULT_MARK_WEIGHTS, APP_TITLE, APP_SUBTITLE, APP_VERSION
)
from database import (
    init_db, add_semester, get_all_semesters, add_subject, update_subject, delete_subject,
    get_subjects_by_semester, get_all_subjects_with_attendance, recompute_semester_sgpa,
    calculate_overall_cgpa, update_attendance_counts, save_study_plan_schedule,
    get_current_study_plan, mark_study_task_completed, get_study_logs, add_study_log,
    get_setting, set_setting, reset_database, get_grade_scale, save_grade_scale,
    get_attendance_rules, save_attendance_rules
)
from sample_data import seed_sample_data
from export_utils import (
    generate_cgpa_csv, generate_study_plan_csv, generate_attendance_csv, generate_comprehensive_pdf
)

# Numerical methods
from numerical.root_finding import (
    bisection_method, newton_raphson_method, solve_required_sgpa_bisection, benchmark_root_finding
)
from numerical.fixed_point import (
    fixed_point_iteration, study_hours_model_g, benchmark_fixed_point
)
from numerical.lagrange import (
    lagrange_interpolation, predict_trajectory_lagrange, benchmark_lagrange
)
from numerical.hermite import (
    hermite_interpolation, compare_lagrange_vs_hermite, benchmark_hermite
)
from numerical.interpolation import (
    newton_polynomial_interpolation, benchmark_interpolation
)
from numerical.least_squares import (
    least_squares_fit, predict_from_least_squares, benchmark_least_squares
)
from numerical.regression import (
    linear_regression, polynomial_regression, predict_regression, benchmark_regression
)
from numerical.numerical_integration import (
    trapezoidal_rule, compute_cumulative_study_effort, benchmark_trapezoid
)
from numerical.runge_kutta import (
    rk4_solve, rk45_adaptive_solve, simulate_retention_and_schedule_revisions, benchmark_runge_kutta
)
from numerical.linear_systems import (
    gauss_elimination, gauss_jordan_elimination, solve_daily_study_hours_allocation,
    solve_mark_requirements_system, benchmark_linear_system
)
from numerical.power_method import (
    power_method, build_subject_priority_matrix, benchmark_power_method
)

# ---------------------------------------------------------------------------
# App Initialization & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Academic Planner & CGPA Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Auto-seed sample data on first launch if semesters table is empty
if len(get_all_semesters()) == 0:
    seed_sample_data()

# Custom UI CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6B7280;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.06);
    }
    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-pass {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-warn {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Navigation & Academic Profile
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"## 🎓 {APP_TITLE}")
st.sidebar.caption(f"{APP_SUBTITLE} (v{APP_VERSION})")

overall_data = calculate_overall_cgpa()
st.sidebar.markdown(f"""
<div class="metric-card">
    <div class="stat-label">Current CGPA</div>
    <div class="stat-value">{overall_data['cgpa']:.2f} <span style="font-size:1rem;font-weight:400;">/ 10.0</span></div>
    <div style="font-size:0.8rem; margin-top:6px; opacity:0.9;">
        {overall_data['total_credits']} Credits | {overall_data['completed_semesters']} Semesters
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("")

# Navigation Menu
menu_options = [
    "🎓 CGPA Calculator & History",
    "🎯 Target CGPA Planner",
    "📅 Study Plan & Timetable Generator",
    "📊 Attendance Manager",
    "⏳ Exam Countdown & Syllabus",
    "🧪 Numerical Methods Lab",
    "📥 Export Reports (CSV & PDF)",
    "⚙️ University Config & Settings"
]
selected_page = st.sidebar.radio("Navigate Application:", menu_options, index=0)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset / Reload Sample Data", use_container_width=True):
    seed_sample_data(force_reset=True)
    st.sidebar.success("Database reloaded with default academic data!")
    st.rerun()


# ===========================================================================
# PAGE 1: CGPA Calculator & History
# ===========================================================================
if selected_page == "🎓 CGPA Calculator & History":
    st.markdown("<div class='main-header'>Semester-Wise CGPA Calculator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Manage semester courses, verify grade points, compute SGPA, and track academic progression.</div>", unsafe_allow_html=True)

    semesters = get_all_semesters()
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        st.metric("Overall Cumulative CGPA", f"{overall_data['cgpa']:.2f} / 10.0")
    with col_b:
        st.metric("Total Credits Earned", f"{overall_data['total_credits']:.1f}")
    with col_c:
        completed_sems = [s for s in semesters if s["is_completed"] == 1]
        sgpa_trend = [s["sgpa"] for s in completed_sems]
        if len(sgpa_trend) >= 2:
            delta_val = round(sgpa_trend[-1] - sgpa_trend[-2], 2)
            st.metric("Latest Semester Delta", f"{sgpa_trend[-1]:.2f}", f"{delta_val:+.2f}")
        elif len(sgpa_trend) == 1:
            st.metric("Semester 1 SGPA", f"{sgpa_trend[0]:.2f}")

    tab_overview, tab_manage, tab_add_sub = st.tabs(["📈 SGPA Trajectory & Semesters", "✏️ Edit / Manage Subjects", "➕ Add New Subject"])

    with tab_overview:
        col_plot, col_table = st.columns([3, 2])
        with col_plot:
            if len(completed_sems) > 0:
                fig, ax = plt.subplots(figsize=(6, 3.2))
                sem_nums = [s["semester_num"] for s in completed_sems]
                sgpas = [s["sgpa"] for s in completed_sems]
                ax.plot(sem_nums, sgpas, marker="o", color="#1E3A8A", linewidth=2.5, markersize=7, label="Actual SGPA")
                ax.axhline(y=overall_data['cgpa'], color="#EF4444", linestyle="--", linewidth=1.5, label=f"Cumulative CGPA ({overall_data['cgpa']:.2f})")
                ax.set_ylim(0, 10.5)
                ax.set_xticks(sem_nums)
                ax.set_xlabel("Semester")
                ax.set_ylabel("SGPA (10-Point Scale)")
                ax.set_title("SGPA Trajectory Across Semesters", fontsize=11, fontweight="bold")
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No completed semesters recorded yet. Add your first semester below!")

        with col_table:
            st.markdown("#### Semester Summary")
            sem_df = pd.DataFrame(semesters)
            if not sem_df.empty:
                sem_df["Status"] = sem_df["is_completed"].apply(lambda x: "Completed" if x == 1 else "In Progress")
                sem_display = sem_df[["semester_num", "credits_registered", "sgpa", "Status"]].rename(
                    columns={"semester_num": "Semester", "credits_registered": "Credits", "sgpa": "SGPA"}
                )
                st.dataframe(sem_display, use_container_width=True, hide_index=True)

    with tab_manage:
        sem_list = [s["semester_num"] for s in semesters]
        if sem_list:
            chosen_sem = st.selectbox("Select Semester to View/Edit:", sem_list, index=len(sem_list)-1)
            subjects = get_subjects_by_semester(chosen_sem)
            if subjects:
                for sub in subjects:
                    with st.expander(f"{sub['subject_code']} - {sub['subject_name']} (Credits: {sub['credits']}, Grade: {sub['grade']})"):
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            new_name = st.text_input("Name", sub["subject_name"], key=f"name_{sub['id']}")
                            new_credits = st.number_input("Credits", min_value=1.0, max_value=10.0, value=float(sub["credits"]), step=0.5, key=f"cred_{sub['id']}")
                        with c2:
                            new_grade = st.selectbox("Grade", GRADE_CHOICES, index=GRADE_CHOICES.index(sub["grade"]) if sub["grade"] in GRADE_CHOICES else 2, key=f"gr_{sub['id']}")
                            new_gp = DEFAULT_GRADE_SCALE[new_grade][0]
                            st.write(f"Grade Point: **{new_gp}**")
                        with c3:
                            new_strength = st.slider("Strength (1=Weak, 5=Mastered)", 1, 5, int(sub["strength"]), key=f"str_{sub['id']}")
                            new_exam = st.date_input("Exam Date", value=datetime.strptime(sub["exam_date"], "%Y-%m-%d").date() if sub.get("exam_date") else date.today() + timedelta(days=15), key=f"ex_{sub['id']}")
                        with c4:
                            new_tot_cls = st.number_input("Total Classes", min_value=1, value=int(sub["total_classes"]), key=f"tc_{sub['id']}")
                            new_att_cls = st.number_input("Attended Classes", min_value=0, max_value=int(new_tot_cls), value=int(sub["attended_classes"]), key=f"ac_{sub['id']}")

                        b_save, b_del = st.columns([1, 1])
                        with b_save:
                            if st.button("💾 Update Subject", key=f"up_{sub['id']}"):
                                update_subject(
                                    sub["id"], new_name, new_credits, new_grade, new_gp,
                                    sub["internal_marks"], sub["midterm_marks"], sub["endsem_marks"],
                                    new_strength, new_exam.isoformat(), sub["syllabus_total"],
                                    sub["syllabus_completed"], new_tot_cls, new_att_cls
                                )
                                st.success("Subject updated successfully!")
                                st.rerun()
                        with b_del:
                            if st.button("🗑️ Delete Subject", key=f"del_{sub['id']}", type="secondary"):
                                delete_subject(sub["id"])
                                st.warning("Subject deleted.")
                                st.rerun()
            else:
                st.info("No subjects recorded for this semester yet.")

    with tab_add_sub:
        st.markdown("#### Add New Subject")
        with st.form("add_subject_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                f_sem = st.number_input("Semester Number", min_value=1, max_value=10, value=6)
                f_code = st.text_input("Subject Code (e.g., CS310)", "CS310")
                f_name = st.text_input("Subject Name", "High Performance Computing")
            with c2:
                f_credits = st.number_input("Credits", min_value=1.0, max_value=8.0, value=4.0, step=0.5)
                f_grade = st.selectbox("Expected / Earned Grade", GRADE_CHOICES, index=1)
                f_strength = st.slider("Subject Strength (1=Weak, 5=Mastered)", 1, 5, 3)
            with c3:
                f_tot_cls = st.number_input("Total Scheduled Classes", min_value=1, value=40)
                f_att_cls = st.number_input("Classes Attended", min_value=0, value=32)
                f_exam = st.date_input("Exam Date", value=date.today() + timedelta(days=20))

            submitted = st.form_submit_button("➕ Add Subject to Semester")
            if submitted:
                f_gp = DEFAULT_GRADE_SCALE[f_grade][0]
                add_subject(
                    semester_num=f_sem,
                    subject_code=f_code,
                    subject_name=f_name,
                    credits=f_credits,
                    grade=f_grade,
                    grade_point=f_gp,
                    strength=f_strength,
                    exam_date=f_exam.isoformat(),
                    total_classes=f_tot_cls,
                    attended_classes=f_att_cls
                )
                st.success(f"Added {f_code} - {f_name} to Semester {f_sem}!")
                st.rerun()


# ===========================================================================
# PAGE 2: Target CGPA Planner
# ===========================================================================
elif selected_page == "🎯 Target CGPA Planner":
    st.markdown("<div class='main-header'>Target CGPA Planner</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Mathematically calculate the exact SGPA needed in remaining semesters to achieve your goal.</div>", unsafe_allow_html=True)

    curr_cgpa = overall_data["cgpa"]
    curr_credits = overall_data["total_credits"]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Input Target Goal")
        target_goal = st.number_input("Desired Target CGPA", min_value=0.0, max_value=10.0, value=max(round(curr_cgpa + 0.3, 2), 8.8), step=0.05)
        remaining_credits = st.number_input("Upcoming Semester Credits (or total remaining credits)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        num_remaining_sems = st.slider("Distribute across remaining semesters", min_value=1, max_value=5, value=1)

    with col2:
        st.markdown("#### Academic Status")
        st.metric("Current Cumulative CGPA", f"{curr_cgpa:.2f}")
        st.metric("Completed Credits", f"{curr_credits:.1f}")
        st.metric("Total Credits at Target", f"{curr_credits + remaining_credits:.1f}")

    st.markdown("---")
    st.markdown("### Numerical Root-Finding Determination")

    # Solve via Bisection Method from scratch: f(SGPA) = CGPA(SGPA) - Target = 0
    try:
        bisect_res = solve_required_sgpa_bisection(
            current_cgpa=curr_cgpa,
            completed_credits=curr_credits,
            upcoming_credits=remaining_credits,
            target_cgpa=target_goal,
            min_sgpa=0.0,
            max_sgpa=10.0,
            tol=1e-6
        )
        required_sgpa = bisect_res["root"]
        is_feasible = (0.0 <= required_sgpa <= 10.0)

        c_res1, c_res2 = st.columns([2, 1])
        with c_res1:
            if is_feasible:
                st.success(f"🎯 **Required SGPA:** You need an average SGPA of **{required_sgpa:.2f}** in the upcoming {remaining_credits:.0f} credits to reach a CGPA of **{target_goal:.2f}**.")
            else:
                st.error(f"⚠️ **Mathematically Infeasible:** To achieve {target_goal:.2f}, you would need an SGPA of **{required_sgpa:.2f}**, which exceeds the maximum achievable 10.0 scale.")

            # Sensitivity Analysis Table
            st.markdown("#### Sensitivity Analysis (Credits vs Required SGPA)")
            cred_scenarios = [16.0, 18.0, 20.0, 22.0, 24.0]
            sens_data = []
            for cr in cred_scenarios:
                try:
                    s_res = solve_required_sgpa_bisection(curr_cgpa, curr_credits, cr, target_goal)
                    req = s_res["root"]
                    feas = "✅ Feasible" if 0 <= req <= 10.0 else "❌ Exceeds 10.0"
                    sens_data.append({"Upcoming Credits": cr, "Required SGPA": round(req, 2), "Feasibility": feas})
                except Exception:
                    sens_data.append({"Upcoming Credits": cr, "Required SGPA": "N/A", "Feasibility": "Out of Bracket"})
            st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)

        with c_res2:
            st.markdown("#### Bisection Convergence Details")
            st.write(f"**Iterations:** {bisect_res['iterations']}")
            st.write(f"**Final Error:** {bisect_res['f_root']:.2e}")
            with st.expander("Show Root-Finding Iterations"):
                st.dataframe(pd.DataFrame(bisect_res["history"]), use_container_width=True)

    except ValueError as e:
        st.warning(f"Target calculation note: {str(e)}")


# ===========================================================================
# PAGE 3: Study Plan & Timetable Generator
# ===========================================================================
elif selected_page == "📅 Study Plan & Timetable Generator":
    st.markdown("<div class='main-header'>Study Plan & Timetable Generator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Powered by the Power Method (Eigenvector Subject Ranking), Gauss Elimination (Hour Allocation), and RK4 (Revision Scheduling).</div>", unsafe_allow_html=True)

    subjects = get_all_subjects_with_attendance()
    # Filter to active/in-progress semester (default Sem 6)
    active_subjects = [s for s in subjects if s["semester_num"] == 6]
    if not active_subjects:
        active_subjects = subjects[-5:] if subjects else []

    if not active_subjects:
        st.info("Please add subjects first in the CGPA Calculator tab.")
    else:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            total_daily_hours = st.slider("Daily Available Study Hours", min_value=1.0, max_value=10.0, value=float(get_setting("daily_study_hours", "4.0")), step=0.5)
            set_setting("daily_study_hours", str(total_daily_hours))
        with col_ctrl2:
            plan_duration_days = st.slider("Plan Schedule Horizon (Days)", min_value=7, max_value=30, value=14, step=7)
        with col_ctrl3:
            enable_rk4_revisions = st.checkbox("Enable Ebbinghaus Forgetting Curve Revision Triggers (RK4)", value=True)

        st.markdown("---")

        # Step 1: Power Method Ranking
        sub_names = [s["subject_name"] for s in active_subjects]
        credits_arr = [float(s["credits"]) for s in active_subjects]
        strength_arr = [int(s["strength"]) for s in active_subjects]
        days_arr = []
        today = date.today()
        for s in active_subjects:
            if s.get("exam_date"):
                d_val = (datetime.strptime(s["exam_date"], "%Y-%m-%d").date() - today).days
                days_arr.append(max(1, d_val))
            else:
                days_arr.append(20)

        pending_arr = []
        for s in active_subjects:
            tot = max(1, s.get("syllabus_total", 10))
            comp = s.get("syllabus_completed", 0)
            pending_arr.append(round(((tot - comp) / tot) * 100.0, 1))

        # Run Power Method
        power_res = build_subject_priority_matrix(
            subject_names=sub_names,
            credits_list=credits_arr,
            strength_ratings=strength_arr,
            days_to_exam=days_arr,
            syllabus_pending_pct=pending_arr
        )

        col_rank, col_alloc = st.columns([1, 1])
        with col_rank:
            st.markdown("#### 1. Subject Priority Ranking (Dominant Eigenvector)")
            rank_df = pd.DataFrame(power_res["ranking"], columns=["Subject", "Priority Weight"])
            rank_df["Priority Weight %"] = (rank_df["Priority Weight"] * 100.0).round(1).astype(str) + "%"
            st.dataframe(rank_df[["Subject", "Priority Weight %"]], use_container_width=True, hide_index=True)

        # Step 2: Linear System (Gauss Elimination) for Hour Allocation
        with col_alloc:
            st.markdown("#### 2. Daily Hour Allocation (Gauss Elimination)")
            diff_weights = [(6 - s) for s in strength_arr]
            gauss_res = solve_daily_study_hours_allocation(sub_names, credits_arr, diff_weights, total_daily_hours)
            alloc_df = pd.DataFrame(list(gauss_res["allocation"].items()), columns=["Subject", "Hours / Day"])
            st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        # Step 3: Runge-Kutta Revision Schedule
        st.markdown("#### 3. Day-by-Day Study Schedule & Milestones")
        rk_sim = simulate_retention_and_schedule_revisions(
            initial_mastery=0.25,
            learning_rate=0.35,
            decay_rate=0.08,
            days=plan_duration_days,
            revision_threshold=0.70
        )
        rev_days = set(rk_sim["revision_days"]) if enable_rk4_revisions else set()

        # Build schedule entries
        schedule = []
        curr_sub_idx = 0
        sorted_subs = [item[0] for item in power_res["ranking"]]

        for d in range(1, plan_duration_days + 1):
            p_date = (today + timedelta(days=d)).isoformat()
            is_revision = (d in rev_days)

            # Assign primary subject for the day based on eigenvector priority order
            assigned_sub = sorted_subs[curr_sub_idx % len(sorted_subs)]
            sub_id = next((s["id"] for s in active_subjects if s["subject_name"] == assigned_sub), None)
            hours = gauss_res["allocation"].get(assigned_sub, total_daily_hours / len(sorted_subs))

            notes = "Scheduled Revision Day: retention decay detected below threshold by RK4 ODE." if is_revision else "Focus on core problem sets & chapter exercises."
            schedule.append({
                "plan_date": p_date,
                "subject_id": sub_id,
                "subject_name": assigned_sub,
                "allocated_hours": hours,
                "completed_hours": 0.0,
                "status": "Pending",
                "revision_day": 1 if is_revision else 0,
                "notes": notes
            })
            curr_sub_idx += 1

        # Save to database
        if st.button("💾 Generate & Save Timetable to Database", use_container_width=True):
            save_study_plan_schedule(schedule)
            st.success(f"Successfully generated and persisted {len(schedule)}-day study timetable!")
            st.rerun()

        # Display saved timetable
        saved_plan = get_current_study_plan()
        if saved_plan:
            plan_df = pd.DataFrame(saved_plan)
            plan_df["Type"] = plan_df["revision_day"].apply(lambda x: "🔄 Revision Milestone" if x == 1 else "📖 Core Concept")
            st.dataframe(
                plan_df[["plan_date", "subject_name", "allocated_hours", "Type", "status", "notes"]].rename(
                    columns={"plan_date": "Date", "subject_name": "Subject", "allocated_hours": "Allocated Hrs", "status": "Status", "notes": "Notes"}
                ),
                use_container_width=True,
                hide_index=True
            )


# ===========================================================================
# PAGE 4: Attendance Manager
# ===========================================================================
elif selected_page == "📊 Attendance Manager":
    st.markdown("<div class='main-header'>Attendance Manager & Buffer Tracker</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Configurable mandatory threshold tracking with exact skip margins and deficit recovery calculations.</div>", unsafe_allow_html=True)

    min_pct = st.number_input(
        "University Minimum Mandatory Attendance %",
        min_value=50.0,
        max_value=95.0,
        value=float(get_setting("min_attendance_pct", str(DEFAULT_MIN_ATTENDANCE_PCT))),
        step=1.0
    )
    set_setting("min_attendance_pct", str(min_pct))

    subjects = get_all_subjects_with_attendance()
    active_subjects = [s for s in subjects if s["semester_num"] == 6]
    if not active_subjects:
        active_subjects = subjects

    if active_subjects:
        # High level summary
        total_held_all = sum(s.get("total_classes", 0) for s in active_subjects)
        total_att_all = sum(s.get("attended_classes", 0) for s in active_subjects)
        overall_att_pct = (total_att_all / total_held_all * 100.0) if total_held_all > 0 else 100.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Classes Held", total_held_all)
        with c2:
            st.metric("Total Attended", total_att_all)
        with c3:
            st.metric("Aggregate Attendance %", f"{overall_att_pct:.1f}%", f"{overall_att_pct - min_pct:+.1f}%")

        st.markdown("---")
        st.markdown("### Subject-by-Subject Attendance Audit")

        for s in active_subjects:
            tot = max(1, s.get("total_classes", 0))
            att = s.get("attended_classes", 0)
            pct = (att / tot) * 100.0

            # Mathematical attendance calculations
            if pct >= min_pct:
                # How many can skip: (att / (tot + x)) >= (min_pct/100)
                max_skip = int((att / (min_pct / 100.0)) - tot)
                badge = f"<span class='badge-pass'>SAFE ({pct:.1f}%)</span>"
                advice = f"🟢 **You can safely skip up to {max(0, max_skip)} more class(es)** without falling below {min_pct:.0f}%."
            else:
                # How many must attend consecutively: (att + n)/(tot + n) >= (min_pct/100)
                must_attend = int(np.ceil((min_pct * tot - 100.0 * att) / (100.0 - min_pct)))
                badge = f"<span class='badge-warn'>SHORTAGE ({pct:.1f}%)</span>"
                advice = f"🔴 **Deficit alert! You must attend the next {max(1, must_attend)} class(es)** consecutively to restore {min_pct:.0f}% eligibility."

            with st.container():
                st.markdown(f"#### {s['subject_code']}: {s['subject_name']} {badge}", unsafe_allow_html=True)
                col_bar, col_act = st.columns([3, 2])
                with col_bar:
                    st.progress(min(1.0, max(0.0, pct / 100.0)))
                    st.markdown(advice)
                    st.caption(f"Record: {att} attended out of {tot} held.")

                with col_act:
                    c_pres, c_abs = st.columns(2)
                    with c_pres:
                        if st.button("✅ +1 Present", key=f"att_p_{s['id']}"):
                            update_attendance_counts(s['id'], tot + 1, att + 1)
                            st.rerun()
                    with c_abs:
                        if st.button("❌ +1 Absent", key=f"att_a_{s['id']}"):
                            update_attendance_counts(s['id'], tot + 1, att)
                            st.rerun()
                st.markdown("---")


# ===========================================================================
# PAGE 5: Exam Countdown & Syllabus Progress
# ===========================================================================
elif selected_page == "⏳ Exam Countdown & Syllabus":
    st.markdown("<div class='main-header'>Exam Countdown & Syllabus Progress</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time countdown clocks, syllabus chapter trackers, and visual completion gauges.</div>", unsafe_allow_html=True)

    subjects = get_all_subjects_with_attendance()
    active_subjects = [s for s in subjects if s["semester_num"] == 6]
    if not active_subjects:
        active_subjects = subjects

    today = date.today()

    col_cd, col_syl = st.columns([1, 1])
    with col_cd:
        st.markdown("### ⏳ Exam Countdown")
        countdown_data = []
        for s in active_subjects:
            if s.get("exam_date"):
                exam_d = datetime.strptime(s["exam_date"], "%Y-%m-%d").date()
                diff = (exam_d - today).days
                urgency = "🔥 Urgent" if diff <= 10 else ("⚡ Approaching" if diff <= 20 else "🟢 Ample Time")
                countdown_data.append({
                    "Subject": s["subject_name"],
                    "Date": s["exam_date"],
                    "Days Left": f"{diff} days" if diff >= 0 else "Exam Completed",
                    "Status": urgency
                })
        if countdown_data:
            st.dataframe(pd.DataFrame(countdown_data), use_container_width=True, hide_index=True)
        else:
            st.info("No upcoming exam dates set. Set exam dates in the CGPA Calculator tab.")

    with col_syl:
        st.markdown("### 📚 Syllabus Completion Tracker")
        for s in active_subjects:
            tot = max(1, s.get("syllabus_total", 10))
            comp = s.get("syllabus_completed", 0)
            pct = (comp / tot) * 100.0
            st.write(f"**{s['subject_name']}** ({comp} of {tot} modules/topics completed - {pct:.0f}%)")
            st.progress(min(1.0, max(0.0, pct / 100.0)))

    st.markdown("---")
    st.markdown("### 📊 Progress Distribution Chart")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = [s["subject_code"] for s in active_subjects]
    completed_pcts = [(s["syllabus_completed"] / max(1, s["syllabus_total"])) * 100.0 for s in active_subjects]
    bars = ax.bar(names, completed_pcts, color="#3B82F6", edgecolor="#1E3A8A", width=0.5)
    ax.set_ylabel("Completion %")
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.0f}%", ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', linestyle=":", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)


# ===========================================================================
# PAGE 6: Numerical Methods Lab
# ===========================================================================
elif selected_page == "🧪 Numerical Methods Lab":
    st.markdown("<div class='main-header'>Numerical Methods Lab (NAP Course Benchmarks)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>From-scratch algorithms paired with SciPy/NumPy implementations, convergence logs, and error metrics.</div>", unsafe_allow_html=True)

    method_tabs = st.tabs([
        "1. Bisection & Newton",
        "2. Fixed-Point",
        "3. Lagrange Interp",
        "4. Hermite Interp",
        "5. Least Squares",
        "6. Trapezoidal Rule",
        "7. Runge-Kutta (RK4/45)",
        "8. Gauss Elimination",
        "9. Power Method"
    ])

    # 1. Root Finding Tab
    with method_tabs[0]:
        st.markdown("### Experiment 1 & 2: Root-Finding (Bisection & Newton-Raphson)")
        st.caption("Application: Solve required marks or SGPA f(x) = 0 to hit university target.")

        c1, c2 = st.columns(2)
        with c1:
            a_val = st.number_input("Left Bracket 'a'", value=0.0, step=0.5)
            b_val = st.number_input("Right Bracket 'b'", value=10.0, step=0.5)
            tol_root = st.number_input("Tolerance", value=1e-6, format="%.1e")
        with c2:
            st.markdown("**Test Function:** $f(x) = x^3 - 2x - 5$ (Root $\\approx 2.094551$)")
            f_test = lambda x: x**3 - 2.0*x - 5.0
            df_test = lambda x: 3.0*x**2 - 2.0

        if st.button("🚀 Run Root-Finding Comparison", key="btn_root"):
            bench = benchmark_root_finding(f_test, a_val, b_val, x0=(a_val + b_val)/2.0, tol=tol_root)
            st.success("Execution complete!")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("#### Bisection Method")
                st.write(f"**From-Scratch Root:** `{bench['scratch_bisect']['root']:.8f}` ({bench['scratch_bisect']['iterations']} iters)")
                st.write(f"**SciPy optimize.bisect:** `{bench['scipy_bisect_root']:.8f}`")
                st.write(f"**Absolute Error:** `{bench['bisect_difference']:.2e}`")
                st.dataframe(pd.DataFrame(bench["scratch_bisect"]["history"]), use_container_width=True)

            with col_m2:
                st.markdown("#### Newton-Raphson Method")
                st.write(f"**From-Scratch Root:** `{bench['scratch_newton']['root']:.8f}` ({bench['scratch_newton']['iterations']} iters)")
                st.write(f"**SciPy optimize.newton:** `{bench['scipy_newton_root']:.8f}`")
                st.write(f"**Absolute Error:** `{bench['newton_difference']:.2e}`")
                st.dataframe(pd.DataFrame(bench["scratch_newton"]["history"]), use_container_width=True)

    # 2. Fixed Point Tab
    with method_tabs[1]:
        st.markdown("### Experiment 3: Fixed-Point Iteration")
        st.caption("Application: Diminishing returns in study hours vs expected marks model x = g(x).")

        c1, c2 = st.columns(2)
        with c1:
            tgt_m = st.slider("Target Marks", 50.0, 95.0, 85.0, step=1.0)
            x0_fp = st.number_input("Initial Guess Hours x0", value=2.0, step=0.5)
        with c2:
            g_model = study_hours_model_g(target_marks=tgt_m, alpha=0.25, max_marks=100.0)
            st.info("Mapping: $M(x) = 100(1 - e^{-0.25x}) = \\text{Target}$. Formulated as contraction mapping $x = g(x)$.")

        if st.button("🚀 Run Fixed-Point Benchmark", key="btn_fp"):
            bench_fp = benchmark_fixed_point(g_model, x0=x0_fp, tol=1e-6)
            st.write(f"**From-Scratch Required Hours:** `{bench_fp['scratch_result']['root']:.4f}` hours")
            st.write(f"**SciPy optimize.fixed_point:** `{bench_fp['scipy_root']:.4f}` hours")
            st.write(f"**Difference:** `{bench_fp['difference']:.2e}`")
            st.write(f"**Estimated Contraction |g'(x)|:** `{bench_fp['scratch_result']['g_prime_estimate']:.4f}` (< 1.0 confirms convergence)")
            st.dataframe(pd.DataFrame(bench_fp["scratch_result"]["history"]), use_container_width=True)

    # 3. Interpolation Tab (Lagrange & Newton)
    with method_tabs[2]:
        st.markdown("### Experiment 7: Polynomial Interpolation (Lagrange & Newton Divided Differences)")
        st.caption("Application: Predict upcoming or missing semester SGPA trajectory from discrete data points.")

        sems = [1.0, 2.0, 3.0, 4.0, 5.0]
        sgpa_vals = [8.8, 8.6, 8.4, 8.7, 8.9]
        eval_pt = st.number_input("Predict Semester:", value=6.0, step=1.0)

        c_lag, c_newt = st.columns(2)
        with c_lag:
            if st.button("🚀 Run Lagrange Interpolation", key="btn_lag"):
                bench_lag = benchmark_lagrange(sems, sgpa_vals, eval_pt)
                st.write(f"**From-Scratch Predicted SGPA:** `{bench_lag['scratch_result']:.4f}`")
                st.write(f"**SciPy interpolate.lagrange:** `{bench_lag['scipy_result']:.4f}`")
                st.write(f"**Absolute Difference:** `{bench_lag['difference']:.2e}`")
                st.markdown("#### Basis Polynomials Breakdown $L_i(x)$")
                st.dataframe(pd.DataFrame(bench_lag["basis_details"]), use_container_width=True)

        with c_newt:
            if st.button("🚀 Run Newton Divided Difference Interpolation", key="btn_newt"):
                res_newt = newton_polynomial_interpolation(sems, sgpa_vals, eval_pt)
                bench_interp = benchmark_interpolation(sems, sgpa_vals, eval_pt)
                st.write(f"**Newton Predicted SGPA:** `{res_newt['y_pred']:.4f}`")
                st.write(f"**Newton vs Lagrange Discrepancy:** `{bench_interp['internal_discrepancy']:.2e}` (Exact match)")
                st.write(f"**SciPy Agreement:** `{bench_interp['scipy_status']}`")
                st.markdown("#### Divided Difference Tableau $f[x_0, \\dots, x_k]$")
                st.dataframe(pd.DataFrame(res_newt["divided_diff_table"]), use_container_width=True)

    # 4. Hermite Tab
    with method_tabs[3]:
        st.markdown("### Experiment 8: Hermite Interpolation (Slope-Guided)")
        st.caption("Application: Uses semester improvement rate f'(x) to prevent unconstrained polynomial overshoot.")

        x_nodes = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_nodes = [8.8, 8.6, 8.4, 8.7, 8.9]
        d_nodes = [-0.2, -0.2, 0.3, 0.2, 0.2]  # Trend momentum

        if st.button("🚀 Compare Lagrange vs Hermite", key="btn_herm"):
            comp = compare_lagrange_vs_hermite(x_nodes, y_nodes, x_eval=6.0, d_nodes=d_nodes)
            bench_herm = benchmark_hermite(x_nodes, y_nodes, x_eval=6.0, d_nodes=d_nodes)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Lagrange Prediction (Sem 6)", f"{comp['lagrange_prediction']:.3f}")
            with col2:
                st.metric("Hermite Prediction (Sem 6)", f"{comp['hermite_prediction']:.3f}")
            st.write(f"**Difference:** `{comp['absolute_difference']:.4f}`")
            st.write(f"**SciPy Benchmark Agreement:** `{bench_herm['scipy_status']}` (Diff: `{bench_herm['difference']:.2e}`)")

    # 5. Least Squares Regression Tab
    with method_tabs[4]:
        st.markdown("### Experiment 9: Least Squares Linear & Polynomial Regression")
        st.caption("Application: Fit study-hours vs marks percentage with residual analysis and R^2 score.")

        logs = get_study_logs()
        if logs:
            x_arr = [float(l["hours_studied"]) for l in logs]
            y_arr = [float(l["marks_scored"]) for l in logs]
        else:
            x_arr = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
            y_arr = [55.0, 64.0, 72.0, 81.0, 90.0, 95.0]

        deg = st.selectbox("Polynomial Degree", [1, 2], index=0)

        if st.button("🚀 Run Least Squares Fit", key="btn_ls"):
            bench_ls = benchmark_regression(x_arr, y_arr, degree=deg)
            st.write(f"**From-Scratch Coefficients:** `{bench_ls['scratch_coefficients']}`")
            st.write(f"**NumPy polyfit Coefficients:** `{bench_ls['numpy_coefficients']}`")
            st.write(f"**Max Error Difference:** `{bench_ls['max_difference']:.2e}`")
            st.metric("Coefficient of Determination R²", f"{bench_ls['r_squared']:.4f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{bench_ls['rmse']:.3f}")

            # Plot regression line
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.scatter(x_arr, y_arr, color="#EF4444", label="Observed Data", zorder=3)
            x_fine = np.linspace(min(x_arr), max(x_arr), 100)
            if deg == 1:
                m, c = bench_ls['scratch_coefficients']
                y_fine = m * x_fine + c
            else:
                a, b, c = bench_ls['scratch_coefficients']
                y_fine = a * x_fine**2 + b * x_fine + c
            ax.plot(x_fine, y_fine, color="#1E3A8A", linewidth=2, label=f"Least Squares Fit (deg={deg})")
            ax.set_xlabel("Daily Study Hours")
            ax.set_ylabel("Marks % Scored")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()
            st.pyplot(fig)

    # 6. Trapezoidal Rule Tab
    with method_tabs[5]:
        st.markdown("### Experiment 10: Composite Trapezoidal Rule")
        st.caption("Application: Integrate daily study hours curve to compute total effort (hours × days) over a semester.")

        daily_study = [2.0, 3.5, 4.0, 1.5, 5.0, 4.5, 6.0]
        days_axis = [1, 2, 3, 4, 5, 6, 7]

        if st.button("🚀 Integrate Cumulative Effort", key="btn_trap"):
            bench_trap = benchmark_trapezoid(daily_study, x_values=days_axis)
            st.write(f"**From-Scratch Total Effort (Integral):** `{bench_trap['scratch_integral']:.4f}` hours")
            st.write(f"**SciPy integrate.trapezoid:** `{bench_trap['scipy_integral']:.4f}` hours")
            st.write(f"**Absolute Difference:** `{bench_trap['difference']:.2e}`")
            st.metric("Average Daily Effort", f"{bench_trap['average_effort']:.2f} hrs/day")
            st.dataframe(pd.DataFrame(bench_trap["interval_table"]), use_container_width=True)

    # 7. Runge-Kutta Tab
    with method_tabs[6]:
        st.markdown("### Experiment 11 & 12: Runge-Kutta Methods (RK4 & Adaptive RK45)")
        st.caption("Application: Solve syllabus retention and forgetting ODE dy/dt = k(1-y) - d*y.")

        k_rate = st.slider("Learning Rate k", 0.1, 0.8, 0.35, step=0.05)
        d_rate = st.slider("Decaying/Forgetting Rate d", 0.01, 0.20, 0.08, step=0.01)

        if st.button("🚀 Run RK4 / RK45 Simulation", key="btn_rk"):
            dydt_model = lambda t, y: k_rate * (1.0 - y) - d_rate * y
            bench_rk = benchmark_runge_kutta(dydt_model, y0=0.20, t_span=(0.0, 30.0), h=0.5)

            st.write(f"**From-Scratch RK4 Final Retention:** `{bench_rk['scratch_rk4_final']*100:.2f}%`")
            st.write(f"**From-Scratch RK45 Adaptive Final:** `{bench_rk['scratch_rk45_final']*100:.2f}%`")
            st.write(f"**SciPy solve_ivp (RK45):** `{bench_rk['scipy_final']*100:.2f}%`")
            st.write(f"**Difference:** `{bench_rk['difference']:.2e}`")

    # 8. Gauss Elimination Tab
    with method_tabs[7]:
        st.markdown("### Experiment 4 & 5: Gauss & Gauss-Jordan Elimination")
        st.caption("Application: Solve multi-subject study hour distribution and mark requirements system.")

        A_mat = [[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]]
        b_vec = [1.0, -2.0, 0.0]

        if st.button("🚀 Solve Linear System with Partial Pivoting", key="btn_gauss"):
            bench_lin = benchmark_linear_system(A_mat, b_vec)
            st.write(f"**Gauss Elimination Solution:** `{bench_lin['scratch_gauss_solution']}`")
            st.write(f"**Gauss-Jordan Solution:** `{bench_lin['scratch_jordan_solution']}`")
            st.write(f"**NumPy linalg.solve:** `{bench_lin['numpy_solution']}`")
            st.write(f"**Max Error:** `{bench_lin['max_difference']:.2e}`")
            st.markdown("#### Step-by-Step Augmented Matrix Elimination")
            for step in bench_lin["step_history"][:4]:
                st.write(f"**{step['stage']}:** {step['description']}")
                st.write(pd.DataFrame(step["matrix"]))

    # 9. Power Method Tab
    with method_tabs[8]:
        st.markdown("### Experiment 6: Power Method for Dominant Eigenvalue/Vector")
        st.caption("Application: Dominant eigenvector yields the subject study priority weights (Perron-Frobenius theorem).")

        A_sym = [[4.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 2.0, 5.0]]

        if st.button("🚀 Compute Dominant Eigenvector", key="btn_power"):
            bench_pow = benchmark_power_method(A_sym)
            st.write(f"**From-Scratch Dominant Eigenvalue:** `{bench_pow['scratch_eigenvalue']:.6f}`")
            st.write(f"**NumPy linalg.eig Dominant Eigenvalue:** `{bench_pow['numpy_eigenvalue']:.6f}`")
            st.write(f"**Eigenvalue Difference:** `{bench_pow['eigenvalue_difference']:.2e}`")
            st.write(f"**Normalized Priority Weights:** `{bench_pow['scratch_weights']}`")
            st.write(f"**Iterations to Converge:** `{bench_pow['iterations']}`")


# ===========================================================================
# PAGE 7: Export Reports (CSV & PDF)
# ===========================================================================
elif selected_page == "📥 Export Reports (CSV & PDF)":
    st.markdown("<div class='main-header'>Export Academic Reports</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Download verified grade transcripts, attendance audits, and study plans in CSV and styled PDF formats.</div>", unsafe_allow_html=True)

    semesters = get_all_semesters()
    subjects = get_all_subjects_with_attendance()
    study_plan = get_current_study_plan()
    student_info = {
        "student_name": get_setting("student_name", "Arjun Sharma"),
        "roll_number": get_setting("roll_number", "23CSB042"),
        "university": get_setting("university", "National Institute of Technology / University NAP Dept.")
    }

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 PDF Comprehensive Report")
        st.write("Generates an academic transcript with CGPA history, attendance audit, study timetable, and numerical verification table.")

        if st.button("Generate & Prepare PDF", use_container_width=True):
            pdf_bytes = generate_comprehensive_pdf(
                student_info=student_info,
                overall_cgpa=overall_data,
                semesters=semesters,
                current_subjects=[s for s in subjects if s["semester_num"] == 6] or subjects,
                study_plan=study_plan,
                min_attendance_pct=float(get_setting("min_attendance_pct", "75.0"))
            )
            st.download_button(
                label="📥 Download Official Academic Report (PDF)",
                data=pdf_bytes,
                file_name=f"Academic_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    with col2:
        st.markdown("### 📊 CSV Spreadsheets")
        st.write("Export raw data sheets for Excel, Google Sheets, or further data processing.")

        csv_cgpa = generate_cgpa_csv(semesters, subjects, overall_data)
        st.download_button(
            label="📥 Download CGPA & Grades (CSV)",
            data=csv_cgpa,
            file_name=f"CGPA_Report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        csv_plan = generate_study_plan_csv(study_plan)
        st.download_button(
            label="📥 Download Study Plan Timetable (CSV)",
            data=csv_plan,
            file_name=f"Study_Plan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        csv_att = generate_attendance_csv(subjects, min_pct=float(get_setting("min_attendance_pct", "75.0")))
        st.download_button(
            label="📥 Download Attendance Audit (CSV)",
            data=csv_att,
            file_name=f"Attendance_Audit_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ===========================================================================
# PAGE 8: University Config & Settings
# ===========================================================================
elif selected_page == "⚙️ University Config & Settings":
    st.markdown("<div class='main-header'>University Configuration & Policy Rules</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Verify and customize the university grade-point scale, attendance thresholds, and student credentials.</div>", unsafe_allow_html=True)

    tab_profile, tab_grade_scale, tab_attendance_rules = st.tabs([
        "👤 Student Profile",
        "⚖️ Grade Scale Setup",
        "📋 Attendance Policy Rules"
    ])

    with tab_profile:
        st.markdown("#### Student Academic Credentials")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            s_name = st.text_input("Student Name", get_setting("student_name", "Arjun Sharma"))
            s_roll = st.text_input("Roll / Registration No.", get_setting("roll_number", "23CSB042"))
        with col_p2:
            s_uni = st.text_input("University / Institution", get_setting("university", "National Institute of Technology / University NAP Dept."))
            s_deg = st.text_input("Degree Program", get_setting("degree_program", "B.Tech Computer Science & Engineering"))

        if st.button("💾 Save Profile Settings", key="btn_save_profile"):
            set_setting("student_name", s_name)
            set_setting("roll_number", s_roll)
            set_setting("university", s_uni)
            set_setting("degree_program", s_deg)
            st.success("Student profile saved successfully!")

    with tab_grade_scale:
        st.markdown("#### Configurable University Grade Scale (10-Point Default)")
        st.caption("Verify against your official university grading regulations (e.g. UGC, AICTE, AKTU, VTU, SPPU, Anna Univ).")

        curr_scale = get_grade_scale()
        scale_list = []
        for g_letter, (g_point, min_m, max_m) in curr_scale.items():
            scale_list.append({
                "Letter Grade": str(g_letter),
                "Grade Point": float(g_point),
                "Min Marks %": int(min_m),
                "Max Marks %": int(max_m)
            })

        scale_df = pd.DataFrame(scale_list)
        edited_df = st.data_editor(
            scale_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Letter Grade": st.column_config.TextColumn("Letter Grade", required=True),
                "Grade Point": st.column_config.NumberColumn("Grade Point", min_value=0.0, max_value=10.0, step=0.5, required=True),
                "Min Marks %": st.column_config.NumberColumn("Min Marks %", min_value=0, max_value=100, step=1, required=True),
                "Max Marks %": st.column_config.NumberColumn("Max Marks %", min_value=0, max_value=100, step=1, required=True)
            },
            key="grade_scale_editor"
        )

        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            if st.button("💾 Save Grade Scale to Database", key="btn_save_scale", use_container_width=True):
                new_scale_dict = {}
                for _, row in edited_df.iterrows():
                    g_lbl = str(row["Letter Grade"]).strip()
                    if g_lbl:
                        new_scale_dict[g_lbl] = (float(row["Grade Point"]), int(row["Min Marks %"]), int(row["Max Marks %"]))
                save_grade_scale(new_scale_dict)
                st.success("Custom university grade scale saved to SQLite database!")
                st.rerun()

        with col_s2:
            if st.button("🔄 Reset to Default 10-Point Scale", key="btn_reset_scale", use_container_width=True):
                save_grade_scale(DEFAULT_GRADE_SCALE)
                st.info("Reset grade scale to default UGC 10-point scale.")
                st.rerun()

    with tab_attendance_rules:
        st.markdown("#### Attendance Threshold & Deficit Policy Rules")
        att_rules = get_attendance_rules()
        c_att1, c_att2 = st.columns(2)
        with c_att1:
            r_min = st.number_input("Mandatory Minimum Attendance %", min_value=50.0, max_value=95.0, value=float(att_rules["min_percentage"]), step=1.0)
            st.caption("Standard requirement for examination eligibility (Default: 75.0%).")
        with c_att2:
            r_warn = st.number_input("Warning Alert Threshold %", min_value=r_min, max_value=100.0, value=float(att_rules["warning_threshold"]), step=1.0)
            st.caption("Trigger caution badges when attendance falls below this buffer.")

        if st.button("💾 Save Attendance Rules", key="btn_save_att_rules"):
            save_attendance_rules(r_min, r_warn)
            st.success("Attendance policy rules updated!")
            st.rerun()
