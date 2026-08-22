"""
analytics_service.py
Computes per-student and cohort-level attendance/performance analytics,
including a transparent, rule-based risk score.

Rule-based rather than a black-box ML model on purpose: for a small
class-scale dataset there usually isn't enough signal to train a reliable
classifier, and an explainable rule ("attendance is below threshold and
falling") is more trustworthy and auditable for an academic setting. See
README "Roadmap" for how this could be swapped for a trained model once
enough historical data exists.
"""

from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import func

from app import models
from app.config import ATTENDANCE_RISK_THRESHOLD


def _attendance_pct(present_days: int, total_days: int) -> float:
    if total_days == 0:
        return 0.0
    return round((present_days / total_days) * 100, 1)


def student_analytics(db: Session, student: models.Student) -> dict:
    total_days = db.query(func.count(models.ClassSession.id)).scalar() or 0
    present_days = (
        db.query(func.count(models.AttendanceRecord.id))
        .filter(models.AttendanceRecord.student_id == student.id)
        .scalar()
        or 0
    )
    pct = _attendance_pct(present_days, total_days)

    avg_score_row = (
        db.query(func.avg(models.AssignmentScore.score / models.AssignmentScore.max_score * 100))
        .filter(models.AssignmentScore.student_id == student.id)
        .scalar()
    )
    avg_score = round(avg_score_row, 1) if avg_score_row is not None else None

    risk_level, risk_reason = _assess_risk(pct, avg_score, total_days)

    return {
        "student_code": student.student_code,
        "name": student.name,
        "total_class_days": total_days,
        "present_days": present_days,
        "attendance_pct": pct,
        "avg_assignment_score": avg_score,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
    }


def _assess_risk(attendance_pct: float, avg_score, total_days: int) -> tuple:
    if total_days == 0:
        return "UNKNOWN", "No class sessions recorded yet."

    reasons = []
    if attendance_pct < ATTENDANCE_RISK_THRESHOLD:
        reasons.append(f"attendance ({attendance_pct}%) is below the {ATTENDANCE_RISK_THRESHOLD}% threshold")

    if avg_score is not None and avg_score < 50:
        reasons.append(f"average assignment score ({avg_score}%) is below 50%")

    if len(reasons) >= 2:
        return "HIGH", "; ".join(reasons).capitalize() + "."
    elif len(reasons) == 1:
        return "MEDIUM", reasons[0].capitalize() + "."
    else:
        return "LOW", "Attendance and assignment performance are within healthy ranges."


def cohort_analytics(db: Session) -> dict:
    students = db.query(models.Student).all()
    total_days = db.query(func.count(models.ClassSession.id)).scalar() or 0

    per_student: List[dict] = [student_analytics(db, s) for s in students]
    at_risk = sum(1 for s in per_student if s["risk_level"] in ("HIGH", "MEDIUM"))
    avg_attendance = (
        round(sum(s["attendance_pct"] for s in per_student) / len(per_student), 1)
        if per_student
        else 0.0
    )

    return {
        "total_students": len(per_student),
        "total_class_days": total_days,
        "average_attendance_pct": avg_attendance,
        "at_risk_count": at_risk,
        "students": per_student,
    }
