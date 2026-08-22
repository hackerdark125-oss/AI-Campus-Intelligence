"""
routers/analytics.py
Cohort and per-student attendance/performance analytics.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.deps import get_current_user, require_admin
from app.services import analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/cohort", response_model=schemas.CohortAnalytics)
def cohort(db: Session = Depends(get_db), _admin: models.User = Depends(require_admin)):
    return analytics_service.cohort_analytics(db)


@router.get("/student/{student_code}", response_model=schemas.StudentAnalytics)
def student(student_code: str, db: Session = Depends(get_db), _user: models.User = Depends(get_current_user)):
    student = db.query(models.Student).filter(models.Student.student_code == student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return analytics_service.student_analytics(db, student)


@router.post("/assignments", status_code=201)
def record_assignment(
    payload: schemas.AssignmentCreate,
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    student = db.query(models.Student).filter(models.Student.student_code == payload.student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    score = models.AssignmentScore(
        student_id=student.id,
        title=payload.title,
        score=payload.score,
        max_score=payload.max_score,
    )
    db.add(score)
    db.commit()
    return {"message": "Assignment score recorded."}
