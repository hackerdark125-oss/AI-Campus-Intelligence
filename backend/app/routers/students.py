"""
routers/students.py
Admin-managed student roster.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app import models, schemas
from app.deps import require_admin, get_current_user

router = APIRouter(prefix="/students", tags=["students"])


@router.post("", response_model=schemas.StudentOut, status_code=status.HTTP_201_CREATED)
def create_student(
    payload: schemas.StudentCreate,
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    if db.query(models.Student).filter(models.Student.student_code == payload.student_code).first():
        raise HTTPException(status_code=400, detail="student_code already exists")

    student = models.Student(
        student_code=payload.student_code,
        name=payload.name,
        department=payload.department,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


@router.get("", response_model=List[schemas.StudentOut])
def list_students(db: Session = Depends(get_db), _user: models.User = Depends(get_current_user)):
    return db.query(models.Student).order_by(models.Student.name).all()


@router.get("/{student_code}", response_model=schemas.StudentOut)
def get_student(student_code: str, db: Session = Depends(get_db), _user: models.User = Depends(get_current_user)):
    student = db.query(models.Student).filter(models.Student.student_code == student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@router.delete("/{student_code}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(
    student_code: str,
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    student = db.query(models.Student).filter(models.Student.student_code == student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    db.delete(student)
    db.commit()
    return None
