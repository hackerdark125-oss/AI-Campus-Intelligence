"""
routers/attendance.py
Face-based attendance:
- POST /attendance/register-face/{student_code}  -> upload one photo at a time (call
  repeatedly, e.g. from a browser webcam capture loop, to build up a face dataset)
- POST /attendance/train                          -> (re)train the recognizer, admin only
- POST /attendance/mark                            -> upload one photo, recognize + mark
- POST /attendance/session                         -> admin declares "class was held today"
- GET  /attendance/today                           -> today's marked attendance
"""

from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app import models, schemas
from app.deps import get_current_user, require_admin
from app.services import face_service

router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.post("/register-face/{student_code}", response_model=schemas.FaceRegisterResult)
def register_face(
    student_code: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    student = db.query(models.Student).filter(models.Student.student_code == student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found. Create the student record first.")

    image_bytes = file.file.read()
    try:
        img = face_service.decode_image(image_bytes)
        face_crop = face_service.extract_largest_face(img)
    except face_service.NoFaceDetected:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    existing = face_service.count_existing_samples(student_code)
    face_service.save_face_sample(student_code, face_crop, existing + 1)

    total = existing + 1
    student.face_sample_count = total
    student.face_registered = total >= 1
    db.commit()

    return schemas.FaceRegisterResult(
        student_code=student_code,
        samples_saved=1,
        total_samples=total,
        message=f"Saved sample {total}. Aim for 20-30+ samples across different angles/lighting, then call /attendance/train.",
    )


@router.post("/train", response_model=schemas.TrainResult)
def train(db: Session = Depends(get_db), _admin: models.User = Depends(require_admin)):
    try:
        result = face_service.train_model()
    except face_service.ModelNotTrained as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return schemas.TrainResult(
        trained_on_students=result["trained_on_students"],
        total_images=result["total_images"],
        message="Model trained successfully.",
    )


@router.post("/mark", response_model=schemas.AttendanceMarkResult)
def mark_attendance(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _user: models.User = Depends(get_current_user),
):
    image_bytes = file.file.read()
    try:
        img = face_service.decode_image(image_bytes)
        face_crop = face_service.extract_largest_face(img)
    except face_service.NoFaceDetected:
        return schemas.AttendanceMarkResult(recognized=False, message="No face detected in the image.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result = face_service.predict(face_crop)
    except face_service.ModelNotTrained as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not result["student_code"]:
        return schemas.AttendanceMarkResult(
            recognized=False,
            confidence=result["confidence"],
            message="Face not recognized with sufficient confidence.",
        )

    student = db.query(models.Student).filter(models.Student.student_code == result["student_code"]).first()
    if not student:
        return schemas.AttendanceMarkResult(recognized=False, message="Recognized ID has no matching student record.")

    today = date.today()
    already = (
        db.query(models.AttendanceRecord)
        .filter(models.AttendanceRecord.student_id == student.id, models.AttendanceRecord.date == today)
        .first()
    )

    if already:
        return schemas.AttendanceMarkResult(
            recognized=True,
            student_code=student.student_code,
            name=student.name,
            confidence=result["confidence"],
            already_marked_today=True,
            message=f"{student.name} was already marked present today at {already.time}.",
        )

    # Ensure today counts as a class session (auto-create if an admin hasn't explicitly declared one)
    if not db.query(models.ClassSession).filter(models.ClassSession.date == today).first():
        db.add(models.ClassSession(date=today))

    record = models.AttendanceRecord(
        student_id=student.id,
        date=today,
        time=datetime.now().strftime("%H:%M:%S"),
        method="face",
        confidence=result["confidence"],
    )
    db.add(record)
    db.commit()

    return schemas.AttendanceMarkResult(
        recognized=True,
        student_code=student.student_code,
        name=student.name,
        confidence=result["confidence"],
        already_marked_today=False,
        message=f"Attendance marked for {student.name}.",
    )


@router.post("/session", status_code=status.HTTP_201_CREATED)
def declare_class_session(
    session_date: date = date.today(),
    subject: str = "General",
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    existing = db.query(models.ClassSession).filter(models.ClassSession.date == session_date).first()
    if existing:
        return {"message": "Session already recorded for this date.", "date": str(session_date)}
    db.add(models.ClassSession(date=session_date, subject=subject))
    db.commit()
    return {"message": "Class session recorded.", "date": str(session_date), "subject": subject}


@router.get("/today", response_model=List[schemas.AttendanceOut])
def todays_attendance(db: Session = Depends(get_db), _user: models.User = Depends(get_current_user)):
    today = date.today()
    return (
        db.query(models.AttendanceRecord)
        .filter(models.AttendanceRecord.date == today)
        .order_by(models.AttendanceRecord.time)
        .all()
    )
