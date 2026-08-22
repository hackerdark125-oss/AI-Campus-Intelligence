"""
schemas.py
Pydantic models for request/response validation.
"""

from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, EmailStr, ConfigDict


# ---------- Auth ----------

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "student"  # "admin" or "student"
    student_code: Optional[str] = None  # link to an existing Student record


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: EmailStr
    role: str
    student_id: Optional[int] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- Students ----------

class StudentCreate(BaseModel):
    student_code: str
    name: str
    department: Optional[str] = None


class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    student_code: str
    name: str
    department: Optional[str] = None
    face_registered: bool
    face_sample_count: int


# ---------- Attendance ----------

class AttendanceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    student_id: int
    date: date
    time: str
    method: str
    confidence: Optional[float] = None


class AttendanceMarkResult(BaseModel):
    recognized: bool
    student_code: Optional[str] = None
    name: Optional[str] = None
    confidence: Optional[float] = None
    already_marked_today: bool = False
    message: str


class FaceRegisterResult(BaseModel):
    student_code: str
    samples_saved: int
    total_samples: int
    message: str


class TrainResult(BaseModel):
    trained_on_students: int
    total_images: int
    message: str


# ---------- Analytics ----------

class StudentAnalytics(BaseModel):
    student_code: str
    name: str
    total_class_days: int
    present_days: int
    attendance_pct: float
    avg_assignment_score: Optional[float] = None
    risk_level: str
    risk_reason: str


class CohortAnalytics(BaseModel):
    total_students: int
    total_class_days: int
    average_attendance_pct: float
    at_risk_count: int
    students: List[StudentAnalytics]


# ---------- Assignments ----------

class AssignmentCreate(BaseModel):
    student_code: str
    title: str
    score: float
    max_score: float = 100.0


# ---------- RAG / Chatbot ----------

class ChatQuery(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    used_llm: bool


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    uploaded_on: datetime
    chunk_count: int
