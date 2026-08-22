
"""
test_api.py
End-to-end integration tests for the AI Campus Intelligence Platform.

Covers:
- Health check
- Admin registration/login
- Student creation
- Face registration
- Face training
- Face recognition
- Attendance
- Analytics
- RAG chatbot
"""

import os
import io
import shutil
import sqlite3
import numpy as np
import cv2
import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# -------------------------------------------------------------------
# TEST ENVIRONMENT
# -------------------------------------------------------------------

os.environ["DATABASE_URL"] = "sqlite:///./test_campus.db"
os.environ["DATASET_DIR"] = "./test_dataset"
os.environ["TRAINER_DIR"] = "./test_trainer"

from app.main import app
from app.database import Base, get_db


TEST_DB_URL = "sqlite:///./test_campus.db"

engine = create_engine(
    TEST_DB_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,
    },
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


# -------------------------------------------------------------------
# CLEANUP HELPERS
# -------------------------------------------------------------------

def _close_sqlite_connections():
    """
    Dispose SQLAlchemy connections so Windows can delete SQLite files.
    """
    try:
        engine.dispose()
    except Exception:
        pass


def _safe_remove(path):
    """
    Windows-safe cleanup.
    """
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

        elif os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass

    except Exception:
        pass


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():

    # Clean old test data before starting
    _close_sqlite_connections()

    _safe_remove("./test_campus.db")
    _safe_remove("./test_dataset")
    _safe_remove("./test_trainer")

    # Create database tables
    Base.metadata.create_all(bind=engine)

    yield

    # Close DB connections
    _close_sqlite_connections()

    # Cleanup
    _safe_remove("./test_dataset")
    _safe_remove("./test_trainer")
    _safe_remove("./test_campus.db")


# -------------------------------------------------------------------
# IMAGE HELPERS
# -------------------------------------------------------------------

def _make_face_jpeg_bytes(seed: int) -> bytes:
    """
    Creates a deterministic synthetic face-like image.

    This test image intentionally does NOT depend on Haar Cascade
    detection. The application can be tested without downloading
    external cascade XML files.
    """

    rng = np.random.default_rng(seed)

    img = np.full(
        (300, 300, 3),
        40,
        dtype=np.uint8,
    )

    # Face
    cv2.ellipse(
        img,
        (150, 150),
        (90, 105),
        0,
        0,
        360,
        (190, 180, 170),
        -1,
    )

    # Eyes
    cv2.circle(
        img,
        (120, 130),
        12,
        (30, 30, 30),
        -1,
    )

    cv2.circle(
        img,
        (180, 130),
        12,
        (30, 30, 30),
        -1,
    )

    # Nose
    cv2.line(
        img,
        (150, 140),
        (145, 170),
        (50, 50, 50),
        4,
    )

    # Mouth
    cv2.ellipse(
        img,
        (150, 195),
        (35, 15),
        0,
        0,
        180,
        (50, 40, 40),
        4,
    )

    # Small deterministic noise
    noise = rng.integers(
        0,
        8,
        size=(300, 300, 3),
        dtype=np.uint8,
    )

    img = cv2.add(img, noise)

    ok, buffer = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    assert ok

    return buffer.tobytes()


# -------------------------------------------------------------------
# AUTH HELPERS
# -------------------------------------------------------------------

def _admin_headers():

    response = client.post(
        "/auth/login",
        data={
            "username": "admin1",
            "password": "adminpass123",
        },
    )

    assert response.status_code == 200, response.text

    token = response.json()["access_token"]

    return {
        "Authorization": f"Bearer {token}"
    }


# -------------------------------------------------------------------
# TEST 1 — HEALTH
# -------------------------------------------------------------------

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok"
    }


# -------------------------------------------------------------------
# TEST 2 — ADMIN REGISTER + LOGIN
# -------------------------------------------------------------------

def test_admin_register_and_login():

    response = client.post(
        "/auth/register",
        json={
            "username": "admin1",
            "email": "admin1@campus.edu",
            "password": "adminpass123",
            "role": "admin",
        },
    )

    assert response.status_code == 201, response.text

    response = client.post(
        "/auth/login",
        data={
            "username": "admin1",
            "password": "adminpass123",
        },
    )

    assert response.status_code == 200, response.text

    token = response.json()["access_token"]

    assert token

    response = client.get(
        "/auth/me",
        headers={
            "Authorization": f"Bearer {token}"
        },
    )

    assert response.status_code == 200

    assert response.json()["role"] == "admin"


# -------------------------------------------------------------------
# TEST 3 — STUDENT CREATION
# -------------------------------------------------------------------

def test_create_student_requires_admin():

    # No authentication
    response = client.post(
        "/students",
        json={
            "student_code": "S001",
            "name": "Shubham Sharma",
        },
    )

    assert response.status_code == 401

    headers = _admin_headers()

    response = client.post(
        "/students",
        json={
            "student_code": "S001",
            "name": "Shubham Sharma",
            "department": "ECE",
        },
        headers=headers,
    )

    assert response.status_code == 201, response.text

    assert response.json()["student_code"] == "S001"

    response = client.post(
        "/students",
        json={
            "student_code": "S002",
            "name": "Aditi Rao",
            "department": "CSE",
        },
        headers=headers,
    )

    assert response.status_code == 201, response.text


# -------------------------------------------------------------------
# TEST 4 — FACE REGISTRATION + TRAINING
# -------------------------------------------------------------------

def test_face_registration_training_and_recognition():

    headers = _admin_headers()

    # Register S001
    for i in range(8):

        image = _make_face_jpeg_bytes(
            seed=100 + i
        )

        response = client.post(
            "/attendance/register-face/S001",
            files={
                "file": (
                    "face.jpg",
                    image,
                    "image/jpeg",
                )
            },
            headers=headers,
        )

        assert response.status_code == 200, response.text

    # Register S002
    for i in range(8):

        image = _make_face_jpeg_bytes(
            seed=200 + i
        )

        response = client.post(
            "/attendance/register-face/S002",
            files={
                "file": (
                    "face.jpg",
                    image,
                    "image/jpeg",
                )
            },
            headers=headers,
        )

        assert response.status_code == 200, response.text

    # Train
    response = client.post(
        "/attendance/train",
        headers=headers,
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["trained_on_students"] == 2

    assert body["total_images"] == 16


# -------------------------------------------------------------------
# TEST 5 — ATTENDANCE
# -------------------------------------------------------------------

def test_mark_attendance_recognizes_and_prevents_duplicate():

    headers = _admin_headers()

    image = _make_face_jpeg_bytes(
        seed=100
    )

    response = client.post(
        "/attendance/mark",
        files={
            "file": (
                "face.jpg",
                image,
                "image/jpeg",
            )
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["recognized"] is True

    assert body["student_code"] == "S001"

    assert body["already_marked_today"] is False

    # Second attempt
    response2 = client.post(
        "/attendance/mark",
        files={
            "file": (
                "face.jpg",
                image,
                "image/jpeg",
            )
        },
        headers=headers,
    )

    assert response2.status_code == 200

    assert response2.json()["already_marked_today"] is True

    # Today's attendance
    response3 = client.get(
        "/attendance/today",
        headers=headers,
    )

    assert response3.status_code == 200

    assert len(response3.json()) == 1


# -------------------------------------------------------------------
# TEST 6 — ANALYTICS
# -------------------------------------------------------------------

def test_analytics_reflects_attendance():

    headers = _admin_headers()

    response = client.post(
        "/analytics/assignments",
        json={
            "student_code": "S001",
            "title": "Assignment 1",
            "score": 85,
            "max_score": 100,
        },
        headers=headers,
    )

    assert response.status_code == 201, response.text

    response = client.get(
        "/analytics/student/S001",
        headers=headers,
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["present_days"] == 1

    assert data["total_class_days"] == 1

    assert data["attendance_pct"] == 100.0

    assert data["avg_assignment_score"] == 85.0

    # S002 has no attendance
    response = client.get(
        "/analytics/student/S002",
        headers=headers,
    )

    assert response.status_code == 200

    data2 = response.json()

    assert data2["present_days"] == 0

    assert data2["risk_level"] in (
        "HIGH",
        "MEDIUM",
    )

    # Cohort
    response = client.get(
        "/analytics/cohort",
        headers=headers,
    )

    assert response.status_code == 200

    cohort = response.json()

    assert cohort["total_students"] == 2


# -------------------------------------------------------------------
# TEST 7 — CHATBOT
# -------------------------------------------------------------------

def test_chatbot_rag_pipeline():

    headers = _admin_headers()

    document = (
        "Examination Notice: The Database Management Systems "
        "(DBMS) end-semester examination will be held on the "
        "14th of December. Students must carry their ID card. "
        "The minimum attendance requirement to be eligible "
        "for the exam is 75 percent. Students below this "
        "threshold must apply for condonation through the "
        "department office."
    )

    response = client.post(
        "/chatbot/documents",
        files={
            "file": (
                "exam_notice.txt",
                document.encode("utf-8"),
                "text/plain",
            )
        },
        headers=headers,
    )

    assert response.status_code == 201, response.text

    assert response.json()["chunk_count"] >= 1

    response = client.post(
        "/chatbot/ask",
        json={
            "question": (
                "What is the minimum attendance "
                "requirement for exams?"
            )
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert "75" in data["answer"]

    assert "exam_notice.txt" in data["sources"]

    # No API key in test environment
    assert data["used_llm"] is False


# -------------------------------------------------------------------
# TEST 8 — UNRELATED CHATBOT QUESTION
# -------------------------------------------------------------------

def test_unrelated_question_has_no_relevant_chunks():

    headers = _admin_headers()

    response = client.post(
        "/chatbot/ask",
        json={
            "question": (
                "What is the airspeed velocity "
                "of an unladen swallow?"
            )
        },
        headers=headers,
    )

    assert response.status_code == 200

    answer = response.json()["answer"].lower()

    sources = response.json()["sources"]

    assert (
        "couldn't find" in answer
        or sources == []
    )