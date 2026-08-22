"""
face_service.py

API-friendly face recognition service.

- Receives uploaded images
- Detects faces using Haar Cascade
- Saves face samples
- Trains OpenCV LBPH recognizer
- Predicts student identity
"""

import os
import cv2
import numpy as np

from app.config import (
    DATASET_DIR,
    TRAINER_DIR,
    TRAINER_FILE,
    ID_MAP_FILE,
    FACE_CONFIDENCE_THRESHOLD,
)


# ============================================================
# HAAR CASCADE
# ============================================================
# Project structure:
#
# ai-campus-intelligence/
# ├── haarcascade_frontalface_default.xml
# └── backend/
#     └── app/
#         └── services/
#             └── face_service.py
#
# Therefore we go 3 levels up from this file.
# ============================================================

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

_CASCADE_PATH = os.path.join(
    _PROJECT_ROOT,
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class NoFaceDetected(Exception):
    pass


class ModelNotTrained(Exception):
    pass


# ============================================================
# FACE DETECTOR
# ============================================================

def _detector():
    """
    Load Haar Cascade safely.
    """

    if not os.path.exists(_CASCADE_PATH):
        raise FileNotFoundError(
            "\nHaar Cascade file not found!\n\n"
            f"Expected location:\n{_CASCADE_PATH}\n\n"
            "Please place haarcascade_frontalface_default.xml "
            "in the ai-campus-intelligence project root."
        )

    detector = cv2.CascadeClassifier(_CASCADE_PATH)

    if detector.empty():
        raise RuntimeError(
            "\nOpenCV could not load the Haar Cascade.\n\n"
            f"File:\n{_CASCADE_PATH}\n\n"
            "Make sure the file is a valid "
            "haarcascade_frontalface_default.xml file."
        )

    return detector


# ============================================================
# DIRECTORIES
# ============================================================

def _ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TRAINER_DIR, exist_ok=True)


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded JPEG/PNG bytes into an OpenCV BGR image.
    """

    arr = np.frombuffer(image_bytes, dtype=np.uint8)

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(
            "Could not decode image. Please upload a valid JPEG or PNG."
        )

    return img


# ============================================================
# FACE DETECTION
# ============================================================

def extract_largest_face(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect faces and return the largest face as grayscale.

    Raises:
        NoFaceDetected: if no face is detected.
    """

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid image.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    detector = _detector()

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        raise NoFaceDetected(
            "No face detected in the image. "
            "Please upload a clear image with your face visible."
        )

    # Select the largest detected face.
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    face_crop = gray[y:y + h, x:x + w]

    if face_crop.size == 0:
        raise NoFaceDetected("Detected face crop is empty.")

    return face_crop


# ============================================================
# SAVE FACE SAMPLE
# ============================================================

def save_face_sample(
    student_code: str,
    face_crop: np.ndarray,
    index: int
) -> str:

    _ensure_dirs()

    filename = os.path.join(
        DATASET_DIR,
        f"{student_code}.{index}.jpg"
    )

    success = cv2.imwrite(
        filename,
        face_crop
    )

    if not success:
        raise IOError(
            f"Could not save face sample: {filename}"
        )

    return filename


# ============================================================
# COUNT EXISTING SAMPLES
# ============================================================

def count_existing_samples(student_code: str) -> int:

    if not os.path.isdir(DATASET_DIR):
        return 0

    prefix = f"{student_code}."

    return len([
        f
        for f in os.listdir(DATASET_DIR)
        if f.startswith(prefix)
        and f.lower().endswith(".jpg")
    ])


# ============================================================
# ID MAP
# ============================================================

def _load_id_map() -> dict:

    mapping = {}

    if not os.path.exists(ID_MAP_FILE):
        return mapping

    with open(
        ID_MAP_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 2:
                continue

            student_code, numeric_id = parts

            mapping[
                student_code
            ] = int(numeric_id)

    return mapping


def _save_id_map(mapping: dict):

    _ensure_dirs()

    with open(
        ID_MAP_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for student_code, numeric_id in mapping.items():

            file.write(
                f"{student_code},{numeric_id}\n"
            )


def _string_id_to_int(
    student_code: str,
    mapping: dict
) -> int:

    if student_code not in mapping:

        mapping[student_code] = (
            max(mapping.values(), default=0) + 1
        )

    return mapping[student_code]


def load_id_map_reversed() -> dict:
    """
    Returns:

        {
            numeric_id: student_code
        }
    """

    return {
        numeric_id: student_code
        for student_code, numeric_id
        in _load_id_map().items()
    }


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model() -> dict:

    _ensure_dirs()

    if not hasattr(cv2, "face"):

        raise RuntimeError(
            "cv2.face is unavailable.\n\n"
            "Install opencv-contrib-python-headless "
            "instead of regular opencv-python."
        )

    image_paths = [
        os.path.join(DATASET_DIR, filename)
        for filename in os.listdir(DATASET_DIR)
        if filename.lower().endswith(".jpg")
    ]

    if not image_paths:

        raise ModelNotTrained(
            "No face samples found. "
            "Register at least one student's face first."
        )

    mapping = _load_id_map()

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    samples = []
    ids = []

    for path in image_paths:

        img = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        filename = os.path.basename(path)

        # Example:
        # S001.1.jpg
        # S001.2.jpg
        student_code = filename.split(".")[0]

        numeric_id = _string_id_to_int(
            student_code,
            mapping
        )

        samples.append(img)
        ids.append(numeric_id)

    if not samples:

        raise ModelNotTrained(
            "No valid face images were found."
        )

    recognizer.train(
        samples,
        np.array(ids)
    )

    recognizer.save(
        TRAINER_FILE
    )

    _save_id_map(
        mapping
    )

    return {
        "trained_on_students": len(set(ids)),
        "total_images": len(samples),
    }


# ============================================================
# FACE PREDICTION
# ============================================================

def predict(face_crop: np.ndarray) -> dict:
    """
    Predict student identity.

    Lower LBPH confidence value = better match.

    Returns:

        {
            "student_code": "S001",
            "confidence": 42.5
        }

    or:

        {
            "student_code": None,
            "confidence": 82.4
        }
    """

    if not os.path.exists(TRAINER_FILE):

        raise ModelNotTrained(
            "Face recognition model has not been trained yet."
        )

    if not hasattr(cv2, "face"):

        raise RuntimeError(
            "cv2.face is unavailable. "
            "Install opencv-contrib-python-headless."
        )

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    recognizer.read(
        TRAINER_FILE
    )

    id_map = load_id_map_reversed()

    numeric_id, confidence = recognizer.predict(
        face_crop
    )

    student_code = id_map.get(
        numeric_id
    )

    # Unknown face
    if (
        student_code is None
        or confidence >= FACE_CONFIDENCE_THRESHOLD
    ):

        return {
            "student_code": None,
            "confidence": float(confidence),
        }

    # Recognized face
    return {
        "student_code": student_code,
        "confidence": float(confidence),
    }