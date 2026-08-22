# AI Campus Intelligence Platform

A full-stack AI system for a college that combines **face-recognition attendance**,
a **RAG-powered college assistant chatbot**, and **student performance analytics** —
built as a coherent product rather than three disconnected scripts.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## What it does

```
                    AI CAMPUS
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   🎥 Vision       🤖 AI Chatbot    📊 Analytics
        │              │              │
   Attendance      College RAG     Dashboard
   (face rec.)     (documents)     (risk scoring)
        └──────────────┼──────────────┘
                       ↓
                REST API (FastAPI)
                       ↓
              Web frontend (HTML/JS)
```

- **Smart attendance** — a browser webcam captures a face, the backend detects and recognizes it (OpenCV LBPH), and marks attendance once per person per day. No manual roll calls.
- **College assistant chatbot** — admins upload syllabus/notices/rules as text; students ask questions in plain English and get answers grounded in those documents (retrieval-augmented generation), with sources cited — not hallucinated.
- **Analytics dashboard** — per-student and cohort attendance %, assignment averages, and a transparent, explainable risk flag (not a black-box model) for students who may need attention.

## Why this over the original single-feature attendance project

| Skill            | Demonstrated |
|-------------------|:---:|
| REST API design (FastAPI)      | ✅ |
| Relational DB modeling (SQLAlchemy) | ✅ |
| Auth (JWT, password hashing, RBAC) | ✅ |
| Computer vision (OpenCV face detection/recognition) | ✅ |
| Information retrieval / RAG      | ✅ |
| LLM API integration (Claude)     | ✅ |
| Data analytics & scoring logic   | ✅ |
| Frontend (webcam capture, fetch API, auth flows) | ✅ |
| Docker / containerized deployment | ✅ |
| Automated testing (pytest, integration tests) | ✅ |

## Project structure

```
ai-campus-intelligence/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entrypoint
│   │   ├── config.py            # env-driven settings
│   │   ├── database.py          # SQLAlchemy engine/session
│   │   ├── models.py            # ORM models
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── security.py          # password hashing + JWT
│   │   ├── deps.py              # auth dependencies / RBAC guards
│   │   ├── routers/
│   │   │   ├── auth.py
│   │   │   ├── students.py
│   │   │   ├── attendance.py    # face registration, training, marking
│   │   │   ├── analytics.py
│   │   │   └── chatbot.py       # document ingestion + RAG query
│   │   └── services/
│   │       ├── face_service.py      # OpenCV LBPH detection/training/prediction
│   │       ├── rag_service.py       # TF-IDF retrieval + Claude generation
│   │       └── analytics_service.py # attendance %, risk scoring
│   ├── tests/test_api.py        # full integration test suite
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/                    # plain HTML/JS, no build step
│   ├── login.html / register.html
│   ├── dashboard.html           # cohort analytics + risk table
│   ├── students.html            # roster management
│   ├── attendance.html          # webcam-driven registration + marking
│   ├── chatbot.html             # document upload + Q&A
│   ├── app.js                   # shared API client
│   ├── styles.css
│   └── Dockerfile
├── documents/
│   └── sample_exam_notice.txt   # demo doc for the chatbot
├── docker-compose.yml           # backend + Postgres + frontend
├── LICENSE
└── README.md
```

## Quickstart (local, no Docker)

**Requirements:** Python 3.10+, a webcam (for the attendance features), modern browser.

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# optional: cp .env.example .env and edit it
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
python -m http.server 8080
```

Open `http://localhost:8080` in your browser. First-time setup:

1. **Register** an admin account (`register.html`).
2. **Students page** — add a few students.
3. **Attendance page** — start the camera, select a student code, and capture ~20 face samples from different angles. Repeat for each student, then click **Train Model**.
4. **Attendance page (right panel)** — start the camera and click **Recognize & Mark** to test live recognition.
5. **Assistant page** — upload `documents/sample_exam_notice.txt` and ask something like *"What is the minimum attendance requirement?"*

> **Note on the chatbot:** without an `ANTHROPIC_API_KEY`, the assistant still works — it returns the most relevant passage from your documents directly (extractive mode) instead of a generated sentence. Set the key in `.env` to get natural-language, Claude-generated answers grounded in the same retrieved context.

## Running the tests

```bash
cd backend
pip install pytest
pytest tests/ -v
```

The suite spins up an isolated SQLite test database and exercises the **entire** pipeline end-to-end: registration/login, RBAC, face registration → training → recognition with synthetic images, duplicate-attendance prevention, risk-scoring analytics, and the full RAG chatbot flow (ingest → retrieve → answer → correct source attribution).

## Running with Docker

```bash
cp backend/.env.example .env   # edit SECRET_KEY / ANTHROPIC_API_KEY as needed
docker compose up --build
```

- Backend API: `http://localhost:8000` (docs at `/docs`)
- Frontend: `http://localhost:8080`
- Postgres persists in a named volume; face data persists in `face_dataset`/`face_trainer` volumes.

## Design decisions worth knowing about

- **Face recognition is LBPH (OpenCV), not a deep embedding model.** It needs no external model download and runs fully offline/CPU-only, which matters for a portfolio project people will actually clone and run. It's genuinely less accurate than FaceNet/dlib-style embeddings at scale — see Roadmap.
- **RAG retrieval is TF-IDF, not dense embeddings.** Same reasoning: zero external downloads, fully deterministic, good enough for a small set of college documents. Swappable for `sentence-transformers` or the Claude embeddings API later without touching the chunking/storage code.
- **Risk scoring is rule-based, not a trained ML model.** For a single class's worth of data there usually isn't enough signal to train a reliable classifier, and an explainable rule is more trustworthy in an academic context than an opaque prediction. See `analytics_service.py` docstring.
- **Frontend is plain HTML/JS, not React/Next.js.** No build step to break on someone else's machine; every page just calls the REST API with `fetch`. Structured so a React/Next rewrite later is a drop-in replacement for the DOM-manipulation parts — the API contract doesn't change.

## Roadmap

- [ ] Swap LBPH for a `face_recognition` (dlib) or ONNX deep-embedding model for higher accuracy at scale
- [ ] Liveness detection (blink/head-pose challenge) to prevent marking attendance from a printed photo
- [ ] Swap TF-IDF retrieval for dense embeddings (`sentence-transformers` or Claude's embeddings API) once external model access is available
- [ ] PDF ingestion for the chatbot (currently `.txt` only — see `chunk_text` in `rag_service.py` for where to plug a PDF-to-text extractor)
- [ ] React/Next.js frontend rewrite for a richer dashboard (charts, drag-and-drop upload)
- [ ] Email/SMS notification when attendance is marked or a student crosses into HIGH risk
- [ ] Multi-camera / classroom deployment support

## License

MIT — see [LICENSE](LICENSE).
