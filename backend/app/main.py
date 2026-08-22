"""
main.py
FastAPI application entrypoint. Wires up the database, routers, and CORS
so the plain-HTML frontend (or anything else) can call this API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.routers import auth, students, attendance, analytics, chatbot

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Campus Intelligence Platform",
    description="Face-recognition attendance + RAG college assistant + student analytics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(students.router)
app.include_router(attendance.router)
app.include_router(analytics.router)
app.include_router(chatbot.router)


@app.get("/health")
def health():
    return {"status": "ok"}
