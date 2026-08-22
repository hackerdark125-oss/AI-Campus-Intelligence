"""
routers/chatbot.py
Document ingestion (syllabus/notices/etc.) and the RAG-powered chatbot endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app import models, schemas
from app.deps import get_current_user, require_admin
from app.services import rag_service

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router.post("/documents", response_model=schemas.DocumentOut, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    raw = file.file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422,
            detail="Only plain-text (.txt) documents are supported in this endpoint. "
                   "For PDFs, extract text first (see README).",
        )

    chunks = rag_service.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document appears to be empty.")

    document = models.Document(filename=file.filename)
    db.add(document)
    db.flush()  # get document.id before inserting chunks

    for idx, chunk in enumerate(chunks):
        db.add(models.DocumentChunk(document_id=document.id, chunk_index=idx, chunk_text=chunk))

    db.commit()
    db.refresh(document)

    return schemas.DocumentOut(
        id=document.id,
        filename=document.filename,
        uploaded_on=document.uploaded_on,
        chunk_count=len(chunks),
    )


@router.get("/documents", response_model=List[schemas.DocumentOut])
def list_documents(db: Session = Depends(get_db), _user: models.User = Depends(get_current_user)):
    docs = db.query(models.Document).all()
    return [
        schemas.DocumentOut(id=d.id, filename=d.filename, uploaded_on=d.uploaded_on, chunk_count=len(d.chunks))
        for d in docs
    ]


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db), _admin: models.User = Depends(require_admin)):
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    return None


@router.post("/ask", response_model=schemas.ChatResponse)
def ask(
    payload: schemas.ChatQuery,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    chunks = db.query(models.DocumentChunk).all()
    corpus = [(c.id, c.chunk_text) for c in chunks]

    retrieved = rag_service.retrieve_top_chunks(payload.question, corpus)
    answer, used_llm = rag_service.generate_answer(payload.question, retrieved)

    sources = []
    for cid, _, _ in retrieved:
        chunk = db.query(models.DocumentChunk).filter(models.DocumentChunk.id == cid).first()
        if chunk and chunk.document.filename not in sources:
            sources.append(chunk.document.filename)

    db.add(models.ChatLog(user_id=current_user.id, question=payload.question, answer=answer))
    db.commit()

    return schemas.ChatResponse(answer=answer, sources=sources, used_llm=used_llm)
