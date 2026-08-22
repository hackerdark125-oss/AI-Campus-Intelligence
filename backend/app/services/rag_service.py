"""
rag_service.py
Retrieval-Augmented Generation for the college assistant chatbot.

Retrieval: TF-IDF + cosine similarity over document chunks (scikit-learn).
This is fully offline — no embedding-model downloads required, which matters
since this environment has no route to huggingface.co. It's a legitimate,
classic IR approach and is easy to swap for dense embeddings later (see
README "Roadmap").

Generation: if ANTHROPIC_API_KEY is set, the retrieved chunks are passed to
Claude to produce a natural-language answer grounded in that context. If no
key is configured, the service falls back to returning the most relevant
chunk(s) directly (still genuinely useful, just extractive instead of
generative) so the feature works out of the box with zero configuration.
"""

from typing import List, Tuple
import httpx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, RAG_TOP_K


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    """Simple sliding-window chunker over raw text (character-based, model-agnostic)."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0 or end >= len(text):
            break
    return chunks


def retrieve_top_chunks(query: str, corpus: List[Tuple[int, str]], top_k: int = None) -> List[Tuple[int, str, float]]:
    """
    corpus: list of (chunk_id, chunk_text)
    Returns list of (chunk_id, chunk_text, score) sorted by relevance, best first.
    """
    top_k = top_k or RAG_TOP_K
    if not corpus:
        return []

    texts = [c[1] for c in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts + [query])
    except ValueError:
        # e.g. empty vocabulary (all stopwords) — no meaningful match possible
        return []

    doc_vectors = matrix[:-1]
    query_vector = matrix[-1]
    scores = cosine_similarity(query_vector, doc_vectors).flatten()

    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    results = [(cid, ctext, float(score)) for (cid, ctext), score in ranked[:top_k] if score > 0]
    return results


def generate_answer(question: str, retrieved: List[Tuple[int, str, float]]) -> Tuple[str, bool]:
    """
    Returns (answer_text, used_llm).
    Calls Claude if ANTHROPIC_API_KEY is configured; otherwise returns an
    extractive fallback built directly from the retrieved chunks.
    """
    if not retrieved:
        return (
            "I couldn't find anything relevant to that in the uploaded college documents. "
            "Try rephrasing, or ask an admin to upload the relevant syllabus/notice.",
            False,
        )

    context = "\n\n---\n\n".join(chunk for _, chunk, _ in retrieved)

    if ANTHROPIC_API_KEY:
        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 500,
                    "system": (
                        "You are a college assistant chatbot. Answer the student's question "
                        "using ONLY the provided context from official college documents. "
                        "If the context doesn't contain the answer, say so plainly instead of "
                        "guessing."
                    ),
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}",
                        }
                    ],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            if text_blocks:
                return "\n".join(text_blocks), True
        except Exception:
            # Network/API issue — fall through to extractive fallback below
            pass

    # Extractive fallback: no LLM configured or the call failed
    best_chunk = retrieved[0][1]
    return (
        f"(Extractive answer — set ANTHROPIC_API_KEY for AI-generated answers)\n\n"
        f"The most relevant passage found:\n\n{best_chunk}",
        False,
    )
