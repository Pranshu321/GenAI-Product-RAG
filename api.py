from contextlib import asynccontextmanager
from typing import Optional

import faiss
import json
import numpy as np
import os
import pandas as pd
import string

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# ─── Constants ───

DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


# ─── Shared state loaded once at startup ───

class AppState:
    df: pd.DataFrame
    model: SentenceTransformer
    index: faiss.Index
    corpus_tokens: list[list[str]]


state = AppState()


def _create_text_repr(row: dict) -> str:
    parts = [f"Assessment: {row.get('name', '')}"]
    if row.get("description"):
        parts.append(f"Description: {row['description']}")
    if row.get("test_type_names"):
        parts.append(f"Test Type: {row['test_type_names']}")
    elif row.get("test_type_codes"):
        names = ", ".join(
            TEST_TYPE_MAP.get(c, c)
            for c in str(row["test_type_codes"]).split()
            if c in TEST_TYPE_MAP
        )
        parts.append(f"Test Type: {names}")
    if row.get("job_levels"):
        parts.append(f"Job Levels: {row['job_levels']}")
    if row.get("duration_minutes"):
        parts.append(f"Duration: {row['duration_minutes']} minutes")
    return ". ".join(parts)


def _tokenize(text: str) -> list[str]:
    t = text.lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return [w for w in t.split() if len(w) > 2]


def _load_all():
    catalog_path = os.path.join(DATA_DIR, "shl_catalog.json")
    prepackaged_path = os.path.join(DATA_DIR, "shl_prepackaged.json")

    with open(catalog_path) as f:
        catalog = json.load(f)
    for item in catalog:
        item.setdefault("category", "Individual")

    prepackaged = []
    if os.path.exists(prepackaged_path):
        with open(prepackaged_path) as f:
            prepackaged = json.load(f)
        for item in prepackaged:
            item.setdefault("category", "Pre-packaged")

    df = pd.DataFrame(catalog + prepackaged)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    df["text_repr"] = df.apply(_create_text_repr, axis=1)
    state.df = df

    state.model = SentenceTransformer("all-mpnet-base-v2")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        state.index = faiss.read_index(FAISS_INDEX_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        if state.index.ntotal != len(df):
            state.index, _ = _build_index(df)
    else:
        state.index, _ = _build_index(df)

    state.corpus_tokens = [_tokenize(t) for t in df["text_repr"].tolist()]


def _build_index(df: pd.DataFrame):
    texts = df["text_repr"].tolist()
    embeddings = state.model.encode(texts, normalize_embeddings=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    return index, embeddings


# ─── Recommendation logic (same as app.py) ───

def _keyword_scores(query_tokens: list[str], corpus_tokens: list[list[str]]) -> np.ndarray:
    query_set = set(query_tokens)
    scores = []
    for doc_tokens in corpus_tokens:
        doc_set = set(doc_tokens)
        if not query_set or not doc_set:
            scores.append(0.0)
            continue
        scores.append(len(query_set & doc_set) / len(query_set))
    return np.array(scores, dtype="float32")


def _recommend(query: str, top_k: int = 10, alpha: float = 0.7) -> pd.DataFrame:
    df = state.df

    q_emb = state.model.encode([query[:500]], normalize_embeddings=True).astype("float32")
    sem_scores_all, sem_indices = state.index.search(q_emb, state.index.ntotal)
    sem_scores = np.zeros(len(df), dtype="float32")
    for score, idx in zip(sem_scores_all[0], sem_indices[0]):
        sem_scores[idx] = score

    q_tokens = _tokenize(query)
    kw_scores = _keyword_scores(q_tokens, state.corpus_tokens)

    if sem_scores.max() > 0:
        sem_scores /= sem_scores.max()
    if kw_scores.max() > 0:
        kw_scores /= kw_scores.max()

    combined = alpha * sem_scores + (1 - alpha) * kw_scores
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["relevance_score"] = combined[top_indices]
    return results.reset_index(drop=True)


def _format_test_types(codes) -> list[str]:
    if not codes or (isinstance(codes, float) and np.isnan(codes)):
        return []
    return [TEST_TYPE_MAP[c.strip()] for c in str(codes).split() if c.strip() in TEST_TYPE_MAP]


# ─── Pydantic models ───

class RecommendRequest(BaseModel):
    query: str

class AssessmentResult(BaseModel):
    assessment_name: str
    url: str
    test_type: list[str]
    duration: Optional[int] = None
    remote_testing: bool
    adaptive_irt: bool

class RecommendResponse(BaseModel):
    recommendations: list[AssessmentResult]

class HealthResponse(BaseModel):
    status: str


# ─── FastAPI app ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    yield

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Returns relevant SHL assessments for a given query or job description.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    results = _recommend(req.query, top_k=10, alpha=0.7)

    recommendations = []
    for _, row in results.iterrows():
        recommendations.append(AssessmentResult(
            assessment_name=row.get("name", ""),
            url=row.get("url", ""),
            test_type=_format_test_types(row.get("test_type_codes", "")),
            duration=int(row["duration_minutes"]) if pd.notna(row.get("duration_minutes")) else None,
            remote_testing=bool(row.get("remote_testing", False)),
            adaptive_irt=bool(row.get("adaptive_irt", False)),
        ))

    return RecommendResponse(recommendations=recommendations)


# make it simple to launch locally or on Render (which supplies PORT env var)
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
