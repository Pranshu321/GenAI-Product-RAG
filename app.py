import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import string
import faiss
from sentence_transformers import SentenceTransformer

# ─── Page config ───
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
)

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

DATA_DIR = "data"


# ─── Load & cache heavy resources ───
@st.cache_resource(show_spinner="Loading assessment catalog...")
def load_catalog():
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
    return df


def _create_text_repr(row):
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


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer("all-mpnet-base-v2")


FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")


@st.cache_resource(show_spinner="Loading search index...")
def build_faiss_index(_df, _model):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        if index.ntotal == len(_df):
            return index, embeddings

    texts = _df["text_repr"].tolist()
    embeddings = _model.encode(texts, normalize_embeddings=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    return index, embeddings


@st.cache_data(show_spinner=False)
def tokenize_corpus(text_list):
    result = []
    for text in text_list:
        t = text.lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        result.append([w for w in t.split() if len(w) > 2])
    return result


# ─── Recommendation logic ───
def keyword_scores(query_tokens, corpus_tokens_list):
    query_set = set(query_tokens)
    scores = []
    for doc_tokens in corpus_tokens_list:
        doc_set = set(doc_tokens)
        if not query_set or not doc_set:
            scores.append(0.0)
            continue
        scores.append(len(query_set & doc_set) / len(query_set))
    return np.array(scores, dtype="float32")


def recommend(query, df, model, index, corpus_tokens, top_k=10, alpha=0.7):
    q_emb = model.encode([query[:500]], normalize_embeddings=True).astype("float32")
    sem_scores_all, sem_indices = index.search(q_emb, index.ntotal)
    sem_scores = np.zeros(len(df), dtype="float32")
    for score, idx in zip(sem_scores_all[0], sem_indices[0]):
        sem_scores[idx] = score

    q_tokens = query.lower().translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    ).split()
    q_tokens = [w for w in q_tokens if len(w) > 2]
    kw_scores = keyword_scores(q_tokens, corpus_tokens)

    if sem_scores.max() > 0:
        sem_scores /= sem_scores.max()
    if kw_scores.max() > 0:
        kw_scores /= kw_scores.max()

    combined = alpha * sem_scores + (1 - alpha) * kw_scores
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["Relevance"] = combined[top_indices]
    return results.reset_index(drop=True)


def format_test_types(codes):
    if not codes or pd.isna(codes):
        return ""
    return ", ".join(TEST_TYPE_MAP.get(c.strip(), c.strip()) for c in str(codes).split() if c.strip())


# ─── UI ───
def main():
    df = load_catalog()
    model = load_embed_model()
    index, _ = build_faiss_index(df, model)
    corpus_tokens = tokenize_corpus(df["text_repr"].tolist())

    st.title("SHL Assessment Recommender")
    st.markdown(
        "Enter a **natural language query** or paste a **job description** to get "
        "the most relevant SHL assessments."
    )

    col_input, col_settings = st.columns([3, 1])
    with col_input:
        query = st.text_area(
            "Query / Job Description",
            height=160,
            placeholder="e.g. I am hiring Java developers who can collaborate with business teams...",
        )
    with col_settings:
        top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=10)
        alpha = st.slider(
            "Semantic vs Keyword weight",
            min_value=0.0, max_value=1.0, value=0.7, step=0.1,
            help="1.0 = pure semantic search, 0.0 = pure keyword matching",
        )

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Finding relevant assessments..."):
            results = recommend(query, df, model, index, corpus_tokens, top_k=top_k, alpha=alpha)

        st.subheader(f"Top {len(results)} Recommended Assessments")

        display = pd.DataFrame({
            "#": range(1, len(results) + 1),
            "Assessment Name": results["name"],
            "Test Type": results.get("test_type_codes", pd.Series(dtype=str)).apply(format_test_types),
            # show -- when duration is missing or NaN
            "Duration (min)": results.get("duration_minutes", pd.Series(dtype=float))
                .apply(lambda x: "--" if pd.isna(x) else x),
            "Remote": results.get("remote_testing", pd.Series(dtype=bool)).apply(
                lambda x: "Yes" if x else "No" if pd.notna(x) else "-"
            ),
            "Relevance": results["Relevance"].apply(lambda x: f"{x:.3f}"),
            "Catalog Link": results["url"].apply(
                lambda u: f"[View]({u})"
            ),
        })

        st.markdown(
            display.to_markdown(index=False),
            unsafe_allow_html=True,
        )

        with st.expander("Detailed view"):
            for i, (_, row) in enumerate(results.iterrows()):
                st.markdown(f"**{i+1}. [{row['name']}]({row['url']})**")
                cols = st.columns(4)
                cols[0].metric("Test Type", format_test_types(row.get("test_type_codes", "")) or "-")
                # display -- when duration is NaN or missing
                dur_val = row.get("duration_minutes")
                dur_label = "--" if pd.isna(dur_val) or dur_val is None else f"{dur_val} min"
                cols[1].metric("Duration", dur_label)
                cols[2].metric("Remote", "Yes" if row.get("remote_testing") else "No")
                cols[3].metric("Relevance", f"{row.get('Relevance', 0):.3f}")
                if row.get("description"):
                    st.caption(row["description"][:200])
                st.divider()

    with st.sidebar:
        st.header("About")
        st.markdown(
            "**SHL Assessment Recommender** uses a RAG pipeline to find relevant "
            "assessments from SHL's product catalog.\n\n"
            "**Pipeline:**\n"
            "1. Sentence-transformer embeddings\n"
            "2. FAISS vector similarity search\n"
            "3. Hybrid semantic + keyword scoring\n\n"
            f"**Catalog size:** {len(df)} assessments"
        )
        st.divider()
        st.caption("Built with Streamlit, FAISS, and sentence-transformers")


if __name__ == "__main__":
    main()
