# SHL Assessment Recommendation System — Approach Document

## 1. Problem Statement

Hiring managers struggle to find the right SHL assessments for open roles. The current catalog relies on manual keyword search, which is time-consuming and misses relevant options. We built an intelligent recommendation system that accepts a natural language query or job description and returns the 10 most relevant SHL assessments, combining retrieval-augmented generation (RAG) with hybrid search.

---

## 2. Data Pipeline

### 2.1 Web Scraping

We crawled the SHL product catalog (`shl.com/solutions/products/product-catalog/`) using `requests` and `BeautifulSoup`. The catalog paginates at 12 items per page, so we iterated through all pages until exhaustion.

**Why scraping?** The assignment requires building the system from the live catalog, not a static dataset. Scraping ensures the data reflects the actual product offering.

- **Phase 1 — Catalog listing pages:** Extracted assessment names, URLs, remote testing availability, adaptive/IRT support, and test type codes (K, P, A, B, S, C, D, E) from the HTML table structure. We parsed the "Individual Test Solutions" section by locating the section marker in the raw HTML.
- **Phase 2 — Detail pages:** Visited each assessment's detail URL to extract: full description, job levels, supported languages, and duration in minutes. Used `<h4>` tag parsing and regex for structured fields like `Completion Time in minutes = N`.
- **Scope:** Scraped **377 Individual Test Solutions** and **~70 Pre-packaged Job Solutions** (added after initial evaluation revealed that train labels reference Pre-packaged solutions).
- **Caching:** All scraped data is saved to JSON files (`shl_catalog.json`, `shl_prepackaged.json`) so scraping runs only once. Rate-limited at 1–1.5 seconds between requests.

### 2.2 Data Processing

Each assessment is converted into a rich text representation combining all metadata:

> *"Assessment: Core Java (Entry Level). Description: Multi-choice test measuring knowledge of basic Java constructs, OOP concepts... Test Type: Knowledge & Skills. Job Levels: Mid-Professional. Duration: 13 minutes."*

**Why?** Embedding models perform best on natural text. Concatenating structured fields into a coherent sentence gives the model context about what each assessment tests, who it's for, and how long it takes.

---

## 3. Embedding & Vector Store

### 3.1 Embedding Model

We use **`all-mpnet-base-v2`** from sentence-transformers (768-dimensional embeddings). Initially we tried `all-MiniLM-L6-v2` (384-dim), but `mpnet` showed better retrieval quality on our train set.

**Why sentence-transformers?** They produce dense semantic embeddings where similar concepts cluster together — "Java programming" and "Core Java test" will have high cosine similarity even without exact keyword overlap.

### 3.2 FAISS Index

We build a **FAISS `IndexFlatIP`** (inner product) index over L2-normalized embeddings, which is equivalent to cosine similarity search. The index is saved to disk (`faiss_index.bin`) so it loads instantly on app restart.

**Why FAISS?** It provides exact nearest-neighbor search over the full catalog (~450 vectors) with sub-millisecond latency. For our catalog size, exact search is fast enough and avoids the accuracy loss of approximate methods.

---

## 4. LLM Integration

We loaded **Qwen/Qwen3-1.7B** via HuggingFace Transformers with float16 precision across AMD GPUs. The LLM serves two roles in the pipeline:

- **Query analysis:** For complex job descriptions (500+ words), the LLM extracts key hiring requirements — technical skills, soft skills, job level, and duration constraints — producing focused search phrases.
- **Candidate re-ranking:** Given retrieved candidates, the LLM selects and orders the most relevant assessments, balancing technical and behavioral test types.

**Why Qwen 1.7B?** It's small enough to run locally with fast inference, while being capable enough for structured extraction and ranking tasks. No API costs or rate limits.

---

## 5. Retrieval Pipeline — Iterative Optimization

### 5.1 Baseline (v1) — Pure Semantic Search + LLM Re-ranking
Single FAISS query → top-30 candidates → Qwen re-ranks to top-10.
**Result: Mean Recall@10 = 0.1844**

*Problem:* Semantic search alone misses assessments where the name/description doesn't textually overlap with the query. LLM re-ranking sometimes pushed relevant items out of the top 10.

### 5.2 Improvement (v2) — Extended Catalog + Better Model
Added Pre-packaged solutions to the catalog. Switched from `MiniLM` to `mpnet`. Increased retrieval pool to top-100. Added multi-query retrieval: Qwen decomposes the query into aspects, each searched separately, results merged.
**Result: Mean Recall@10 = 0.2133**

### 5.3 Final (v3) — Hybrid Semantic + Keyword Search
Combined normalized semantic scores (FAISS cosine) with keyword overlap scores (Jaccard-like set intersection of query tokens vs assessment tokens). Swept the weighting parameter `alpha` from 0.3 to 1.0; best at **alpha = 0.7** (70% semantic, 30% keyword). Removed LLM re-ranking to prevent good candidates from being reordered poorly.
**Result: Mean Recall@10 = 0.2167**

**Why hybrid?** Pure semantic search misses exact skill mentions (e.g., query says "Java" but embedding ranks a generically similar assessment higher). Keyword matching catches these direct matches. The alpha parameter balances exploration (semantic) with precision (keyword).

---

## 6. Evaluation

- **Metric:** Mean Recall@K (K=10), measuring what fraction of human-labeled relevant assessments appear in our top-10 recommendations, averaged over all queries.
- **Train set:** 10 queries with 65 human-labeled query–assessment pairs.
- **URL normalization:** Slug-based comparison (extracting the last URL path segment) to handle format differences between `/solutions/products/` and `/products/` paths.

### Root Cause Analysis of Remaining Errors

Diagnostic analysis revealed a fundamental semantic gap: generic assessments like OPQ32r (personality) and Verify Verbal/Numerical (cognitive aptitude) are labeled as relevant for most roles but have zero textual similarity to any specific JD. These items consistently ranked 300+ in semantic search. Additionally, some JDs mention high-level domains (e.g., "AI/ML") while the expected assessments test specific skills (e.g., SQL, Selenium) — a gap that embedding models cannot bridge without external knowledge.

---

## 7. Web Application

Built with **Streamlit**, the frontend provides:
- Text area for natural language queries or full job descriptions
- Adjustable parameters (number of results, semantic vs keyword weight)
- Results table with assessment name, test type, duration, remote testing, relevance score, and clickable catalog links
- Detailed expandable view with per-assessment metrics

All heavy resources (catalog, embeddings, FAISS index) are cached to disk and loaded on startup for fast response times.

---

## 8. Technology Stack

| Component | Technology | Role |
|---|---|---|
| Web Scraping | requests, BeautifulSoup4 | Catalog data collection |
| Embeddings | sentence-transformers (all-mpnet-base-v2) | Semantic representation |
| Vector Store | FAISS (IndexFlatIP) | Fast similarity search |
| LLM | Qwen/Qwen3-1.7B | Query understanding, skill extraction |
| Hybrid Search | NumPy | Keyword + semantic score fusion |
| Evaluation | pandas, NumPy | Mean Recall@K computation |
| Frontend | Streamlit | Interactive web application |
| Runtime | PyTorch 2.10 + ROCm 7.1 | GPU acceleration |

---

## 9. Files & Artifacts

| File | Description |
|---|---|
| `shl_rag_recommendation.ipynb` | Full experiment notebook: scraping, embeddings, RAG pipeline, evaluation, predictions |
| `app.py` | Streamlit web application |
| `data/shl_catalog.json` | Scraped Individual Test Solutions (377 items) |
| `data/shl_prepackaged.json` | Scraped Pre-packaged Job Solutions (~70 items) |
| `data/faiss_index.bin` | Saved FAISS index for instant loading |
| `data/embeddings.npy` | Saved assessment embeddings |
| `predictions.csv` | Final predictions on the test set (9 queries × 10 recommendations) |
