import os

def _first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]

# ── Base directory ────────────────────────────────────────────
BASE_DIR     = os.environ.get("RECIPE_BASE_DIR", os.path.join(os.path.dirname(__file__), "data"))
PROJECT_ROOT = os.path.dirname(__file__)

CHROMA_DB_PATH    = os.path.join(BASE_DIR, "chroma_db")
BM25_INDEX_PATH   = os.path.join(BASE_DIR, "bm25_index.pkl")
DATAFRAME_PATH    = os.path.join(BASE_DIR, "recipes_clean.parquet")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "eval_results.csv")

# ── Dataset ───────────────────────────────────────────────────
CSV_PATH = _first_existing_path(
    os.path.join(BASE_DIR, "full_dataset.csv"),
    os.path.join(PROJECT_ROOT, "person3", "data", "full_dataset.csv"),
)
NUM_RECIPES     = 50_000
MIN_INGREDIENTS = 3

# ── Models ────────────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "recipes"

# retrieval
EMBED_BATCH_SIZE  = 256
BM25_CANDIDATES   = 50      #  30-100
DENSE_CANDIDATES  = 50      #  30-100
RRF_K             = 60      #  10-120
TOP_K             = 5       #  3-10

# reranker
USE_RERANKER  = True        # set False to skip reranking and compare
RERANK_TOP_N  = 15          # candidates fed to cross-encoder (10-25)

# eval
NUM_EVAL_QUERIES = 10

METRIC_WEIGHTS = {
    "faithfulness":         0.35,
    "answer_relevance":     0.30,
    "contextual_precision": 0.20,
    "contextual_recall":    0.15,
}

OVERLAP_THRESHOLD = 3