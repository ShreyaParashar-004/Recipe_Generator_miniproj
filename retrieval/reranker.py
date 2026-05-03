# Cross-encoder reranking — runs after RRF fusion to rescore top candidates.
# Uses a small cross-encoder model that scores query-document pairs properly,
# unlike bi-encoders (MiniLM) which encode query and doc independently.
#
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2
#   - 80MB, fast enough on CPU for top-10 reranking
#   - Trained on MS MARCO passage retrieval
#   - Returns a relevance score per (query, document) pair

from sentence_transformers import CrossEncoder
import torch

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None  # module-level cache — load once, reuse


def load_reranker() -> CrossEncoder:
    """
    Load the cross-encoder model. Cached after first call.
    CPU is fine for reranking top-10 — takes ~0.5s per query.
    """
    global _reranker
    if _reranker is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading reranker '{RERANKER_MODEL}' on {device}...")
        _reranker = CrossEncoder(RERANKER_MODEL, device=device)
        print("Reranker ready.")
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Rerank candidate recipes using cross-encoder scores.

    Cross-encoder sees the full (query, document) pair together,
    giving much better relevance scores than cosine similarity alone.

    Args:
        query:      original or optimized query string
        candidates: list of recipe dicts from hybrid_retrieve()
                    each must have 'full_text' key
        top_n:      how many to return after reranking

    Returns:
        Top-n recipes sorted by cross-encoder relevance score (descending),
        with 'rerank_score' added to each dict.
    """
    reranker = load_reranker()

    if not candidates:
        return candidates

    # Build (query, document) pairs for the cross-encoder
    # Use first 512 chars of full_text — cross-encoder has token limit
    pairs = [(query, c["full_text"][:512]) for c in candidates]

    # Score all pairs — returns numpy array of floats
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Attach scores to candidates
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = round(float(scores[i]), 4)

    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_n]