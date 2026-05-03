# Merges BM25 sparse + dense vector results using Reciprocal Rank Fusion (RRF)
# Then reranks with a cross-encoder for better precision.

import pandas as pd
from config import BM25_CANDIDATES, DENSE_CANDIDATES, RRF_K, TOP_K, RERANK_TOP_N, USE_RERANKER
from retrieval.bm25_retriever import query_bm25
from retrieval.vector_store import query_dense


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """
    Merge multiple ranked lists into one using RRF.
    k=60 is the standard default (Cormack et al. 2009).
    Try 10-120: lower = top ranks matter more, higher = flatter blend.
    """
    scores: dict[int, float] = {}
    for ranked_list in rankings:
        for rank, doc_idx in enumerate(ranked_list):
            if doc_idx not in scores:
                scores[doc_idx] = 0.0
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def _filter_by_exclusions(results: list[dict], exclusions: list[str]) -> list[dict]:
    """Remove recipes whose full_text contains any excluded ingredient."""
    if not exclusions:
        return results
    filtered = []
    for recipe in results:
        full_text_lower = recipe["full_text"].lower()
        if not any(excl in full_text_lower for excl in exclusions):
            filtered.append(recipe)
    return filtered


def hybrid_retrieve(
    query: str,
    df: pd.DataFrame,
    bm25,
    collection,
    embedder,
    exclusions: list[str] = None,
    top_k: int = TOP_K,
    bm25_candidates: int = BM25_CANDIDATES,
    dense_candidates: int = DENSE_CANDIDATES,
) -> list[dict]:
    """
    Hybrid retrieval: BM25 sparse + MiniLM dense, merged with RRF,
    exclusion filtering, and optional cross-encoder reranking.

    Pipeline:
        1. BM25 sparse retrieval      -> top bm25_candidates
        2. Dense vector retrieval     -> top dense_candidates
        3. RRF fusion                 -> single ranked list
        4. Exclusion filtering        -> remove banned ingredients
        5. Cross-encoder reranking    -> if USE_RERANKER=True in config
        6. Return top_k results

    Args:
        query:            optimized keyword query from optimize_query()
        df:               cleaned recipes DataFrame
        bm25:             loaded BM25Okapi index
        collection:       loaded ChromaDB collection
        embedder:         loaded SentenceTransformer model
        exclusions:       ingredients to exclude (from optimize_query)
        top_k:            number of final results to return
        bm25_candidates:  pool size for BM25
        dense_candidates: pool size for dense search

    Returns:
        List of dicts: title, ingredients, full_text, df_index,
        and rerank_score if reranking was used.
    """
    exclusions = exclusions or []

    if exclusions:
        print(f"[hybrid_retriever] Excluding: {exclusions}")

    # Step 1 & 2: Sparse + dense retrieval
    bm25_indices  = query_bm25(bm25, query, n_results=bm25_candidates)
    dense_indices = query_dense(collection, embedder, query, n_results=dense_candidates)

    # Step 3: RRF fusion
    fused = reciprocal_rank_fusion([bm25_indices, dense_indices])

    # Step 4: Build candidate pool + filter exclusions
    candidates = []
    for idx in fused:
        row = df.iloc[idx]
        candidates.append({
            "title":       row["title"],
            "ingredients": row["ingredients"],
            "full_text":   row["full_text"],
            "df_index":    int(idx)
        })

    filtered = _filter_by_exclusions(candidates, exclusions)

    if exclusions and len(filtered) < top_k:
        print(
            f"[hybrid_retriever] Warning: only {len(filtered)} results after "
            f"exclusion filter (wanted {top_k}). "
            f"Try increasing BM25_CANDIDATES/DENSE_CANDIDATES in config.py."
        )

    final_candidates = filtered if filtered else candidates

    # Step 5: Cross-encoder reranking (optional, toggle in config)
    if USE_RERANKER:
        from retrieval.reranker import rerank
        pool = final_candidates[:RERANK_TOP_N]
        print(f"[hybrid_retriever] Reranking top {len(pool)} candidates...")
        final = rerank(query, pool, top_n=top_k)
    else:
        final = final_candidates[:top_k]

    return final