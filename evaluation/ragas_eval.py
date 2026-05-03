import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import (
    EVAL_RESULTS_PATH,
    EMBEDDING_MODEL,
    METRIC_WEIGHTS,
    OVERLAP_THRESHOLD,
)
import torch


#metrics

def compute_faithfulness(answer: str, retrieved_docs: list[str], available_ingredients: list[str] = []) -> float:
    """
    Faithfulness: fraction of answer ingredients that appear in retrieved docscj
    or in the user's stated available ingredients.

    Score = non-hallucinated ingredients / total ingredients in answer.
    Target: >= 0.70
    """
    answer_lower = answer.lower()
    if "ingredients:" in answer_lower:
        ing_section = answer_lower.split("ingredients:")[1].split("instructions:")[0]
    else:
        ing_section = answer_lower

    answer_tokens = set(ing_section.replace(",", " ").replace("\n", " ").split())
    answer_tokens = {t for t in answer_tokens if len(t) > 3}

    if not answer_tokens:
        return 1.0

    reference_text = " ".join(retrieved_docs + available_ingredients).lower()
    reference_tokens = set(reference_text.replace(",", " ").replace("\n", " ").split())

    grounded = answer_tokens & reference_tokens
    score = len(grounded) / len(answer_tokens)
    return round(score, 4)


def compute_answer_relevance(query: str, answer: str, embedder) -> float:
    """
    Answer Relevance: cosine similarity between query embedding and answer embedding.
    Target: >= 0.75
    """
    vecs = embedder.encode([query, answer], normalize_embeddings=True)
    score = float(np.dot(vecs[0], vecs[1]))
    return round(score, 4)


def compute_contextual_precision(answer: str, retrieved_docs: list[str], overlap_threshold: int = OVERLAP_THRESHOLD) -> float:
    """
    Contextual Precision: fraction of retrieved docs that share >= overlap_threshold
    ingredient tokens with the answer.

    overlap_threshold is pulled from config.OVERLAP_THRESHOLD (default 3).
    Try values 2–5 to tune strictness.
    Target: >= 0.65
    """
    if not retrieved_docs:
        return 0.0

    answer_tokens = set(answer.lower().split())
    precise = 0

    for doc in retrieved_docs:
        doc_tokens = set(doc.lower().split())
        overlap = len(answer_tokens & doc_tokens)
        if overlap >= overlap_threshold:
            precise += 1

    score = precise / len(retrieved_docs)
    return round(score, 4)


def compute_contextual_recall(answer: str, available_ingredients: list[str]) -> float:
    """
    Contextual Recall: fraction of user's available ingredients that appear in the answer.
    Target: >= 0.65
    """
    if not available_ingredients:
        return 1.0

    answer_lower = answer.lower()
    used = sum(1 for ing in available_ingredients if ing.lower() in answer_lower)
    score = used / len(available_ingredients)
    return round(score, 4)


def compute_overall_score(
    faithfulness: float,
    answer_relevance: float,
    contextual_precision: float,
    contextual_recall: float,
    weights: dict = METRIC_WEIGHTS,
) -> float:
    """
    Weighted overall score across all 4 metrics.
    Weights are pulled from config.METRIC_WEIGHTS.
    Must sum to 1.0 — see config.py comments for tuning tips.
    """
    score = (
        weights["faithfulness"]         * faithfulness +
        weights["answer_relevance"]     * answer_relevance +
        weights["contextual_precision"] * contextual_precision +
        weights["contextual_recall"]    * contextual_recall
    )
    return round(score, 4)


# eval 

def run_evaluation(
    eval_rows: list[dict],
    embedder,
    model_name: str = "hybrid_rag",
    weights: dict = METRIC_WEIGHTS,
) -> pd.DataFrame:
    """
    Run all 4 metrics + weighted overall score over a list of eval rows.

    Each eval_row must have:
        - query                  (str)
        - answer                 (str)
        - contexts               (list of retrieved doc strings)
        - available_ingredients  (list of str, optional)

    Args:
        eval_rows:  list of dicts as described above
        embedder:   loaded SentenceTransformer
        model_name: label for this model (baseline_llm / naive_rag / hybrid_rag)
        weights:    metric weight dict (defaults to config.METRIC_WEIGHTS)

    Returns:
        DataFrame with one row per query and columns for each metric + overall_score
    """
    # Validate weights must sum to ~1.0
    weight_sum = sum(weights.values())
    if not (0.99 <= weight_sum <= 1.01):
        raise ValueError(f"METRIC_WEIGHTS must sum to 1.0, got {weight_sum:.3f}. Check config.py.")

    results = []

    for i, row in enumerate(eval_rows):
        query      = row["query"]
        answer     = row["answer"]
        contexts   = row.get("contexts", [])
        avail_ings = row.get("available_ingredients", [])

        faithfulness         = compute_faithfulness(answer, contexts, avail_ings)
        answer_relevance     = compute_answer_relevance(query, answer, embedder)
        contextual_precision = compute_contextual_precision(answer, contexts)
        contextual_recall    = compute_contextual_recall(answer, avail_ings)
        overall              = compute_overall_score(
            faithfulness, answer_relevance, contextual_precision, contextual_recall, weights
        )

        results.append({
            "model":                 model_name,
            "query":                 query,
            "answer":                answer[:200],
            "faithfulness":          faithfulness,
            "answer_relevance":      answer_relevance,
            "contextual_precision":  contextual_precision,
            "contextual_recall":     contextual_recall,
            "overall_score":         overall,
        })

        print(
            f"[{i+1}/{len(eval_rows)}] F={faithfulness:.2f} "
            f"AR={answer_relevance:.2f} "
            f"CP={contextual_precision:.2f} "
            f"CR={contextual_recall:.2f} "
            f"Overall={overall:.2f} | {query[:40]}..."
        )

    df_results = pd.DataFrame(results)

    # table output
    print("\n── Summary ──────────────────────────────────")
    print(f"Model: {model_name}")
    print(f"\n  Weights used:")
    for metric, w in weights.items():
        print(f"    {metric:<25} {w:.2f}")
    print(f"  Overlap threshold:        {OVERLAP_THRESHOLD}")
    print()
    print(f"  Faithfulness:          {df_results['faithfulness'].mean():.3f}  (target ≥ 0.70)  [w={weights['faithfulness']}]")
    print(f"  Answer Relevance:      {df_results['answer_relevance'].mean():.3f}  (target ≥ 0.75)  [w={weights['answer_relevance']}]")
    print(f"  Contextual Precision:  {df_results['contextual_precision'].mean():.3f}  (target ≥ 0.65)  [w={weights['contextual_precision']}]")
    print(f"  Contextual Recall:     {df_results['contextual_recall'].mean():.3f}  (target ≥ 0.65)  [w={weights['contextual_recall']}]")
    print(f"  ── Overall Score:      {df_results['overall_score'].mean():.3f}")

    return df_results


def save_results(df_results: pd.DataFrame, path: str = EVAL_RESULTS_PATH):
    """Append results to the eval CSV (creates it if it doesn't exist)."""
    if pd.io.common.file_exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df_results], ignore_index=True)
    else:
        combined = df_results

    combined.to_csv(path, index=False)
    print(f"Results saved to {path}")