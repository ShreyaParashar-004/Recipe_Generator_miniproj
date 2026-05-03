# Usage: python test_eval.py

import pandas as pd
from config import DATAFRAME_PATH, METRIC_WEIGHTS, OVERLAP_THRESHOLD

from retrieval.embedder import load_embedder
from retrieval.bm25_retriever import load_bm25_index
from retrieval.vector_store import load_vector_store
from retrieval.hybrid_retriever import hybrid_retrieve
from evaluation.query_expander import optimize_query
from evaluation.ragas_eval import run_evaluation, save_results

RAW_QUERIES = [
    "i have leftover boiled potatoes and some onions, what do I make?",
    "i want a cake but without chocolate",
    "i have eggs, bread, and butter, quick breakfast",
    "i want spicy chicken curry under 30 minutes",
    "gluten free cookies with oats",
    "pasta with no cream, healthy",
    "i have bananas going bad, easy dessert",
    "something with rice and vegetables, no meat",
]

print("=" * 55)
print("Loading from disk...")
print("=" * 55)
df         = pd.read_parquet(DATAFRAME_PATH)
bm25       = load_bm25_index()
embedder   = load_embedder()
collection = load_vector_store()
print(f"Loaded {len(df)} recipes.\n")

print("=" * 55)
print("Building eval rows...")
print("=" * 55)
eval_rows = []
for raw_query in RAW_QUERIES:
    opt = optimize_query(raw_query)
    retrieved = hybrid_retrieve(
        query=opt["optimized_query"],
        exclusions=opt["exclusions"],
        df=df, bm25=bm25,
        collection=collection, embedder=embedder,
    )
    contexts         = [r["full_text"] for r in retrieved]
    simulated_answer = retrieved[0]["full_text"] if retrieved else ""
    ingredients      = [w for w in opt["optimized_query"].split() if len(w) > 3]
    eval_rows.append({
        "query":                 raw_query,
        "answer":                simulated_answer,
        "contexts":              contexts,
        "available_ingredients": ingredients,
    })
    print(f"  {raw_query[:50]}...")
    print(f"  → {retrieved[0]['title'] if retrieved else 'None'}\n")

print("=" * 55)
print("Running evaluation...")
print("=" * 55)
df_results = run_evaluation(eval_rows=eval_rows, embedder=embedder, model_name="hybrid_rag")
save_results(df_results)

print("\n── Results ───────────────────────────────────")
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.3f}".format)
print(df_results[[
    "query", "faithfulness", "answer_relevance",
    "contextual_precision", "contextual_recall", "overall_score"
]].to_string(index=False))