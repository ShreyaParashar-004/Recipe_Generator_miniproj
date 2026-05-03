# Runs evaluation across multiple weight configurations and prints a comparison table.
# Shows which weight setup produces the best overall score for your system.

import pandas as pd
from config import DATAFRAME_PATH
from retrieval.embedder import load_embedder
from retrieval.bm25_retriever import load_bm25_index
from retrieval.vector_store import load_vector_store
from retrieval.hybrid_retriever import hybrid_retrieve
from evaluation.query_expander import optimize_query
from evaluation.ragas_eval import run_evaluation, save_results

# ── Weight configurations to test ────────────────────────────
# Each entry: (label, weights_dict)
EXPERIMENTS = [
    ("equal_weights", {
        "faithfulness": 0.25, "answer_relevance": 0.25,
        "contextual_precision": 0.25, "contextual_recall": 0.25,
    }),
    ("faithfulness_heavy", {
        "faithfulness": 0.45, "answer_relevance": 0.25,
        "contextual_precision": 0.15, "contextual_recall": 0.15,
    }),
    ("relevance_heavy", {
        "faithfulness": 0.20, "answer_relevance": 0.45,
        "contextual_precision": 0.20, "contextual_recall": 0.15,
    }),
    ("retrieval_heavy", {
        "faithfulness": 0.20, "answer_relevance": 0.20,
        "contextual_precision": 0.40, "contextual_recall": 0.20,
    }),
    ("recall_heavy", {
        "faithfulness": 0.20, "answer_relevance": 0.20,
        "contextual_precision": 0.20, "contextual_recall": 0.40,
    }),
    ("default_config", {
        "faithfulness": 0.35, "answer_relevance": 0.30,
        "contextual_precision": 0.20, "contextual_recall": 0.15,
    }),
]

# ── Test queries ──────────────────────────────────────────────
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


def build_eval_rows(df, bm25, collection, embedder) -> list[dict]:
    """Build eval rows once — reuse across all weight experiments."""
    print("Building eval rows (retrieval runs once)...")
    eval_rows = []
    for raw_query in RAW_QUERIES:
        opt = optimize_query(raw_query)
        retrieved = hybrid_retrieve(
            query=opt["optimized_query"],
            exclusions=opt["exclusions"],
            df=df, bm25=bm25,
            collection=collection, embedder=embedder,
        )
        contexts = [r["full_text"] for r in retrieved]
        simulated_answer = retrieved[0]["full_text"] if retrieved else ""
        ingredients = [
            w for w in opt["optimized_query"].split() if len(w) > 3
        ]
        eval_rows.append({
            "query":                 raw_query,
            "answer":                simulated_answer,
            "contexts":              contexts,
            "available_ingredients": ingredients,
        })
        print(f"  Built: {raw_query[:50]}...")
    return eval_rows


def run_experiments():
    # Load everything from disk
    print("=" * 60)
    print("Loading models from disk...")
    print("=" * 60)
    df         = pd.read_parquet(DATAFRAME_PATH)
    bm25       = load_bm25_index()
    embedder   = load_embedder()
    collection = load_vector_store()
    print(f"Loaded {len(df)} recipes.\n")

    # Build eval rows once
    eval_rows = build_eval_rows(df, bm25, collection, embedder)

    # Run each weight configuration
    summary_rows = []
    all_results  = []

    print("\n" + "=" * 60)
    print("Running weight experiments...")
    print("=" * 60)

    for label, weights in EXPERIMENTS:
        print(f"\n── Experiment: {label} ──────────────────────")
        for k, v in weights.items():
            print(f"   {k:<25} {v}")
        print()

        df_results = run_evaluation(
            eval_rows=eval_rows,
            embedder=embedder,
            model_name=label,
            weights=weights,
        )
        all_results.append(df_results)

        summary_rows.append({
            "experiment":            label,
            "w_faithfulness":        weights["faithfulness"],
            "w_answer_relevance":    weights["answer_relevance"],
            "w_contextual_precision":weights["contextual_precision"],
            "w_contextual_recall":   weights["contextual_recall"],
            "mean_faithfulness":     round(df_results["faithfulness"].mean(), 4),
            "mean_answer_relevance": round(df_results["answer_relevance"].mean(), 4),
            "mean_cp":               round(df_results["contextual_precision"].mean(), 4),
            "mean_cr":               round(df_results["contextual_recall"].mean(), 4),
            "mean_overall":          round(df_results["overall_score"].mean(), 4),
        })

    # comparing results
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_overall", ascending=False)

    print("\n" + "=" * 60)
    print("RESULTS — sorted by overall score (best first)")
    print("=" * 60)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary_df[[
        "experiment", "mean_faithfulness", "mean_answer_relevance",
        "mean_cp", "mean_cr", "mean_overall"
    ]].to_string(index=False))

    best = summary_df.iloc[0]
    print(f"\n Best config: {best['experiment']} — overall score {best['mean_overall']:.4f}")

    # Save
    summary_path = "data/weight_experiment_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f" Summary saved to {summary_path}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("data/weight_experiment_all_rows.csv", index=False)
    print(f" All rows saved to data/weight_experiment_all_rows.csv")


if __name__ == "__main__":
    run_experiments()