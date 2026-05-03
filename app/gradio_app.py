
# app/gradio_app.py — Integrated with Person 3 + Themed UI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image
import re

from config import (
    DATAFRAME_PATH, EMBEDDING_MODEL, BM25_CANDIDATES, DENSE_CANDIDATES, RRF_K, TOP_K
)
from evaluation.query_expander import optimize_query
from retrieval.embedder import load_embedder
from retrieval.vector_store import load_vector_store
from retrieval.bm25_retriever import load_bm25_index
from retrieval.hybrid_retriever import hybrid_retrieve
from generation.llm import generate_recipe
from app.clip_classifier import load_clip_model, classify_dish
from substitution.substitution import (
    get_substitutes,
    estimate_cost_inr,
    evaluate_recipe,
    normalize_dietary_restrictions,
)
from evaluation.ragas_eval import (
    compute_faithfulness,
    compute_answer_relevance,
    compute_contextual_precision,
    compute_contextual_recall,
    compute_overall_score,
    save_results,
)

import pandas as pd

# ── Global state for retrieved docs (for evaluation) ──────────
global_retrieved_docs = []
global_query = ""

# ── Load all models at startup ─────────────────────────────────
print("Loading embedder...")
embedder = load_embedder()
print("Loading ChromaDB...")
collection = load_vector_store()
print("Loading BM25 index...")
bm25 = load_bm25_index()
print("Loading recipes dataframe...")
df = pd.read_parquet(DATAFRAME_PATH)
print("Loading CLIP model...")
clip_model, clip_processor, clip_device = load_clip_model()
print("All models loaded. Starting Gradio app...")


# ── CSS matching the slide design ─────────────────────────────
CUSTOM_CSS = """
/* ════════════════════════════════════════════════════════════════════ */
/* BEAUTIFUL & MODERN UI STYLING FOR RECIPERAG                        */
/* ════════════════════════════════════════════════════════════════════ */

/* ── GLOBAL SETTINGS ── */
* {
    box-sizing: border-box;
}

html, body, .gradio-container {
    background: linear-gradient(135deg, #FFF9F0 0%, #FFF5E6 100%) !important;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif !important;
    color: #2C1810 !important;
}

/* ── HEADER & TITLE ── */
.gr-markdown h1 {
    background: linear-gradient(90deg, #C8601A 0%, #E8860A 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2.5em !important;
    font-weight: 900 !important;
    text-align: center !important;
    margin-bottom: 10px !important;
    text-shadow: none !important;
}

.gr-markdown h2 {
    color: #C8601A !important;
    font-size: 1.6em !important;
    font-weight: 700 !important;
    border-left: 5px solid #E8860A !important;
    padding-left: 12px !important;
    margin-top: 20px !important;
    margin-bottom: 12px !important;
}

.gr-markdown h3 {
    color: #7B3F00 !important;
    font-size: 1.2em !important;
    font-weight: 600 !important;
    margin-top: 12px !important;
}

.tabs {
    border-bottom: 3px solid #E8C99A !important;
}

.tab-nav {
    gap: 8px !important;
    background-color: transparent !important;
}

.tab-nav button {
    background: linear-gradient(180deg, #FFF0D6 0%, #FFE8C0 100%) !important;
    color: #7B3F00 !important;
    font-weight: 700 !important;
    border-radius: 12px 12px 0 0 !important;
    font-size: 16px !important;
    border: 2px solid #D4A96A !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.tab-nav button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(200, 96, 26, 0.2) !important;
}

.tab-nav button.selected {
    background: linear-gradient(180deg, #C8601A 0%, #A0420D 100%) !important;
    color: white !important;
    border-color: #7B3F00 !important;
    box-shadow: 0 6px 12px rgba(200, 96, 26, 0.3) !important;
}

.block {
    background: linear-gradient(135deg, #FFFDF7 0%, #FFF8F0 100%) !important;
    border: 2px solid #E8C99A !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 12px rgba(123, 63, 0, 0.1) !important;
    transition: all 0.3s ease !important;
}

.block:hover {
    box-shadow: 0 6px 16px rgba(200, 96, 26, 0.15) !important;
    border-color: #C8601A !important;
}

.gr-form {
    background: linear-gradient(135deg, #FFFDF7 0%, #FFF8F0 100%) !important;
    border: 2px solid #E8C99A !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

label, .gr-form label, .label-wrap label {
    color: #7B3F00 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

label span {
    color: #7B3F00 !important;
    font-weight: 700 !important;
}

textarea, input[type=text], input[type=number] {
    background-color: #FFFDF7 !important;
    border: 2px solid #D4A96A !important;
    border-radius: 12px !important;
    color: #2C1810 !important;
    font-size: 14px !important;
    padding: 12px 14px !important;
    transition: all 0.3s ease !important;
}

textarea:focus, input[type=text]:focus {
    border-color: #C8601A !important;
    box-shadow: 0 0 0 3px rgba(200, 96, 26, 0.1) !important;
    outline: none !important;
}

.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #C8601A 0%, #A0420D 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    box-shadow: 0 4px 12px rgba(200, 96, 26, 0.3) !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover, button.primary:hover {
    background: linear-gradient(135deg, #A0420D 0%, #7B3F00 100%) !important;
    transform: translateY(-2px) !important;
}

input[type=range] {
    accent-color: #C8601A !important;
}

.gr-markdown table th {
    color: #7B3F00 !important;
    font-weight: 700 !important;
    padding: 12px !important;
    border: 1px solid #E8C99A !important;
    background-color: #FFF0D6 !important;
}

.gr-markdown table td {
    padding: 12px !important;
    border: 1px solid #E8C99A !important;
    color: #2C1810 !important;
}

.gr-markdown table tbody tr:nth-child(even) {
    background-color: #FFFDF7 !important;
}

::-webkit-scrollbar { width: 10px !important; }
::-webkit-scrollbar-track { background: #FFF5E6 !important; }
::-webkit-scrollbar-thumb { background: #D4A96A !important; border-radius: 5px !important; }
::-webkit-scrollbar-thumb:hover { background: #C8601A !important; }
"""


# ── Helper: Parse generated recipe text ───────────────────────
def parse_recipe_output(recipe_text: str):
    ingredients = []
    steps = []

    ing_match = re.search(
        r"Ingredients:\s*(.*?)(?=Instructions:|$)",
        recipe_text, re.DOTALL | re.IGNORECASE
    )
    if ing_match:
        for line in ing_match.group(1).strip().split("\n"):
            line = line.strip().lstrip("-•*").strip()
            if line:
                ingredients.append(line)

    inst_match = re.search(
        r"Instructions:\s*(.*?)(?=Estimated Time:|Serves:|$)",
        recipe_text, re.DOTALL | re.IGNORECASE
    )
    if inst_match:
        for line in inst_match.group(1).strip().split("\n"):
            line = re.sub(r"^\d+\.\s*", "", line.strip()).strip()
            if line:
                steps.append(line)

    time_match = re.search(r"Estimated Time:\s*(.+)", recipe_text, re.IGNORECASE)
    estimated_time = time_match.group(1).strip() if time_match else ""

    return ingredients, steps, estimated_time


# ── Core pipeline ──────────────────────────────────────────────
def run_pipeline(
    text_query, dish_image, available_ingredients,
    diet, appliance, time_limit, budget
):
    hf_token = os.environ.get("HF_TOKEN", "")

    if dish_image is not None:
        pil_image = Image.fromarray(dish_image).convert("RGB")
        dish_name = classify_dish(pil_image, clip_model, clip_processor, clip_device)
        query = dish_name
        query_display = f"Detected dish: **{dish_name}**"
        if text_query.strip():
            query = f"{dish_name} {text_query.strip()}"
            query_display += f" + your notes: *{text_query.strip()}*"
    elif text_query.strip():
        query = text_query.strip()
        query_display = f" Query: **{query}**"
    else:
        return " Please enter a query or upload a dish photo.", "", "", "", ""

    # ── Build constraints ──────────────────────────────────────
    constraints = {}
    if available_ingredients.strip():
        constraints["ingredients"] = [
            i.strip() for i in available_ingredients.split(",") if i.strip()
        ]
    if diet and diet != "None":
        constraints["diet"] = diet
    if appliance and appliance != "None":
        constraints["appliance"] = appliance
    if time_limit.strip():
        constraints["time"] = time_limit.strip()
    if budget.strip():
        constraints["budget"] = budget.strip()

    # ── Query optimization ─────────────────────────────────────
    if dish_image is not None:
        opt = optimize_query(text_query.strip()) if text_query.strip() else {}
        optimized_query = f"{query} {opt.get('optimized_query', '')}".strip()
        exclusions = opt.get("exclusions", [])
        detected_diet = opt.get("diet")
    else:
        opt = optimize_query(query)
        optimized_query = opt["optimized_query"]
        exclusions = opt["exclusions"]
        detected_diet = opt.get("diet")

    if detected_diet and (not diet or diet == "None"):
        constraints["diet"] = detected_diet

    # ── Retrieval ──────────────────────────────────────────────
    try:
        retrieved = hybrid_retrieve(
            query=optimized_query,
            exclusions=exclusions,
            df=df, bm25=bm25,
            collection=collection, embedder=embedder
        )
    except Exception as e:
        return query_display, f" Retrieval error: {str(e)}", "", "", ""

    if not retrieved:
        return query_display, " No relevant recipes found.", "", "", ""

    global global_retrieved_docs
    global global_query
    global_retrieved_docs = [r['full_text'] for r in retrieved]
    global_query = query

    retrieved_titles = "\n".join([f"{r['title']}" for r in retrieved])
    retrieval_info = f"{query_display}\n\n **Retrieved References:**\n{retrieved_titles}"

    try:
        recipe_output = generate_recipe(
            query=query, retrieved_recipes=retrieved,
            hf_token=hf_token, constraints=constraints
        )
    except Exception as e:
        return retrieval_info, f" Generation error: {str(e)}", "", "", ""

    # Cost + Evaluation
    try:
        avail_list = [i.strip() for i in available_ingredients.split(",") if i.strip()]
        diet_list = normalize_dietary_restrictions(diet) if diet and diet != "None" else []
        appliance_list = [appliance.lower()] if appliance and appliance != "None" else []

        parsed_ingredients, parsed_steps, _ = parse_recipe_output(recipe_output)

        cost_result = estimate_cost_inr(parsed_ingredients, servings=2)
        cost_display = (
            f"## {cost_result.get('budget_category', ' Cost Estimate')}\n\n"
            f"**Total Cost:** {cost_result.get('total_cost', '₹?')}  "
            f"**Per Serving:** {cost_result.get('cost_per_serving', '₹?')}\n\n"
            f"**Breakdown:**\n"
        )
        for item in cost_result.get("breakdown", []):
            cost_display += f"- {item['ingredient']}: {item['cost']}\n"

        eval_result = evaluate_recipe(
            recipe_steps=parsed_steps,
            recipe_ingredients=parsed_ingredients,
            available_ingredients=avail_list if avail_list else parsed_ingredients,
            dietary_restrictions=diet_list if diet_list else None,
            available_appliances=appliance_list if appliance_list else None,
            estimated_time_minutes=30
        )

        score = eval_result.get("final_reward", 0)
        if score >= 0.75:
            verdict = f" Excellent Recipe!"
        elif score >= 0.55:
            verdict = f" Good Recipe"
        elif score >= 0.35:
            verdict = f" Fair Recipe"
        else:
            verdict = f" Needs Improvement"

        eval_display = (
            f"## {verdict}  Score: {score}\n\n"
            f"| Metric | Score |\n"
            f"|--------|-------|\n"
            f"|  Coherence | {eval_result.get('coherence_score', 'N/A')} |\n"
            f"|  Constraints | {eval_result.get('constraint_satisfaction_score', 'N/A')} |\n"
            f"|  Feasibility | {eval_result.get('ingredient_feasibility_score', 'N/A')} |\n"
            f"|  Final Reward | **{score}** |\n"
        )

    except Exception as e:
        cost_display = f"Cost estimation unavailable: {str(e)}"
        eval_display = f"Evaluation unavailable: {str(e)}"
        parsed_ingredients = []

    return (retrieval_info, recipe_output, cost_display,
            eval_display, ", ".join(parsed_ingredients))


# ── Substitution ───────────────────────────────────────────────
def run_substitution(ingredient_query, top_k):
    if not ingredient_query.strip():
        return " Please enter an ingredient or query."

    query = ingredient_query.lower().strip()
    detected_restrictions = []
    restriction_aliases = {
        "vegan": "vegan", "vegetarian": "vegetarian", "jain": "jain",
        "gluten free": "gluten_free", "gluten-free": "gluten_free",
        "dairy free": "dairy_free", "dairy-free": "dairy_free",
    }
    for alias, canonical in restriction_aliases.items():
        if alias in query and canonical not in detected_restrictions:
            detected_restrictions.append(canonical)

    patterns = [
        r"(?:vegan|vegetarian|jain|gluten[-\s]?free|dairy[-\s]?free)(?:\s+and\s+(?:vegan|vegetarian|jain|gluten[-\s]?free|dairy[-\s]?free))*\s+(?:substitute|alternative)\s+(?:for|to)\s+(.+)",
        r"substitute\s+for\s+(.+)", r"replace\s+(.+)", r"instead\s+of\s+(.+)",
        r"alternative\s+to\s+(.+)", r"(.+)\s+substitute",
    ]
    ingredient = query
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            ingredient = match.group(1).strip()
            break

    subs = get_substitutes(ingredient, top_k=int(top_k), dietary_restrictions=detected_restrictions or None)
    if not subs:
        if detected_restrictions:
            restrictions_text = ", ".join(r.replace("_", "-") for r in detected_restrictions)
            return f"No valid substitutes found for **{ingredient}** under **{restrictions_text}** restrictions."
        return f"No substitutes found for **{ingredient}**"

    output = f"###  Top substitutes for **{ingredient}**:\n\n"
    if detected_restrictions:
        restrictions_text = ", ".join(r.replace("_", "-") for r in detected_restrictions)
        output += f"Applied dietary filters: **{restrictions_text}**\n\n"
    for i, s in enumerate(subs, 1):
        tag = "🇮🇳 Indian" if s.get("is_indian") else " Global"
        src = " Curated" if s.get("source") == "curated_kb" else " AI"
        output += (
            f"**{i}. {s['ingredient'].title()}** — {tag} · {src}\n"
            f"> Flavor match: `{s.get('flavor_similarity', 'N/A')}` | "
            f"Semantic: `{s.get('embedding_similarity', 'N/A')}` | "
            f"Score: `{s.get('final_score', 'N/A')}`\n\n"
        )
    return output


# ── Cost standalone ────────────────────────────────────────────
def run_cost_estimator(ingredients_text, servings):
    ings = [i.strip() for i in ingredients_text.strip().split("\n") if i.strip()]
    if not ings:
        return " Please enter at least one ingredient."
    result = estimate_cost_inr(ings, servings=int(servings))
    output = (
        f"## {result.get('budget_category', '')}\n\n"
        f"**Total Cost:** {result.get('total_cost', '₹?')} &nbsp;&nbsp; "
        f"**Per Serving:** {result.get('cost_per_serving', '₹?')}\n\n"
        f"---\n\n**Ingredient Breakdown:**\n\n"
    )
    for item in result.get("breakdown", []):
        output += f"- **{item['ingredient']}**: {item['cost']} *(unit: {item['unit_price']})*\n"
    if result.get("ingredients_not_found"):
        output += f"\n Prices not found for: `{', '.join(result['ingredients_not_found'])}`"
    return output


# ── Evaluator (RAGAS only, no duplicate score) ────────────────
def _time_text_to_minutes(time_text: str):
    if not time_text:
        return None
    text = str(time_text).lower()
    hours   = re.search(r"(\d+)\s*h(?:our|ours)?", text)
    minutes = re.search(r"(\d+)\s*m(?:in|ins|inute|inutes)?", text)
    total = 0
    if hours:   total += int(hours.group(1)) * 60
    if minutes: total += int(minutes.group(1))
    if total > 0: return total
    first_number = re.search(r"(\d+)", text)
    return int(first_number.group(1)) if first_number else None


def run_evaluator(recipe_text, available_ingredients_text, diet_value, appliance_value, time_limit_text):
    if not recipe_text or not recipe_text.strip():
        return "Generate a recipe first in the Kitchen tab, then evaluate here."

    parsed_ingredients, parsed_steps, estimated_time_text = parse_recipe_output(recipe_text)
    if not parsed_steps or not parsed_ingredients:
        return "Could not parse recipe. Please regenerate and try again."

    available_ingredients = [
        i.strip() for i in str(available_ingredients_text or "").split(",") if i.strip()
    ] or parsed_ingredients

    if not global_retrieved_docs or not global_query:
        return "No retrieved documents found. Generate a recipe first, then evaluate."

    try:
        faithfulness         = compute_faithfulness(recipe_text, global_retrieved_docs, available_ingredients)
        answer_relevance     = compute_answer_relevance(global_query, recipe_text, embedder)
        contextual_precision = compute_contextual_precision(recipe_text, global_retrieved_docs)
        contextual_recall    = compute_contextual_recall(recipe_text, available_ingredients)
        overall              = compute_overall_score(faithfulness, answer_relevance, contextual_precision, contextual_recall)
    except Exception as e:
        return f"Evaluation error: {str(e)}"

    # Save to CSV so Metrics tab updates
    try:
        save_results(pd.DataFrame([{
            "model":                 "hybrid_rag",
            "query":                 global_query,
            "answer":                recipe_text[:200],
            "faithfulness":          faithfulness,
            "answer_relevance":      answer_relevance,
            "contextual_precision":  contextual_precision,
            "contextual_recall":     contextual_recall,
            "overall_score":         overall,
        }]))
    except Exception:
        pass

    def status(score, target):
        return "✅" if score >= target else "❌"

    return (
        f"### RAGAS Evaluation — RAG Quality\n\n"
        f"| Metric | Score | Target | Status |\n"
        f"|--------|-------|--------|--------|\n"
        f"| Faithfulness | {faithfulness} | ≥0.70 | {status(faithfulness, 0.70)} |\n"
        f"| Answer Relevance | {answer_relevance} | ≥0.75 | {status(answer_relevance, 0.75)} |\n"
        f"| Contextual Precision | {contextual_precision} | ≥0.65 | {status(contextual_precision, 0.65)} |\n"
        f"| Contextual Recall | {contextual_recall} | ≥0.65 | {status(contextual_recall, 0.65)} |\n"
        f"| **Overall Score** | **{overall}** | weighted | — |\n\n"
        f"---\n\n"
        f"**Faithfulness**: Ingredients grounded in retrieved docs — detects hallucination.\n\n"
        f"**Answer Relevance**: Semantic match between your query and the recipe.\n\n"
        f"**Contextual Precision**: Fraction of retrieved recipes that were actually relevant.\n\n"
        f"**Contextual Recall**: Fraction of your available ingredients used.\n\n"
        f"**Overall**: F×0.35 + AR×0.30 + CP×0.20 + CR×0.15\n\n"
        f"*{len(parsed_ingredients)} ingredients · {len(parsed_steps)} steps · "
        f"Est. time: {estimated_time_text or 'not found'}*"
    )


# ── Metrics tab ────────────────────────────────────────────────
def render_metrics_tab():
    from config import EVAL_RESULTS_PATH, METRIC_WEIGHTS, OVERLAP_THRESHOLD
    if not os.path.exists(EVAL_RESULTS_PATH):
        return "No evaluation results yet. Run a query and evaluate it first."
    data = pd.read_csv(EVAL_RESULTS_PATH)
    if data.empty:
        return "No results in eval CSV yet."
    latest = data.tail(10)
    mean_f  = latest["faithfulness"].mean()
    mean_ar = latest["answer_relevance"].mean()
    mean_cp = latest["contextual_precision"].mean()
    mean_cr = latest["contextual_recall"].mean()
    mean_ov = latest["overall_score"].mean() if "overall_score" in latest.columns else None

    def bar(score, width=20):
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled) + f"  {score:.3f}"

    def status(score, target):
        return "✅" if score >= target else "❌"

    out = (
        f"### RAGAS Metric Scores (last {len(latest)} evaluations)\n\n"
        f"```\n"
        f"Faithfulness         {bar(mean_f)}\n"
        f"Answer Relevance     {bar(mean_ar)}\n"
        f"Contextual Precision {bar(mean_cp)}\n"
        f"Contextual Recall    {bar(mean_cr)}\n"
        f"{'─'*40}\n"
    )
    if mean_ov is not None:
        out += f"Overall Score        {bar(mean_ov)}\n"
    out += "```\n\n"
    out += (
        f"### Target Status\n\n"
        f"| Metric | Score | Target | Status |\n"
        f"|--------|-------|--------|--------|\n"
        f"| Faithfulness | {mean_f:.3f} | ≥0.70 | {status(mean_f, 0.70)} |\n"
        f"| Answer Relevance | {mean_ar:.3f} | ≥0.75 | {status(mean_ar, 0.75)} |\n"
        f"| Contextual Precision | {mean_cp:.3f} | ≥0.65 | {status(mean_cp, 0.65)} |\n"
        f"| Contextual Recall | {mean_cr:.3f} | ≥0.65 | {status(mean_cr, 0.65)} |\n"
    )
    if mean_ov is not None:
        out += f"| **Overall** | **{mean_ov:.3f}** | weighted | — |\n"
    out += f"\n### Active Weight Configuration\n\n| Metric | Weight |\n|--------|--------|\n"
    for metric, w in METRIC_WEIGHTS.items():
        out += f"| {metric} | {w} |\n"
    out += f"\nOverlap Threshold: `{OVERLAP_THRESHOLD}`\n"
    out += f"\n### Last {len(latest)} Query Results\n\n"
    out += "| Query | F | AR | CP | CR | Overall |\n|-------|---|----|----|----|---------|\n"
    for _, row in latest.iterrows():
        q = str(row["query"])[:35] + "..."
        ov = f"{row['overall_score']:.3f}" if "overall_score" in row else "N/A"
        out += f"| {q} | {row['faithfulness']:.2f} | {row['answer_relevance']:.2f} | {row['contextual_precision']:.2f} | {row['contextual_recall']:.2f} | {ov} |\n"
    return out


# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(title="RecipeRAG") as demo:

    gr.Markdown("""
    #  RecipeRAG
    ### Context-Aware Recipe Generation with RAG + Personalization
    
    Generate **personalized recipes** using advanced Retrieval-Augmented Generation.
    Provide a text query **or** upload a dish photo. Add constraints to customize further.
    
    ---
    """)

    with gr.Tabs():

        # ════════════════════════════════════════════════════
        # TAB 1 — Kitchen (Text-Based Generation)
        # ════════════════════════════════════════════════════
        with gr.Tab("Kitchen"):
            with gr.Row():
                gr.Markdown("""
                ####  Describe Your Recipe Idea
                Tell us what you want to cook. Add ingredients and constraints for personalization.
                """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  Your Recipe Request")
                    text_query = gr.Textbox(label=" Describe your recipe", placeholder="e.g. quick vegetarian dal with spinach under 30 minutes", lines=3, scale=1)
                    gr.Markdown("###  Customize Your Recipe")
                    available_ingredients = gr.Textbox(label=" Available Ingredients", placeholder="paneer, spinach, onion, tomato, garam masala, ghee", scale=1)
                    with gr.Row():
                        diet     = gr.Dropdown(label=" Dietary Restriction", choices=["None","Vegetarian","Vegan","Gluten-free","Jain","Halal"], value="None", scale=1)
                        appliance = gr.Dropdown(label=" Cooking Appliance", choices=["None","Stovetop only","Microwave only","Oven","Air fryer","Pressure cooker"], value="None", scale=1)
                    with gr.Row():
                        time_limit = gr.Textbox(label="Time Limit", placeholder="e.g. 30 minutes", scale=1)
                        budget     = gr.Textbox(label=" Budget", placeholder="e.g. under ₹200", scale=1)
                    submit_btn_kitchen = gr.Button(" Generate My Recipe!", variant="primary", scale=1, size="lg")
                with gr.Column(scale=1):
                    gr.Markdown("###  Results & Analysis")
                    gr.Markdown("####  Retrieved References")
                    retrieval_output_kitchen = gr.Markdown(value="*Your retrieval info will appear here...*")
                    gr.Markdown("####  Your Personalized Recipe")
                    recipe_output_kitchen = gr.Textbox(label="Generated Recipe", lines=16, interactive=False, placeholder="Your recipe will appear here after generation...")
                    parsed_ingredients_state_kitchen = gr.Textbox(label=" Parsed Ingredients", interactive=False, visible=False)
            gr.Markdown("---")
            gr.Markdown("###  Cost & Quality Analysis")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("####  Cost Breakdown")
                    cost_output_kitchen = gr.Markdown(value="*Cost estimate will appear after recipe generation...*")
                with gr.Column():
                    gr.Markdown("####  Quality Score")
                    eval_output_kitchen = gr.Markdown(value="*Quality evaluation will appear after recipe generation...*")
            submit_btn_kitchen.click(
                fn=run_pipeline,
                inputs=[text_query, gr.State(None), available_ingredients, diet, appliance, time_limit, budget],
                outputs=[retrieval_output_kitchen, recipe_output_kitchen, cost_output_kitchen, eval_output_kitchen, parsed_ingredients_state_kitchen]
            )

        # ════════════════════════════════════════════════════
        # TAB 2 — Image
        # ════════════════════════════════════════════════════
        with gr.Tab("Image"):
            with gr.Row():
                gr.Markdown("####  Upload a Dish Photo\nUpload an image of a dish you want to recreate.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  Your Recipe Request")
                    dish_image = gr.Image(label="Upload a dish photo", type="numpy", scale=1)
                    text_query_image = gr.Textbox(label=" Additional notes (optional)", placeholder="e.g. make it vegan, under 30 minutes", lines=2, scale=1)
                    gr.Markdown("###  Customize Your Recipe")
                    available_ingredients_image = gr.Textbox(label=" Available Ingredients", placeholder="paneer, spinach, onion, tomato, garam masala, ghee", scale=1)
                    with gr.Row():
                        diet_image     = gr.Dropdown(label=" Dietary Restriction", choices=["None","Vegetarian","Vegan","Gluten-free","Jain","Halal"], value="None", scale=1)
                        appliance_image = gr.Dropdown(label=" Cooking Appliance", choices=["None","Stovetop only","Microwave only","Oven","Air fryer","Pressure cooker"], value="None", scale=1)
                    with gr.Row():
                        time_limit_image = gr.Textbox(label="Time Limit", placeholder="e.g. 30 minutes", scale=1)
                        budget_image     = gr.Textbox(label=" Budget", placeholder="e.g. under ₹200", scale=1)
                    submit_btn_image = gr.Button(" Generate My Recipe!", variant="primary", scale=1, size="lg")
                with gr.Column(scale=1):
                    gr.Markdown("###  Results & Analysis")
                    retrieval_output_image = gr.Markdown(value="*Your retrieval info will appear here...*")
                    recipe_output_image    = gr.Textbox(label="Generated Recipe", lines=16, interactive=False, placeholder="Your recipe will appear here after generation...")
                    parsed_ingredients_state_image = gr.Textbox(label=" Parsed Ingredients", interactive=False, visible=False)
            gr.Markdown("---")
            gr.Markdown("###  Cost & Quality Analysis")
            with gr.Row():
                with gr.Column():
                    cost_output_image = gr.Markdown(value="*Cost estimate will appear after recipe generation...*")
                with gr.Column():
                    eval_output_image = gr.Markdown(value="*Quality evaluation will appear after recipe generation...*")
            submit_btn_image.click(
                fn=run_pipeline,
                inputs=[text_query_image, dish_image, available_ingredients_image, diet_image, appliance_image, time_limit_image, budget_image],
                outputs=[retrieval_output_image, recipe_output_image, cost_output_image, eval_output_image, parsed_ingredients_state_image]
            )

        # ════════════════════════════════════════════════════
        # TAB 3 — Substitution
        # ════════════════════════════════════════════════════
        with gr.Tab("Substitution"):
            gr.Markdown("""
            ##  Find Perfect Ingredient Substitutes
            Ask in **natural language**: "vegan alternative to butter", "substitute for paneer"
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  Your Substitution Request")
                    sub_query    = gr.Textbox(label="Describe what you need", placeholder="vegan alternative to butter\nsubstitute for paneer\nreplace ghee", lines=4)
                    top_k_slider = gr.Slider(minimum=3, maximum=10, value=5, step=1, label=" Number of alternatives to show")
                    sub_btn      = gr.Button(" Find Best Substitutes", variant="primary", size="lg")
                with gr.Column(scale=1):
                    gr.Markdown("### Top Alternatives")
                    sub_output = gr.Markdown(value="*Substitutes will appear here...*")
            sub_btn.click(fn=run_substitution, inputs=[sub_query, top_k_slider], outputs=[sub_output])

        # ════════════════════════════════════════════════════
        # TAB 4 — Cost Estimator
        # ════════════════════════════════════════════════════
        with gr.Tab("Cost Estimator"):
            gr.Markdown("##  Recipe Cost Estimator\nGet **accurate Indian market prices** for your recipe in ₹.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  Your Ingredients")
                    cost_ing_input = gr.Textbox(label="List ingredients", placeholder="toor dal\n2 onion\n3 tomato\nghee\ncumin", lines=10)
                    cost_servings  = gr.Slider(minimum=1, maximum=10, value=2, step=1, label=" Number of servings")
                    cost_btn       = gr.Button(" Calculate Cost", variant="primary", size="lg")
                with gr.Column(scale=1):
                    gr.Markdown("###  Price Breakdown")
                    cost_result_output = gr.Markdown(value="*Cost breakdown will appear here...*")
            cost_btn.click(fn=run_cost_estimator, inputs=[cost_ing_input, cost_servings], outputs=[cost_result_output])

        # ════════════════════════════════════════════════════
        # TAB 5 — Evaluator (RAGAS only)
        # ════════════════════════════════════════════════════
        with gr.Tab("Evaluator"):
            gr.Markdown("""
            ##  Evaluate Generated Recipe
            Generates a recipe first in the Kitchen tab, then paste it here for RAGAS evaluation.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("###  Recipe Input")
                    recipe_text_input          = gr.Textbox(label="Generated Recipe", placeholder="Paste your generated recipe here...", lines=10)
                    available_ingredients_eval = gr.Textbox(label="Available Ingredients", placeholder="e.g. paneer, spinach, onion, tomato")
                    with gr.Row():
                        diet_eval     = gr.Dropdown(label="Dietary Restriction", choices=["None","Vegetarian","Vegan","Gluten-free","Jain","Halal"], value="None")
                        appliance_eval = gr.Dropdown(label="Cooking Appliance", choices=["None","Stovetop only","Microwave only","Oven","Air fryer","Pressure cooker"], value="None")
                    time_limit_eval = gr.Textbox(label="Time Limit", placeholder="e.g. 30 minutes")
                    eval_btn        = gr.Button("Evaluate Recipe", variant="primary", size="lg")
                with gr.Column(scale=1):
                    gr.Markdown("###  RAGAS Results")
                    eval_result_output = gr.Markdown(value="*Paste a recipe and click evaluate...*")
            eval_btn.click(fn=run_evaluator, inputs=[recipe_text_input, available_ingredients_eval, diet_eval, appliance_eval, time_limit_eval], outputs=[eval_result_output])

        # ════════════════════════════════════════════════════
        # TAB 6 — Statistics
        # ════════════════════════════════════════════════════
        with gr.Tab("Statistics"):
            gr.Markdown("##  Project Statistics")
            gr.Markdown(f"""
### Dataset & Models
- **Total Recipes in Database**: {collection.count()}
- **Embedding Model**: {EMBEDDING_MODEL}
- **LLM**: Llama 3.1-8b (via Groq)
- **Image Model**: CLIP (for dish recognition)

### Retrieval Configuration
- **Method**: Hybrid BM25 + Dense Vector Search + RRF + Cross-encoder Reranker
- **BM25 Candidates**: {BM25_CANDIDATES}
- **Dense Candidates**: {DENSE_CANDIDATES}
- **RRF K-Value**: {RRF_K}
- **Top-K Results**: {TOP_K}

### Evaluation Metrics
- **RAGAS**: Faithfulness, Answer Relevance, Contextual Precision, Contextual Recall
- **Weighted Overall Score**: F×0.35 + AR×0.30 + CP×0.20 + CR×0.15
- **Recipe Quality** (Kitchen tab): Coherence, Constraint Satisfaction, Feasibility
""")

        # ════════════════════════════════════════════════════
        # TAB 7 — Metrics
        # ════════════════════════════════════════════════════
        with gr.Tab("Metrics"):
            gr.Markdown("""
            ## RAGAS Evaluation Metrics
            Live view of your evaluation scores, targets, and weight configuration.
            Click Refresh after running evaluations to update.
            """)
            refresh_btn    = gr.Button("Refresh Metrics", variant="primary")
            metrics_output = gr.Markdown(value="*Click Refresh to load metrics...*")
            refresh_btn.click(fn=render_metrics_tab, inputs=[], outputs=[metrics_output])

    with gr.Row():
        gr.Markdown("""
---
<div style="text-align: center; padding: 20px; color: #7B3F00;">
    <p><strong> RecipeRAG</strong></p>
    <p style="font-size: 13px; margin-top: 8px;">
        Combining semantic search, BM25 retrieval, and LLM generation for personalized recipes.
    </p>
    <p style="font-size: 11px; color: #A06030; margin-top: 12px;">
        Models: MiniLM · Llama 3.1-8b · CLIP · Cross-encoder Reranker
    </p>
</div>
""")

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft(), css=CUSTOM_CSS)