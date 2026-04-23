# app/gradio_app.py — Integrated with Person 3 + Themed UI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image
import re

from config import DATAFRAME_PATH
from retrieval.embedder import load_embedder
from retrieval.vector_store import load_vector_store
from retrieval.bm25_retriever import load_bm25_index
from retrieval.hybrid_retriever import hybrid_retrieve
from generation.llm import generate_recipe
from app.clip_classifier import load_clip_model, classify_dish
from substitution.substitutor import (
    get_substitutes,
    estimate_cost_inr,
    evaluate_recipe,
    normalize_dietary_restrictions,
)

import pandas as pd

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
/* 🎨 BEAUTIFUL & MODERN UI STYLING FOR RECIPERAG                        */
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

/* ── TABS STYLING ── */
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

/* ── BLOCKS & CONTAINERS ── */
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

/* ── LABELS ── */
label, .gr-form label, .label-wrap label {
    color: #7B3F00 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    text-transform: none !important;
}

label span {
    color: #7B3F00 !important;
    font-weight: 700 !important;
}

/* ── TEXTBOX & TEXTAREA ── */
textarea, input[type=text], input[type=number], input[type=email], input[type=password] {
    background-color: #FFFDF7 !important;
    border: 2px solid #D4A96A !important;
    border-radius: 12px !important;
    color: #2C1810 !important;
    font-size: 14px !important;
    padding: 12px 14px !important;
    transition: all 0.3s ease !important;
    font-family: 'Segoe UI', sans-serif !important;
}

textarea:focus, input[type=text]:focus, input[type=number]:focus {
    border-color: #C8601A !important;
    box-shadow: 0 0 0 3px rgba(200, 96, 26, 0.1) !important;
    outline: none !important;
}

textarea::placeholder, input[type=text]::placeholder {
    color: #A06030 !important;
}

/* ── DROPDOWNS & SELECT ── */
select, .gr-dropdown {
    background-color: #FFFDF7 !important;
    border: 2px solid #D4A96A !important;
    border-radius: 12px !important;
    color: #2C1810 !important;
    font-weight: 500 !important;
    padding: 10px 12px !important;
    transition: all 0.3s ease !important;
}

select:hover, .gr-dropdown:hover {
    border-color: #C8601A !important;
}

select:focus, .gr-dropdown:focus {
    border-color: #C8601A !important;
    box-shadow: 0 0 0 3px rgba(200, 96, 26, 0.1) !important;
}

/* ── PRIMARY BUTTON ── */
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
    text-transform: none !important;
}

.gr-button-primary:hover, button.primary:hover {
    background: linear-gradient(135deg, #A0420D 0%, #7B3F00 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(200, 96, 26, 0.4) !important;
}

.gr-button-primary:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(200, 96, 26, 0.3) !important;
}

/* ── SECONDARY BUTTON ── */
.gr-button-secondary, button.secondary {
    background-color: #FFF0D6 !important;
    color: #7B3F00 !important;
    border: 2px solid #C8601A !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

.gr-button-secondary:hover {
    background-color: #FFE8C0 !important;
    border-color: #A0420D !important;
    transform: translateY(-2px) !important;
}

/* ── SLIDERS ── */
input[type=range] {
    accent-color: #C8601A !important;
    height: 6px !important;
}

/* ── CHECKBOXES & RADIOS ── */
input[type=checkbox], input[type=radio] {
    accent-color: #C8601A !important;
    width: 20px !important;
    height: 20px !important;
    cursor: pointer !important;
}

/* ── CHECKBOX GROUPS ── */
.gr-checkbox-group {
    gap: 12px !important;
}

.gr-checkbox-group .gr-checkbox-wrapper {
    background: linear-gradient(135deg, #FFFDF7 0%, #FFF8F0 100%) !important;
    border: 1px solid #E8C99A !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    transition: all 0.3s ease !important;
}

.gr-checkbox-group .gr-checkbox-wrapper:hover {
    border-color: #C8601A !important;
    background-color: #FFF5E6 !important;
}

/* ── IMAGE UPLOAD ── */
.gr-image, .image-container {
    border: 3px dashed #D4A96A !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #FFFDF7, #FFF5E6) !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

.gr-image:hover {
    border-color: #C8601A !important;
    background: linear-gradient(135deg, #FFF5E6, #FFFDF7) !important;
}

/* ── OUTPUT MARKDOWN ── */
.gr-markdown {
    background: transparent !important;
}

.gr-markdown table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 12px 0 !important;
}

.gr-markdown table thead {
    background-color: #FFF0D6 !important;
}

.gr-markdown table th {
    color: #7B3F00 !important;
    font-weight: 700 !important;
    padding: 12px !important;
    border: 1px solid #E8C99A !important;
}

.gr-markdown table td {
    padding: 12px !important;
    border: 1px solid #E8C99A !important;
    color: #2C1810 !important;
}

.gr-markdown table tbody tr:nth-child(even) {
    background-color: #FFFDF7 !important;
}

.gr-markdown table tbody tr:hover {
    background-color: #FFF5E6 !important;
}

/* ── PROGRESS & LOADING ── */
.gr-progress-container {
    background-color: #E8C99A !important;
    border-radius: 12px !important;
}

/* ── SPACING ── */
.gr-row {
    gap: 16px !important;
}

.gr-column {
    gap: 12px !important;
}

/* ── SCROLLBAR STYLING ── */
::-webkit-scrollbar {
    width: 10px !important;
    height: 10px !important;
}

::-webkit-scrollbar-track {
    background: #FFF5E6 !important;
}

::-webkit-scrollbar-thumb {
    background: #D4A96A !important;
    border-radius: 5px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: #C8601A !important;
}

/* ── RESPONSIVE ADJUSTMENTS ── */
@media (max-width: 768px) {
    .gr-markdown h1 {
        font-size: 1.8em !important;
    }
    
    .gr-markdown h2 {
        font-size: 1.3em !important;
    }
    
    .tab-nav button {
        font-size: 14px !important;
        padding: 10px 16px !important;
    }
    
    .gr-button-primary, button.primary {
        font-size: 15px !important;
        padding: 12px 24px !important;
    }
}

/* ── ANIMATION EFFECTS ── */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.block {
    animation: fadeIn 0.3s ease-in-out !important;
}
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
        query_display = f"📷 Detected dish: **{dish_name}**"
        if text_query.strip():
            query = f"{dish_name} {text_query.strip()}"
            query_display += f" + your notes: *{text_query.strip()}*"
    elif text_query.strip():
        query = text_query.strip()
        query_display = f"🔍 Query: **{query}**"
    else:
        return "⚠️ Please enter a query or upload a dish photo.", "", "", "", ""

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

    try:
        retrieved = hybrid_retrieve(
            query=query, df=df, bm25=bm25,
            collection=collection, embedder=embedder
        )
    except Exception as e:
        return query_display, f"❌ Retrieval error: {str(e)}", "", "", ""

    if not retrieved:
        return query_display, "❌ No relevant recipes found.", "", "", ""

    retrieved_titles = "\n".join([f"🍽️ {r['title']}" for r in retrieved])
    retrieval_info = f"{query_display}\n\n📚 **Retrieved References:**\n{retrieved_titles}"

    try:
        recipe_output = generate_recipe(
            query=query, retrieved_recipes=retrieved,
            hf_token=hf_token, constraints=constraints
        )
    except Exception as e:
        return retrieval_info, f"❌ Generation error: {str(e)}", "", "", ""

    # Person 3 — Cost + Evaluation
    try:
        avail_list = [i.strip() for i in available_ingredients.split(",") if i.strip()]
        diet_list = normalize_dietary_restrictions(diet) if diet and diet != "None" else []
        appliance_list = [appliance.lower()] if appliance and appliance != "None" else []

        parsed_ingredients, parsed_steps, _ = parse_recipe_output(recipe_output)

        cost_result = estimate_cost_inr(parsed_ingredients, servings=2)
        cost_display = (
            f"## {cost_result.get('budget_category', '💰 Cost Estimate')}\n\n"
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
            verdict = f"✅ Excellent Recipe!"
        elif score >= 0.55:
            verdict = f"⚠️ Good Recipe"
        elif score >= 0.35:
            verdict = f"🟠 Fair Recipe"
        else:
            verdict = f"❌ Needs Improvement"

        eval_display = (
            f"## {verdict}  Score: {score}\n\n"
            f"| Metric | Score |\n"
            f"|--------|-------|\n"
            f"| 🧠 Coherence | {eval_result.get('coherence_score', 'N/A')} |\n"
            f"| ✅ Constraints | {eval_result.get('constraint_satisfaction_score', 'N/A')} |\n"
            f"| 🥘 Feasibility | {eval_result.get('ingredient_feasibility_score', 'N/A')} |\n"
            f"| 🏆 Final Reward | **{score}** |\n"
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
        return "⚠️ Please enter an ingredient or query."

    query = ingredient_query.lower().strip()
    detected_restrictions = []
    restriction_aliases = {
        "vegan": "vegan",
        "vegetarian": "vegetarian",
        "jain": "jain",
        "gluten free": "gluten_free",
        "gluten-free": "gluten_free",
        "dairy free": "dairy_free",
        "dairy-free": "dairy_free",
    }
    for alias, canonical in restriction_aliases.items():
        if alias in query and canonical not in detected_restrictions:
            detected_restrictions.append(canonical)

    patterns = [
        r"(?:vegan|vegetarian|jain|gluten[-\s]?free|dairy[-\s]?free)(?:\s+and\s+(?:vegan|vegetarian|jain|gluten[-\s]?free|dairy[-\s]?free))*\s+(?:substitute|alternative)\s+(?:for|to)\s+(.+)",
        r"substitute\s+for\s+(.+)",
        r"replace\s+(.+)",
        r"instead\s+of\s+(.+)",
        r"alternative\s+to\s+(.+)",
        r"(.+)\s+substitute",
    ]
    ingredient = query
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            ingredient = match.group(1).strip()
            break

    subs = get_substitutes(
        ingredient,
        top_k=int(top_k),
        dietary_restrictions=detected_restrictions or None
    )
    if not subs:
        if detected_restrictions:
            restrictions_text = ", ".join(r.replace("_", "-") for r in detected_restrictions)
            return (
                f"No valid substitutes found for **{ingredient}** "
                f"under **{restrictions_text}** restrictions."
            )
        return f"No substitutes found for **{ingredient}**"

    output = f"### 🔄 Top substitutes for **{ingredient}**:\n\n"
    if detected_restrictions:
        restrictions_text = ", ".join(r.replace("_", "-") for r in detected_restrictions)
        output += f"Applied dietary filters: **{restrictions_text}**\n\n"

    for i, s in enumerate(subs, 1):
        tag = "🇮🇳 Indian" if s.get("is_indian") else "🌍 Global"
        src = "📖 Curated" if s.get("source") == "curated_kb" else "🤖 AI"
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
        return "⚠️ Please enter at least one ingredient."
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
        output += f"\n⚠️ Prices not found for: `{', '.join(result['ingredients_not_found'])}`"
    return output


# ── Evaluator standalone ───────────────────────────────────────
def _time_text_to_minutes(time_text: str):
    if not time_text:
        return None
    text = str(time_text).lower()

    hours = re.search(r"(\d+)\s*h(?:our|ours)?", text)
    minutes = re.search(r"(\d+)\s*m(?:in|ins|inute|inutes)?", text)

    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))

    if total > 0:
        return total

    first_number = re.search(r"(\d+)", text)
    return int(first_number.group(1)) if first_number else None


def run_evaluator(recipe_text, available_ingredients_text, diet_value, appliance_value, time_limit_text):
    if not recipe_text or not recipe_text.strip():
        return "Generate a recipe first in the Generate Recipe tab, then evaluate it here."

    parsed_ingredients, parsed_steps, estimated_time_text = parse_recipe_output(recipe_text)
    if not parsed_steps or not parsed_ingredients:
        return "Could not parse recipe steps/ingredients. Please regenerate the recipe and try again."

    available_ingredients = [
        i.strip() for i in str(available_ingredients_text or "").split(",") if i.strip()
    ]
    if not available_ingredients:
        available_ingredients = parsed_ingredients

    diet_list = normalize_dietary_restrictions(diet_value) if diet_value and diet_value != "None" else []

    appliance_map = {
        "Stovetop only": ["stovetop"],
        "Microwave only": ["microwave"],
        "Oven": ["oven"],
        "Air fryer": ["air fryer"],
        "Pressure cooker": ["pressure cooker"],
    }
    appliance_list = appliance_map.get(appliance_value, []) if appliance_value and appliance_value != "None" else []

    max_time_minutes = _time_text_to_minutes(time_limit_text)
    estimated_time_minutes = _time_text_to_minutes(estimated_time_text)

    result = evaluate_recipe(
        recipe_steps=parsed_steps,
        recipe_ingredients=parsed_ingredients,
        available_ingredients=available_ingredients,
        dietary_restrictions=diet_list if diet_list else None,
        available_appliances=appliance_list if appliance_list else None,
        max_time_minutes=max_time_minutes,
        estimated_time_minutes=estimated_time_minutes
    )

    score = result.get("final_reward", 0)
    if score >= 0.75:
        verdict = "Excellent Recipe"
    elif score >= 0.55:
        verdict = "Good Recipe"
    elif score >= 0.35:
        verdict = "Fair Recipe"
    else:
        verdict = "Needs Improvement"

    return (
        f"## {verdict}\n\n"
        f"**Final Reward Score: {score}**\n\n"
        f"---\n\n"
        f"**Auto-used context:**\n"
        f"- Parsed ingredients: {len(parsed_ingredients)}\n"
        f"- Parsed steps: {len(parsed_steps)}\n"
        f"- Time limit: {time_limit_text if time_limit_text else 'Not set'}\n"
        f"- Estimated time from recipe: {estimated_time_text if estimated_time_text else 'Not found'}\n\n"
        f"| Metric | Score |\n"
        f"|--------|-------|\n"
        f"| Coherence | {result.get('coherence_score', 'N/A')} |\n"
        f"| Constraint Satisfaction | {result.get('constraint_satisfaction_score', 'N/A')} |\n"
        f"| Ingredient Feasibility | {result.get('ingredient_feasibility_score', 'N/A')} |\n"
        f"| **Final Reward** | **{score}** |\n"
    )
# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(title="RecipeRAG") as demo:

    gr.Markdown("""
    # 🍳 RecipeRAG
    ### Context-Aware Recipe Generation with RAG + Personalization
    
    Generate **personalized recipes** using advanced Retrieval-Augmented Generation.
    Provide a text query **or** upload a dish photo. Add constraints to customize further.
    
    ---
    """)

    with gr.Tabs():

        # ════════════════════════════════════════════════════
        # TAB 1 — Generate Recipe
        # ════════════════════════════════════════════════════
        with gr.Tab("🍴 Generate Recipe"):
            with gr.Row():
                gr.Markdown("""
                #### 🎯 What do you want to cook today?
                
                Describe your recipe idea or upload a dish photo, then customize with your constraints.
                """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Your Recipe Request")
                    
                    text_query = gr.Textbox(
                        label="🔍 Describe your recipe",
                        placeholder="e.g. quick vegetarian dal with spinach under 30 minutes",
                        lines=3,
                        scale=1
                    )
                    
                    dish_image = gr.Image(
                        label="📷 Or upload a dish photo",
                        type="numpy",
                        scale=1
                    )
                    
                    gr.Markdown("### ⚙️ Customize Your Recipe")
                    
                    available_ingredients = gr.Textbox(
                        label="🧺 Available Ingredients",
                        placeholder="paneer, spinach, onion, tomato, garam masala, ghee",
                        scale=1
                    )
                    
                    with gr.Row():
                        diet = gr.Dropdown(
                            label="🥗 Dietary Restriction",
                            choices=["None", "Vegetarian", "Vegan",
                                     "Gluten-free", "Jain", "Halal"],
                            value="None",
                            scale=1
                        )
                        appliance = gr.Dropdown(
                            label="🍳 Cooking Appliance",
                            choices=["None", "Stovetop only", "Microwave only",
                                     "Oven", "Air fryer", "Pressure cooker"],
                            value="None",
                            scale=1
                        )
                    
                    with gr.Row():
                        time_limit = gr.Textbox(
                            label="⏱️ Time Limit",
                            placeholder="e.g. 30 minutes",
                            scale=1
                        )
                        budget = gr.Textbox(
                            label="💰 Budget",
                            placeholder="e.g. under ₹200",
                            scale=1
                        )
                    
                    submit_btn = gr.Button(
                        "🚀 Generate My Recipe!",
                        variant="primary",
                        scale=1,
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Results & Analysis")
                    
                    gr.Markdown("#### 📚 Retrieved References")
                    retrieval_output = gr.Markdown(
                        value="*Your retrieval info will appear here...*"
                    )
                    
                    gr.Markdown("#### 🍴 Your Personalized Recipe")
                    recipe_output = gr.Textbox(
                        label="Generated Recipe",
                        lines=16,
                        interactive=False,
                        placeholder="Your recipe will appear here after generation..."
                    )
                    
                    parsed_ingredients_state = gr.Textbox(
                        label="📝 Parsed Ingredients",
                        interactive=False,
                        visible=False
                    )
            
            gr.Markdown("---")
            gr.Markdown("### 💰 Cost & Quality Analysis")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 💸 Cost Breakdown")
                    cost_output = gr.Markdown(
                        value="*Cost estimate will appear after recipe generation...*"
                    )
                with gr.Column():
                    gr.Markdown("#### 📊 Quality Score")
                    eval_output = gr.Markdown(
                        value="*Quality evaluation will appear after recipe generation...*"
                    )
            
            submit_btn.click(
                fn=run_pipeline,
                inputs=[text_query, dish_image, available_ingredients,
                        diet, appliance, time_limit, budget],
                outputs=[retrieval_output, recipe_output,
                         cost_output, eval_output, parsed_ingredients_state]
            )

        # ════════════════════════════════════════════════════
        # TAB 2 — Ingredient Substitution
        # ════════════════════════════════════════════════════
        with gr.Tab("🔄 Ingredient Substitution"):
            gr.Markdown("""
            ## 🔄 Find Perfect Ingredient Substitutes
            
            Ask in **natural language**. The system understands requests like:
            - "vegan alternative to butter"
            - "substitute for paneer"
            - "gluten free alternative to atta"
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 💬 Your Substitution Request")
                    sub_query = gr.Textbox(
                        label="Describe what you need",
                        placeholder=(
                            "vegan alternative to butter\n"
                            "substitute for paneer\n"
                            "replace ghee\n"
                            "gluten free alternative to atta"
                        ),
                        lines=4
                    )
                    top_k_slider = gr.Slider(
                        minimum=3, maximum=10, value=5, step=1,
                        label="📊 Number of alternatives to show"
                    )
                    sub_btn = gr.Button(
                        "🔍 Find Best Substitutes",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 🌿 Top Alternatives")
                    sub_output = gr.Markdown(
                        value="*Substitutes will appear here...*"
                    )
            
            sub_btn.click(
                fn=run_substitution,
                inputs=[sub_query, top_k_slider],
                outputs=[sub_output]
            )

        # ════════════════════════════════════════════════════
        # TAB 3 — Cost Estimator
        # ════════════════════════════════════════════════════
        with gr.Tab("💸 Cost Estimator"):
            gr.Markdown("""
            ## 💸 Recipe Cost Estimator
            Get **accurate Indian market prices** for your recipe in ₹ with full breakdown.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧺 Your Ingredients")
                    cost_ing_input = gr.Textbox(
                        label="List ingredients",
                        placeholder=(
                            "toor dal\n"
                            "2 onion\n"
                            "3 tomato\n"
                            "ghee\n"
                            "cumin\n"
                            "1/2 tsp turmeric\n"
                            "paneer"
                        ),
                        lines=10
                    )
                    cost_servings = gr.Slider(
                        minimum=1, maximum=10, value=2, step=1,
                        label="👨‍👩‍👧 Number of servings"
                    )
                    cost_btn = gr.Button(
                        "💰 Calculate Cost",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Price Breakdown")
                    cost_result_output = gr.Markdown(
                        value="*Cost breakdown will appear here...*"
                    )
            
            cost_btn.click(
                fn=run_cost_estimator,
                inputs=[cost_ing_input, cost_servings],
                outputs=[cost_result_output]
            )

        # ════════════════════════════════════════════════════
        # TAB 4 — Recipe Evaluator
        # ════════════════════════════════════════════════════
        with gr.Tab("📊 Recipe Evaluator"):
            gr.Markdown("""
            ## 📊 Evaluate Generated Recipe
            This tab reuses the latest recipe and constraints from **Generate Recipe**.
            No re-entry needed.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔗 Auto-Linked Inputs")
                    gr.Markdown(
                        "- Uses **Generated Recipe** from Tab 1\n"
                        "- Uses **Available Ingredients** from Tab 1\n"
                        "- Uses **Dietary Restriction** from Tab 1\n"
                        "- Uses **Appliance** from Tab 1\n"
                        "- Uses **Time Limit** from Tab 1"
                    )

                    eval_btn = gr.Button(
                        "📊 Evaluate Generated Recipe",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 🏆 Results")
                    eval_result_output = gr.Markdown(
                        value="*Click evaluate after generating a recipe in Tab 1...*"
                    )

            eval_btn.click(
                fn=run_evaluator,
                inputs=[recipe_output, available_ingredients, diet, appliance, time_limit],
                outputs=[eval_result_output]
            )
    # FOOTER
    # ════════════════════════════════════════════════════
    with gr.Row():
        gr.Markdown("""
        ---
        
        <div style="text-align: center; padding: 20px; color: #7B3F00;">
            <p><strong>🍳 RecipeRAG</strong> — Powered by Retrieval-Augmented Generation</p>
            <p style="font-size: 13px; margin-top: 8px;">
                Combining semantic search, BM25 retrieval, and LLM generation for personalized recipes.
                <br/>
                Cost estimation & quality evaluation included.
            </p>
            <p style="font-size: 11px; color: #A06030; margin-top: 12px;">
                Made with ❤️ using Gradio • Models: MiniLM, Mistral-7B, CLIP
            </p>
        </div>
        """)

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft(), css=CUSTOM_CSS)




