import os
import re
import requests

DIETARY_RULES = {
    "vegan": [
        "milk", "curd", "yogurt", "paneer", "ghee", "butter", "cream",
        "khoya", "honey", "egg", "meat", "chicken", "fish", "prawn",
        "mutton", "condensed milk", "buttermilk", "cheese"
    ],
    "vegetarian": [
        "meat", "chicken", "fish", "prawn", "mutton", "beef", "pork",
        "egg", "seafood"
    ],
    "jain": [
        "onion", "garlic", "potato", "carrot", "radish", "beetroot",
        "meat", "chicken", "fish", "egg"
    ],
    "gluten_free": [
        "wheat", "atta", "maida", "sooji", "bread", "barley", "rye",
        "semolina"
    ],
    "dairy_free": [
        "milk", "curd", "yogurt", "paneer", "ghee", "butter", "cream",
        "khoya", "condensed milk", "buttermilk", "cheese"
    ],
    "halal": []
}

DIET_ALIASES = {
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "jain": "jain",
    "gluten free": "gluten_free",
    "gluten-free": "gluten_free",
    "gluten_free": "gluten_free",
    "dairy free": "dairy_free",
    "dairy-free": "dairy_free",
    "dairy_free": "dairy_free",
    "halal": "halal",
}


def _normalize_dietary_restrictions(diet) -> list[str]:
    if not diet:
        return []

    values = [diet] if isinstance(diet, str) else diet
    normalized = []

    for item in values:
        key = re.sub(r"\s+", " ", str(item).strip().lower())
        canonical = DIET_ALIASES.get(key)
        if canonical and canonical not in normalized:
            normalized.append(canonical)

    return normalized


def _get_banned_ingredients(dietary_restrictions: list[str]) -> list[str]:
    banned = []
    for restriction in dietary_restrictions:
        banned.extend(DIETARY_RULES.get(restriction, []))

    seen = set()
    deduped = []
    for term in banned:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def _extract_diet_violations(recipe_text: str, dietary_restrictions: list[str]) -> list[str]:
    if not recipe_text or not dietary_restrictions:
        return []

    recipe_lower = recipe_text.lower()
    banned = _get_banned_ingredients(dietary_restrictions)
    return [term for term in banned if term in recipe_lower]


def build_prompt(query: str, retrieved_recipes: list[dict], constraints: dict = None) -> str:
    if constraints is None:
        constraints = {}

    context_blocks = []
    for i, recipe in enumerate(retrieved_recipes, 1):
        ingredients = (
            ", ".join(recipe["ingredients"])
            if isinstance(recipe["ingredients"], list)
            else recipe["ingredients"]
        )
        context_blocks.append(
            f"Reference Recipe {i}: {recipe['title']}\n"
            f"Ingredients: {ingredients}\n"
            f"Instructions: {recipe['full_text'][:500]}"
        )
    context_str = "\n\n".join(context_blocks)

    dietary_restrictions = _normalize_dietary_restrictions(constraints.get("diet"))
    banned_ingredients = _get_banned_ingredients(dietary_restrictions)

    constraint_lines = []
    if constraints.get("ingredients"):
        ing_list = ", ".join(constraints["ingredients"])
        constraint_lines.append(
            f"- Available ingredients: {ing_list}. Use ONLY these ingredients unless absolutely essential."
        )
    if dietary_restrictions:
        pretty_diets = ", ".join(d.replace("_", "-") for d in dietary_restrictions)
        constraint_lines.append(
            f"- Dietary restrictions (mandatory): {pretty_diets}."
        )
        if banned_ingredients:
            constraint_lines.append(
                f"- Forbidden ingredients for these restrictions: {', '.join(banned_ingredients)}."
            )
    if constraints.get("appliance"):
        constraint_lines.append(
            f"- Appliance constraint: {constraints['appliance']}. Only suggest cooking methods compatible with this."
        )
    if constraints.get("time"):
        constraint_lines.append(
            f"- Time limit: {constraints['time']}. The total cooking + prep time must fit within this."
        )
    if constraints.get("budget"):
        constraint_lines.append(
            f"- Budget: {constraints['budget']}. Keep the recipe affordable within this limit."
        )

    constraint_str = (
        "\n".join(constraint_lines)
        if constraint_lines
        else "- No specific constraints. Generate the best recipe possible."
    )

    prompt = f"""You are an expert chef and recipe writer. Generate a clear, detailed, practical recipe.

You have {len(retrieved_recipes)} reference recipes retrieved from a recipe database. Use them as grounding.
Do not hallucinate ingredients or techniques not supported by these references or the user's available ingredients.

--- REFERENCE RECIPES ---
{context_str}

--- USER REQUEST ---
Query: {query}

--- CONSTRAINTS (must follow all) ---
{constraint_str}

--- YOUR TASK ---
Generate a complete recipe that:
1. Directly addresses the user's query
2. Respects every constraint listed above
3. Never includes forbidden ingredients from dietary restrictions
4. Is grounded in the reference recipes
5. Is written in clear, simple steps for a home cook

Format your response exactly like this:

Recipe Name: <name>

Ingredients:
- <ingredient 1 with quantity>
- <ingredient 2 with quantity>

Instructions:
1. <step 1>
2. <step 2>

Estimated Time: <prep + cook time>
Serves: <number of servings>

Now generate the recipe:"""

    return prompt


def _call_groq(messages: list[dict], groq_key: str) -> str:
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": 700,
        "temperature": 0.35
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def generate_recipe(
    query: str,
    retrieved_recipes: list[dict],
    hf_token: str = None,
    constraints: dict = None
) -> str:
    if not retrieved_recipes:
        return "No relevant recipes found. Please try a different query."

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not set in .env file.")

    if constraints is None:
        constraints = {}

    dietary_restrictions = _normalize_dietary_restrictions(constraints.get("diet"))
    prompt = build_prompt(query, retrieved_recipes, constraints)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict recipe assistant. "
                    "Never include ingredients that violate user dietary restrictions."
                )
            },
            {"role": "user", "content": prompt}
        ]
        generated = _call_groq(messages, groq_key)

        violations = _extract_diet_violations(generated, dietary_restrictions)
        if violations:
            violation_text = ", ".join(violations[:10])
            retry_messages = messages + [
                {"role": "assistant", "content": generated},
                {
                    "role": "user",
                    "content": (
                        f"Your previous recipe included forbidden ingredients: {violation_text}. "
                        "Regenerate the full recipe and strictly avoid all forbidden ingredients."
                    )
                }
            ]
            regenerated = _call_groq(retry_messages, groq_key)
            if regenerated:
                generated = regenerated

        return generated if generated else "Error: Empty response from model."

    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"Error: API returned {e.response.status_code}. {e.response.text[:200]}"
    except Exception as e:
        return f"Error generating recipe: {str(e)}"
