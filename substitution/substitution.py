# substitution/substitution.py
# Person 3's implementation wired into the main repo

import os
import importlib.util
import re

# Load person3 substitution model
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sub_path = os.path.join(_base, "person3", "substitution", "substitution_model.py")

_spec = importlib.util.spec_from_file_location("substitution_model", _sub_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_get_substitutes = _mod.get_substitutes

# Load person3 cost estimator
_cost_path = os.path.join(_base, "person3", "cost_estimator", "cost_estimator.py")
_cost_spec = importlib.util.spec_from_file_location("cost_estimator", _cost_path)
_cost_mod = importlib.util.module_from_spec(_cost_spec)
_cost_spec.loader.exec_module(_cost_mod)

_estimate_cost = _cost_mod.estimate_cost

# Load person3 reward function
_reward_path = os.path.join(_base, "person3", "mcts", "reward_function.py")
_reward_spec = importlib.util.spec_from_file_location("reward_function", _reward_path)
_reward_mod = importlib.util.module_from_spec(_reward_spec)
_reward_spec.loader.exec_module(_reward_mod)

_compute_reward = _reward_mod.compute_reward

DIETARY_BANNED_INGREDIENTS = {
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
    ]
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
}


def normalize_dietary_restrictions(dietary_restrictions) -> list[str]:
    if not dietary_restrictions:
        return []
    if isinstance(dietary_restrictions, str):
        dietary_restrictions = [dietary_restrictions]

    normalized = []
    for restriction in dietary_restrictions:
        key = re.sub(r"\s+", " ", str(restriction).strip().lower())
        canonical = DIET_ALIASES.get(key)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _ingredient_violates_restrictions(ingredient_name: str, restrictions: list[str]) -> bool:
    ingredient_name = ingredient_name.lower().strip()
    banned = []
    for restriction in restrictions:
        banned.extend(DIETARY_BANNED_INGREDIENTS.get(restriction, []))
    return any(term in ingredient_name for term in banned)


def get_substitutes(
    ingredient: str,
    top_k: int = 3,
    dietary_restrictions=None
) -> list[dict]:
    """
    Return ranked ingredient substitutes using FlavorDB + food embeddings.
    Indian ingredients are prioritised.
    If dietary restrictions are provided, return only valid substitutes.
    """
    try:
        restrictions = normalize_dietary_restrictions(dietary_restrictions)
        fetch_k = max(top_k, top_k * 4) if restrictions else top_k
        results = _get_substitutes(ingredient, top_k=fetch_k)

        if restrictions:
            results = [
                item for item in results
                if not _ingredient_violates_restrictions(item.get("ingredient", ""), restrictions)
            ]

        return results[:top_k]
    except Exception as e:
        return [{"substitute": "N/A", "similarity_score": 0.0, "notes": str(e)}]


def estimate_cost_inr(ingredients: list[str], servings: int = 2) -> dict:
    """
    Estimate total recipe cost in INR with full breakdown.
    """
    try:
        return _estimate_cost(ingredients, servings=servings)
    except Exception as e:
        return {"total_cost": "INR ?", "error": str(e)}


def evaluate_recipe(
    recipe_steps: list[str],
    recipe_ingredients: list[str],
    available_ingredients: list[str],
    dietary_restrictions: list[str] = None,
    available_appliances: list[str] = None,
    max_time_minutes: int = None,
    estimated_time_minutes: int = None
) -> dict:
    """
    Evaluate a generated recipe using the MCTS reward function.
    Returns coherence, constraint satisfaction, feasibility scores.
    """
    try:
        normalized_diet = normalize_dietary_restrictions(dietary_restrictions)
        return _compute_reward(
            recipe_steps=recipe_steps,
            recipe_ingredients=recipe_ingredients,
            available_ingredients=available_ingredients,
            dietary_restrictions=normalized_diet if normalized_diet else None,
            available_appliances=available_appliances,
            max_time_minutes=max_time_minutes,
            estimated_time_minutes=estimated_time_minutes
        )
    except Exception as e:
        return {"error": str(e), "final_reward": 0.0}
