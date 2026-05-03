# Query optimizer + synonym expander.
# Cleans natural language into retrieval-ready keywords,extracts exclusions, and expands with WordNet synonyms.

import os
import re
import json
import requests

#defined stopwords 
_STOPWORDS = {
    "i", "want", "to", "make", "a", "an", "the", "and", "or", "but",
    "it", "should", "be", "can", "you", "give", "me", "some", "any",
    "please", "help", "with", "for", "of", "in", "is", "are", "do",
    "have", "has", "get", "my", "we", "us", "how", "what", "which",
    "something", "anything", "ideas", "idea", "recipe", "dish", "meal",
    "food", "cook", "cooking", "prepare", "need", "like", "would",
    "could", "also", "just", "really", "very", "quite", "that", "this",
    "good", "nice", "great", "quick", "easy", "simple", "best", "try",
}

# diet related preference
_DIET_KEYWORDS = {
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "jain": "jain",
    "gluten-free": "gluten_free",
    "gluten free": "gluten_free",
    "dairy-free": "dairy_free",
    "dairy free": "dairy_free",
    "halal": "halal",
}

# negations
_NEGATIVE_PATTERNS = [
    r"\bno\s+(\w+)",
    r"\bwithout\s+(\w+)",
    r"\b(\w+)[-\s]free\b",
    r"\bdon'?t\s+(?:use|add|want|include)\s+(\w+)",
    r"\bavoid\s+(\w+)",
    r"\bexclude\s+(\w+)",
    r"\bnot\s+(?:use|add|include)\s+(\w+)",
]

_IGNORE_EXCLUSION_TOKENS = {
    "the", "any", "all", "that", "this", "with", "from", "and", "or",
    "not", "too", "very", "much", "many", "it", "gluten", "dairy",
}

_CUISINE_KEYWORDS = [
    "indian", "italian", "mexican", "chinese", "thai", "french",
    "japanese", "korean", "mediterranean", "american", "greek"
]

# food synonyms
# WordNet doesn't know domain-specific food terms well,
# so we maintain a curated map for the most common recipe words.
_FOOD_SYNONYMS = {
    "cake":      ["pastry", "torte", "gateau", "loaf"],
    "cookie":    ["biscuit", "shortbread"],
    "bread":     ["loaf", "roll", "bun"],
    "pasta":     ["noodles", "spaghetti", "penne", "fettuccine"],
    "chicken":   ["poultry", "hen"],
    "potato":    ["spud", "aloo"],
    "tomato":    ["tamatar"],
    "spinach":   ["palak"],
    "lentil":    ["dal", "dhal"],
    "rice":      ["pilaf", "pulao", "biryani"],
    "yogurt":    ["curd", "dahi"],
    "flatbread": ["roti", "chapati", "naan", "paratha"],
    "pancake":   ["crepe", "dosa"],
    "stew":      ["curry", "gravy", "braise"],
    "salad":     ["slaw", "raita"],
    "soup":      ["broth", "chowder", "bisque", "rasam"],
    "fry":       ["saute", "stir-fry", "toss"],
    "moist":     ["tender", "soft", "fluffy"],
    "crispy":    ["crunchy", "golden", "crisp"],
}


def _get_wordnet_synonyms(word: str, max_synonyms: int = 2) -> list[str]:
    """
    Get synonyms from WordNet via nltk.
    Downloads wordnet on first call if not present.
    Returns empty list on any failure — non-critical.
    """
    try:
        import nltk
        from nltk.corpus import wordnet
        try:
            wordnet.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word and len(name) > 2:
                    synonyms.add(name)
                if len(synonyms) >= max_synonyms:
                    break
            if len(synonyms) >= max_synonyms:
                break
        return list(synonyms)
    except Exception:
        return []


def _expand_with_synonyms(keywords: list[str]) -> list[str]:
    """
    Expand a keyword list with synonyms.
    Checks curated map first, falls back to WordNet.
    Returns original + synonyms, deduplicated.
    """
    expanded = list(keywords)
    seen = set(keywords)

    for kw in keywords:
        # Curated synonyms first (more reliable for food)
        curated = _FOOD_SYNONYMS.get(kw, [])
        for syn in curated[:2]:
            if syn not in seen:
                expanded.append(syn)
                seen.add(syn)

        # WordNet fallback for words not in curated map
        if kw not in _FOOD_SYNONYMS:
            wn_syns = _get_wordnet_synonyms(kw, max_synonyms=2)
            for syn in wn_syns:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

    return expanded


# grok

def _call_groq(prompt: str, groq_key: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.0,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _llm_optimize(query: str, groq_key: str) -> dict:
    prompt = (
        "You are a query optimizer for a recipe search engine.\n"
        "Given a user's natural language query, extract the following and return ONLY valid JSON:\n"
        "{\n"
        '  "optimized_query": "<2-5 keywords: dish type, cuisine, texture, diet — NO filler words>",\n'
        '  "exclusions": ["<ingredient user said NO to>", ...],\n'
        '  "diet": "<vegan|vegetarian|jain|gluten_free|dairy_free|halal|null>",\n'
        '  "cuisine": "<indian|italian|mexican|chinese|etc|null>"\n'
        "}\n\n"
        "Rules:\n"
        "- optimized_query: SHORT keyword string only. No filler words.\n"
        "- exclusions: ingredients explicitly rejected\n"
        "- diet/cuisine: only if explicitly mentioned, else null\n"
        "- Return ONLY the JSON. No explanation, no markdown.\n\n"
        f"Query: {query}\n\nJSON:"
    )
    try:
        raw = _call_groq(prompt, groq_key)
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[query_expander] LLM parse failed: {e}. Using rule-based fallback.")
        return {}


# fallback incase word not recognised 

def _rule_based_optimize(query: str) -> dict:
    query_lower = query.lower()

    exclusions = set()
    for pattern in _NEGATIVE_PATTERNS:
        for match in re.finditer(pattern, query_lower):
            token = match.group(1).strip().rstrip("s")
            if token and token not in _IGNORE_EXCLUSION_TOKENS and len(token) > 2:
                exclusions.add(token)

    diet = None
    for keyword, canonical in _DIET_KEYWORDS.items():
        if keyword in query_lower:
            diet = canonical
            break

    cuisine = next((c for c in _CUISINE_KEYWORDS if c in query_lower), None)

    cleaned = query_lower
    for pattern in _NEGATIVE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    seen = set()
    keywords = []
    for t in cleaned.split():
        if t not in _STOPWORDS and len(t) > 2 and t not in seen:
            seen.add(t)
            keywords.append(t)

    optimized_query = " ".join(keywords) if keywords else query

    return {
        "optimized_query": optimized_query,
        "exclusions": list(exclusions),
        "diet": diet,
        "cuisine": cuisine,
    }


#main

def optimize_query(query: str, expand_synonyms: bool = True) -> dict:
    """
    Takes raw user query, returns structured dict for retrieval.

    Returns:
        {
            "optimized_query":  "cake vegan indian pastry torte",  ← keywords + synonyms
            "original_query":   "i want a vegan indian cake...",   ← for LLM
            "exclusions":       ["chocolate"],                     ← for filtering
            "diet":             "vegan",                           ← for constraints
            "cuisine":          "indian",
        }

    expand_synonyms=True adds WordNet + curated synonyms to optimized_query,
    improving BM25 recall for recipes that use different terminology.
    Set False if you want to compare retrieval with/without expansion.
    """
    groq_key = os.environ.get("GROQ_API_KEY")

    llm_result = {}
    method = "fallback"
    if groq_key:
        llm_result = _llm_optimize(query, groq_key)
        method = "llm"

    rule_result = _rule_based_optimize(query)

    optimized_query = llm_result.get("optimized_query") or rule_result["optimized_query"]
    exclusions      = llm_result.get("exclusions")      or rule_result["exclusions"]
    diet            = llm_result.get("diet")             or rule_result.get("diet")
    cuisine         = llm_result.get("cuisine")          or rule_result.get("cuisine")

    # Synonym expansion
    if expand_synonyms and optimized_query:
        keywords = optimized_query.split()
        expanded = _expand_with_synonyms(keywords)
        optimized_query = " ".join(expanded)

    result = {
        "optimized_query": optimized_query,
        "original_query":  query,
        "exclusions":      exclusions,
        "diet":            diet,
        "cuisine":         cuisine,
    }

    print(
        f"[query_expander] ({method}) '{query[:60]}'\n"
        f"  → optimized : '{optimized_query}'\n"
        f"  → exclusions: {exclusions}\n"
        f"  → diet: {diet} | cuisine: {cuisine}"
    )

    return result