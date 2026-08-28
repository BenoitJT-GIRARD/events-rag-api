"""Build the evaluation set of 30 Q&A cases and write it to data/eval/reference_qa.json.

Distribution:
  - 20 positive cases (2 per category x 10 categories)
  - 5 negative cases, out of corpus: another city, or a category the corpus lacks
  - 5 ambiguous cases, vague or dependent on an unstated date
"""

import json
import re
import sys
from pathlib import Path

from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential

from events_rag.config import get_settings

# ---------------------------------------------------------------------------
# Category → keyword mapping
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "musique": ["musique", "concert", "musical", "jazz", "rock", "classique"],
    "théâtre": ["théâtre", "pièce", "spectacle", "comédie", "dramaturgie"],
    "danse": ["danse", "ballet", "chorégraphie", "danseur"],
    "exposition": ["exposition", "art", "musée", "galerie", "œuvre", "peinture"],
    "gratuit": ["gratuit", "free", "sans réservation"],
    "enfants": ["enfant", "famille", "jeune public", "kids"],
    "sport": ["sport", "marathon", "course", "vélo", "natation"],
    "patrimoine": ["patrimoine", "histoire", "château", "monument", "archéologie"],
    "conférence": ["conférence", "débat", "forum", "rencontre", "atelier"],
    "festival": ["festival", "fête", "carnaval"],
}

MAX_EVENTS_PER_CATEGORY = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_events(events_path: Path) -> list[dict]:
    with events_path.open(encoding="utf-8") as f:
        return json.load(f)


def select_events_for_category(events: list[dict], keywords: list[str]) -> list[dict]:
    """Return up to MAX_EVENTS_PER_CATEGORY events whose text contains any keyword."""
    matched: list[dict] = []
    for event in events:
        text_lower = event.get("text", "").lower()
        if any(kw.lower() in text_lower for kw in keywords):
            matched.append(event)
            if len(matched) >= MAX_EVENTS_PER_CATEGORY:
                break
    return matched


def format_events_text(events: list[dict]) -> str:
    """Format a list of events into a human-readable string for the prompt."""
    lines: list[str] = []
    for event in events:
        meta = event.get("metadata", {})
        lines.append(
            f"- Titre: {meta.get('title', 'N/A')}\n"
            f"  Lieu: {meta.get('location_name', 'N/A')}, {meta.get('city', 'N/A')}\n"
            f"  Date: {meta.get('date', 'N/A')}\n"
            f"  Conditions: {meta.get('conditions', 'N/A')}\n"
            f"  Mots-clés: {', '.join(meta.get('keywords', []))}\n"
            f"  Texte: {event.get('text', '')[:300]}"
        )
    return "\n\n".join(lines)


def parse_json_from_llm(text: str) -> list[dict]:
    """Strip markdown fences and parse JSON from the LLM response."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(model: ChatMistralAI, prompt: str) -> str:
    response = model.invoke([("human", prompt)])
    return response.content


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_positive_prompt(category: str, events_text: str) -> str:
    return (
        "Tu es un générateur de dataset d'évaluation pour un système RAG d'événements culturels.\n"
        f"À partir des événements suivants (catégorie : {category}), "
        'génère 2 questions-réponses de type "positif".\n\n'
        f"Événements:\n{events_text}\n\n"
        "Format JSON attendu (tableau de 2 objets):\n"
        "[\n"
        "  {{\n"
        '    "id": "q_cat_1",\n'
        '    "case_type": "positive",\n'
        '    "question": "...",\n'
        '    "ground_truth": "Réponse factuelle de 1-3 phrases mentionnant'
        ' titre, lieu, date et/ou conditions.",\n'
        '    "expected_keywords": ["mot1", "mot2"],\n'
        '    "expected_city": "Montpellier"\n'
        "  }}\n"
        "]\n\n"
        "Retourne UNIQUEMENT le JSON, sans commentaires."
    )


def build_negative_prompt() -> str:
    return (
        "Tu es un générateur de dataset d'évaluation pour un système RAG "
        "d'événements culturels à Montpellier.\n"
        'Génère 5 questions-réponses de type "négatif" — ces questions portent '
        "sur des événements qui N'EXISTENT PAS dans le corpus "
        "(autre ville comme Paris/Lyon, catégorie absente comme opéra/cirque).\n\n"
        "Format JSON attendu (tableau de 5 objets):\n"
        "[\n"
        "  {{\n"
        '    "id": "q_neg_1",\n'
        '    "case_type": "negative",\n'
        '    "question": "...",\n'
        '    "ground_truth": "Il n\'y a pas d\'événements de ce type dans le corpus.",\n'
        '    "expected_keywords": [],\n'
        '    "expected_city": null\n'
        "  }}\n"
        "]\n\n"
        "Retourne UNIQUEMENT le JSON, sans commentaires."
    )


def build_ambiguous_prompt() -> str:
    return (
        "Tu es un générateur de dataset d'évaluation pour un système RAG "
        "d'événements culturels à Montpellier.\n"
        'Génère 5 questions-réponses de type "ambigu" — questions vagues ou '
        "dépendantes d'une date spécifique non précisée.\n\n"
        "Format JSON attendu (tableau de 5 objets):\n"
        "[\n"
        "  {{\n"
        '    "id": "q_amb_1",\n'
        '    "case_type": "ambiguous",\n'
        '    "question": "...",\n'
        '    "ground_truth": "La réponse dépend de la date ou est trop vague pour être précise.",\n'
        '    "expected_keywords": [],\n'
        '    "expected_city": null\n'
        "  }}\n"
        "]\n\n"
        "Retourne UNIQUEMENT le JSON, sans commentaires."
    )


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_eval_dataset() -> list[dict]:
    settings = get_settings()

    if not settings.mistral_api_key:
        print("ERROR: EVENTS_RAG_MISTRAL_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    events_path = settings.raw_data_dir / "events.json"
    if not events_path.exists():
        print(f"ERROR: events file not found at {events_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading events from {events_path}...")
    events = load_events(events_path)
    print(f"  Loaded {len(events)} events.")

    model = ChatMistralAI(
        model=settings.chat_model,
        api_key=settings.mistral_api_key,
        temperature=0,
    )

    all_cases: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Positive cases — 2 per category × 10 categories = 20
    # ------------------------------------------------------------------
    for category, keywords in CATEGORY_KEYWORDS.items():
        print(f"Generating positive cases for category: {category}...")
        selected = select_events_for_category(events, keywords)

        if not selected:
            print(f"  WARNING: no events found for category '{category}', skipping.")
            continue

        events_text = format_events_text(selected[:2])
        prompt = build_positive_prompt(category, events_text)

        try:
            raw = call_llm(model, prompt)
            cases = parse_json_from_llm(raw)
            all_cases.extend(cases)
            print(f"  Generated {len(cases)} case(s).")
        except Exception as exc:
            print(f"  ERROR generating positive cases for '{category}': {exc}", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Negative cases — 5 total
    # ------------------------------------------------------------------
    print("Generating negative cases...")
    try:
        raw = call_llm(model, build_negative_prompt())
        neg_cases = parse_json_from_llm(raw)
        all_cases.extend(neg_cases)
        print(f"  Generated {len(neg_cases)} negative case(s).")
    except Exception as exc:
        print(f"ERROR generating negative cases: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Ambiguous cases — 5 total
    # ------------------------------------------------------------------
    print("Generating ambiguous cases...")
    try:
        raw = call_llm(model, build_ambiguous_prompt())
        amb_cases = parse_json_from_llm(raw)
        all_cases.extend(amb_cases)
        print(f"  Generated {len(amb_cases)} ambiguous case(s).")
    except Exception as exc:
        print(f"ERROR generating ambiguous cases: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Reassign IDs sequentially
    # ------------------------------------------------------------------
    if len(all_cases) != 30:
        print(f"ERROR: expected 30 cases, got {len(all_cases)}", file=sys.stderr)
        sys.exit(1)

    for i, case in enumerate(all_cases, start=1):
        case["id"] = f"q{i:02d}"

    # ------------------------------------------------------------------
    # 4b. Normalise negative cases
    # ------------------------------------------------------------------
    # The prompt asks for empty keywords on negative cases and the model ignores it,
    # returning the terms of the question instead. Keeping them would make keyword
    # coverage score 1.0 whenever the answer merely echoes the question, and
    # expected_city would name a city the corpus cannot contain by construction.
    for case in all_cases:
        if case.get("case_type") == "negative":
            case["expected_keywords"] = []
            case["expected_city"] = None

    # ------------------------------------------------------------------
    # 5. Write output
    # ------------------------------------------------------------------
    output_path = settings.eval_data_dir / "reference_qa.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(all_cases)} cases to {output_path}")
    return all_cases


def summarize_cases(cases: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"total": len(cases)}
    for case in cases:
        ct = case.get("case_type", "positive")
        counts[ct] = counts.get(ct, 0) + 1
    return counts
