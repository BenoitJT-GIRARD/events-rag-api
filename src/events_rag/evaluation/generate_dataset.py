"""Build the evaluation set of 30 Q&A cases and write it to data/eval/reference_qa.json.

Distribution:
  - 20 positive cases (2 per category x 10 categories)
  - 5 negative cases, out of corpus: another city, or a category the corpus lacks
  - 5 ambiguous cases, vague or dependent on an unstated date
"""

import json
import random
import re
import sys
import time
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


def select_events_for_category(
    events: list[dict],
    keywords: list[str],
    rng: random.Random,
) -> list[dict]:
    """Sample up to MAX_EVENTS_PER_CATEGORY events matching any keyword.

    Sampling is random under a fixed seed rather than first-match: taking the head of the
    file biased the evaluation set toward one slice of the corpus, and made the set
    unstable to compare across ablation runs.
    """
    matched = [
        event
        for event in events
        if any(kw.lower() in event.get("text", "").lower() for kw in keywords)
    ]
    if len(matched) <= MAX_EVENTS_PER_CATEGORY:
        return matched
    return rng.sample(matched, MAX_EVENTS_PER_CATEGORY)


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


# Building the set is 22 calls in a row, which the API treats as a burst and rate-limits.
# Three attempts capped at ten seconds was not enough to ride one out, and a single 429
# aborted the run and discarded every case generated before it.
THROTTLE_SECONDS = 1.5


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def call_llm(model: ChatMistralAI, prompt: str) -> str:
    response = model.invoke([("human", prompt)])
    time.sleep(THROTTLE_SECONDS)
    return response.content


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_positive_prompt(category: str, event_text: str) -> str:
    """Ask for one question about one event.

    One event per call is what makes `source_uid` certain: asking for two questions
    about two events leaves no way to know which event produced which question, and
    that mapping is the relevance label the whole retrieval benchmark rests on.
    """
    return (
        "Tu es un générateur de dataset d'évaluation pour un système RAG d'événements culturels.\n"
        f"À partir de l'événement suivant (catégorie : {category}), "
        'génère UNE question-réponse de type "positif".\n\n'
        f"Événement:\n{event_text}\n\n"
        "Format JSON attendu (tableau d'un seul objet):\n"
        "[\n"
        "  {\n"
        '    "case_type": "positive",\n'
        '    "question": "...",\n'
        '    "ground_truth": "Réponse factuelle de 1-3 phrases mentionnant'
        ' titre, lieu, date et/ou conditions.",\n'
        '    "expected_keywords": ["mot1", "mot2"]\n'
        "  }\n"
        "]\n\n"
        "Retourne UNIQUEMENT le JSON, sans commentaires."
    )


def build_negative_prompt() -> str:
    return (
        "Tu es un générateur de dataset d'évaluation pour un système RAG "
        "d'événements culturels en Occitanie.\n"
        'Génère 5 questions-réponses de type "négatif" — ces questions portent '
        "sur des événements qui N'EXISTENT PAS dans le corpus "
        "(région extérieure comme Paris/Lyon/Bordeaux, ou catégorie absente).\n\n"
        "Format JSON attendu (tableau de 5 objets):\n"
        "[\n"
        "  {\n"
        '    "case_type": "negative",\n'
        '    "question": "...",\n'
        '    "ground_truth": "Il n\'y a pas d\'événements de ce type dans le corpus.",\n'
        '    "expected_keywords": [],\n'
        '    "expected_city": null\n'
        "  }\n"
        "]\n\n"
        "Retourne UNIQUEMENT le JSON, sans commentaires."
    )


def build_ambiguous_prompt() -> str:
    return (
        "Tu es un générateur de dataset d'évaluation pour un système RAG "
        "d'événements culturels en Occitanie.\n"
        'Génère 5 questions-réponses de type "ambigu" — questions vagues ou '
        "dépendantes d'une date spécifique non précisée.\n\n"
        "Format JSON attendu (tableau de 5 objets):\n"
        "[\n"
        "  {\n"
        '    "case_type": "ambiguous",\n'
        '    "question": "...",\n'
        '    "ground_truth": "La réponse dépend de la date ou est trop vague pour être précise.",\n'
        '    "expected_keywords": [],\n'
        '    "expected_city": null\n'
        "  }\n"
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

    rng = random.Random(settings.eval_seed)
    all_cases: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Positive cases — 2 per category x 10 categories = 20
    # ------------------------------------------------------------------
    for category, keywords in CATEGORY_KEYWORDS.items():
        print(f"Generating positive cases for category: {category}...")
        selected = select_events_for_category(events, keywords, rng)

        if not selected:
            print(f"  WARNING: no events found for category '{category}', skipping.")
            continue

        for event in selected:
            prompt = build_positive_prompt(category, format_events_text([event]))

            try:
                raw = call_llm(model, prompt)
                cases = parse_json_from_llm(raw)
            except Exception as exc:
                print(f"  ERROR generating a case for '{category}': {exc}", file=sys.stderr)
                sys.exit(1)

            # Both labels come from the event, never from the model: source_uid is the
            # relevance label the retrieval metrics rest on, and expected_city was being
            # copied from a prompt example naming a town holding 18 of 1000 events.
            for case in cases:
                case["source_uid"] = event["metadata"]["uid"]
                case["expected_city"] = event["metadata"].get("city")
            all_cases.extend(cases)

        print(f"  Generated {len(selected)} case(s).")

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
