from events_rag.evaluation.difficulty import content_tokens, lexical_overlap


def test_content_tokens_drops_short_words_and_punctuation() -> None:
    assert content_tokens("Où et quand a lieu le concert ?") == {"quand", "lieu", "concert"}


def test_overlap_is_one_when_every_content_word_appears_in_the_document() -> None:
    assert lexical_overlap("concert chorale", "Concert de la chorale du village") == 1.0


def test_overlap_is_zero_when_nothing_matches() -> None:
    assert lexical_overlap("archeologie musee", "concert de jazz") == 0.0


def test_overlap_ignores_case_and_accents() -> None:
    assert lexical_overlap("ARCHÉOLOGIE", "village de l'archeologie") == 1.0


def test_a_question_quoting_the_title_scores_higher_than_a_paraphrase() -> None:
    document = "Concert de Noel de Goma Esperance, chorale et orchestre a Castanet-Tolosan"

    quoted = lexical_overlap("Quel est le titre du concert de Noel de Goma Esperance ?", document)
    paraphrased = lexical_overlap(
        "Une chorale donne un spectacle pour les fetes pres de Toulouse ?", document
    )

    assert quoted > paraphrased


def test_overlap_of_an_empty_question_is_zero() -> None:
    assert lexical_overlap("", "anything") == 0.0
