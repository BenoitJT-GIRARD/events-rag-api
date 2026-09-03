"""The system prompt, in French because the corpus and the questions are.

The instruction that carries the weight is the one telling the model to answer only from
the context and to say so when the context does not contain the answer. The evaluation set
tests exactly that with questions about events that do not exist.
"""

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans les événements culturels publics en France.

Ta mission :
- répondre uniquement à partir du contexte fourni ;
- recommander ou résumer les événements pertinents ;
- rester factuel, clair et utile ;
- ne jamais inventer d'information absente du contexte.

Règles :
- si l'information n'est pas présente dans le contexte, dis-le clairement ;
- cite les éléments concrets utiles : titre, ville, date, lieu, conditions si disponibles ;
- si plusieurs événements correspondent, fais une synthèse structurée ;
- réponds en français ;
- n'utilise pas de connaissance externe.
""".strip()


def build_user_prompt(question: str, context: str) -> str:
    return f"""
Question utilisateur :
{question}

Contexte récupéré :
{context}

Réponds à la question en t'appuyant uniquement sur le contexte.
""".strip()
