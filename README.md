# Events RAG

POC de système **RAG (Retrieval-Augmented Generation)** pour la recommandation d'événements culturels à partir du dataset public OpenDataSoft `evenements-publics-openagenda`, avec **LangChain**, **Mistral**, **FAISS** et une **API FastAPI**.

---

## 1. Objectif du projet

L'objectif de ce projet est de démontrer la faisabilité technique d'un assistant capable de répondre à des questions en langage naturel sur des événements culturels, en s'appuyant sur :

- une ingestion configurable des données d'événements ;
- un prétraitement et un chunking des descriptions ;
- une vectorisation sémantique via **Mistral Embeddings** ;
- une indexation vectorielle locale avec **FAISS** ;
- un retrieval sémantique puis une génération augmentée via **Mistral Chat** ;
- une **API REST FastAPI** permettant de tester rapidement la solution ;
- des **tests**, une **évaluation automatisée** et une **documentation de reproduction**.

---

## 2. Périmètre retenu

Le cahier des charges autorise à cibler une zone géographique au choix, à condition de travailler sur des événements récents (moins d'un an) ou à venir.

Le corpus retenu pour le développement principal est configurable via variables d'environnement. La configuration de référence utilisée pendant le développement est :

- `location_field=city`
- `location_value=Montpellier`
- `date_window_mode=rolling`
- `date_window_days=365`

Ce paramétrage produit un corpus d'environ **688 événements**, ce qui constitue une taille adaptée pour un POC démontrable, reproductible et raisonnable en coût/temps d'indexation.

---

## 3. Source de données

Le projet s'appuie sur le dataset public OpenDataSoft :

- **Dataset** : `evenements-publics-openagenda`
- **Endpoint** : `/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records`

### Pourquoi ce choix ?

Le dataset public OpenDataSoft a été retenu. Ce choix permet :

- une **reproductibilité directe** ;
- l'absence de clé API spécifique à OpenAgenda ;

Le projet reste donc aligné avec le besoin métier : interroger des événements publics issus de l'écosystème OpenAgenda, via leur exposition publique sur OpenDataSoft.

---

## 4. Stack technique

### Runtime / API / orchestration

- **Python 3.12**
- **FastAPI**
- **Uvicorn**
- **LangChain**

### RAG / vectorisation

- **MistralAIEmbeddings** pour les embeddings
- **ChatMistralAI** pour la génération
- **FAISS** pour la base vectorielle
- **RecursiveCharacterTextSplitter** pour le chunking

### Qualité / industrialisation

- **uv** pour la gestion d'environnement et des dépendances
- **Git Flow** pour l'organisation Git
- **pytest** pour les tests
- **Ruff** pour lint + format
- **Bandit** pour l'analyse sécurité
- **pre-commit**
- **GitHub Actions** pour la CI
- **structlog** pour les logs
- **Docker** pour l'exécution locale conteneurisée

### Évaluation

- **Évaluation heuristique custom**
- **Ragas** pour les métriques RAG avancées
- **datasets** pour le format d'entrée Ragas

---

## 5. Structure du projet

```text
events-rag/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── eval.yml
├── data/
│   ├── raw/
│   │   └── events.json
│   ├── eval/
│   │   ├── reference_qa.json
│   │   ├── evaluation_results.json
│   │   └── ragas_results.json
│   └── faiss/
│       └── events_index/
├── docs/
│   ├── rapport_technique.md
│   ├── autoevaluation_notes.md
│   └── demo_scenarios.md
├── scripts/
│   ├── api_test.py
│   ├── build_dataset.py
│   ├── build_index.py
│   ├── evaluate_rag.py
│   ├── evaluate_ragas.py
│   └── run_local.py
├── src/
│   └── events_rag/
│       ├── api/
│       ├── evaluation/
│       ├── ingestion/
│       ├── rag/
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── integration/
│   └── unit/
├── .env.example
├── .dockerignore
├── Dockerfile
├── README.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

## 6. Architecture fonctionnelle

Le pipeline suit les étapes suivantes :

1. **Récupération des événements** depuis OpenDataSoft
2. **Filtrage géographique et temporel**
3. **Transformation** de chaque événement en document textuel
4. **Chunking** des documents
5. **Embeddings Mistral**
6. **Indexation FAISS**
7. **Recherche sémantique** des documents les plus proches
8. **Construction d'un prompt contextuel**
9. **Génération de réponse** via Mistral
10. **Exposition du système** via FastAPI

---

## 7. Installation

### 7.1. Cloner le projet

```bash
git clone <repo_url>
cd events-rag
```

### 7.2. Installer les dépendances avec uv

```bash
uv sync --all-groups
```

Le projet utilise `pyproject.toml` et `uv.lock` comme source principale de vérité.  
Un `requirements.txt` exporté est également fourni pour faciliter certaines reproductions.

### 7.3. Variables d'environnement

Créer un fichier `.env` à partir de `.env.example`.

Exemple minimal :

```dotenv
EVENTS_ENV=dev
EVENTS_LOG_LEVEL=INFO

EVENTS_LOCATION_FIELD=city
EVENTS_LOCATION_VALUE=Montpellier
EVENTS_LANG=fr
EVENTS_TIMEZONE=Europe/Paris

EVENTS_DATE_WINDOW_MODE=rolling
EVENTS_DATE_WINDOW_DAYS=365

EVENTS_INGESTION_BATCH_SIZE=100
EVENTS_INGESTION_MAX_RECORDS=700

EVENTS_MISTRAL_API_KEY=your_mistral_api_key
EVENTS_EMBEDDING_MODEL=mistral-embed
EVENTS_CHAT_MODEL=mistral-small-latest

EVENTS_FAISS_INDEX_NAME=events_index
EVENTS_REBUILD_TOKEN=change_me_local_token
```

---

## 8. Dépendances

### Dépendances runtime

Le projet dépend notamment de :

- `fastapi`
- `uvicorn[standard]`
- `httpx`
- `langchain`
- `langchain-community`
- `langchain-mistralai`
- `langchain-text-splitters`
- `faiss-cpu`
- `pydantic-settings`
- `python-dotenv`
- `structlog`
- `tenacity`

### Dépendances de test / qualité

- `pytest`
- `pytest-cov`
- `pytest-asyncio`
- `ragas`
- `datasets`
- `ruff`
- `bandit[toml]`
- `pre-commit`

---

## 9. Reproduction du pipeline

### 9.1. Construire le dataset brut

```bash
uv run python scripts/build_dataset.py
```

Sortie attendue :

- `data/raw/events.json`

### 9.2. Construire l'index vectoriel

```bash
uv run python scripts/build_index.py
```

Sorties attendues :

- `data/faiss/events_index/index.faiss`
- `data/faiss/events_index/index.pkl`
- `data/faiss/events_index/manifest.json`

### 9.3. Lancer les tests

```bash
uv run pytest
```

### 9.4. Lancer l'évaluation heuristique

```bash
uv run python scripts/evaluate_rag.py
```

Sortie :

- `data/eval/evaluation_results.json`

### 9.5. Lancer l'évaluation Ragas

```bash
uv run python scripts/evaluate_ragas.py
```

Sortie :

- `data/eval/ragas_results.json`

### 9.6. Lancer l'API

```bash
uv run fastapi dev src/events_rag/api/main.py
```

Documentation interactive :

- `http://127.0.0.1:8000/docs`

### 9.7. Lancer tout le pipeline localement

```bash
uv run python scripts/run_local.py
```

Ce script exécute automatiquement :

- la construction du dataset ;
- la reconstruction de l'index ;
- l'évaluation heuristique ;
- puis le lancement de l'API.

---

## 10. API REST

### `GET /health`

Vérifie que l'API répond.

Exemple de réponse :

```json
{
  "status": "ok"
}
```

### `GET /metadata`

Retourne la configuration métier du corpus actif :

- filtre géographique ;
- valeur géographique ;
- langue ;
- mode temporel ;
- fenêtre temporelle ;
- `retrieval_k`.

### `POST /ask`

Interroge le système RAG.

Exemple de requête :

```json
{
  "question": "Quels événements gratuits ont lieu à Montpellier ?",
  "top_k": 5
}
```

Exemple de réponse :

```json
{
  "answer": "…",
  "sources": [
    {
      "uid": "123",
      "title": "Nom de l'événement",
      "city": "Montpellier",
      "date": "2026-03-10T18:00:00+00:00",
      "score": null
    }
  ]
}
```

### `POST /rebuild`

Reconstruit l'index vectoriel à la demande.

Exemple de requête :

```json
{
  "token": "change_me_local_token"
}
```

Le token protège l'endpoint contre une reconstruction non autorisée.

---

## 11. Test fonctionnel de l'API

Un script dédié est fourni :

```bash
uv run python scripts/api_test.py
```

Ce script teste :

- `/health`
- `/metadata`
- `/ask`

Il constitue une preuve simple et relançable de bon fonctionnement local de l'API.

---

## 12. Qualité logicielle

### Tests

Le projet contient :

- des **tests unitaires** sur l'ingestion, le preprocessing, l'indexation et les fonctions d'évaluation ;
- des **tests d'intégration API** pour vérifier la validation et les endpoints de base ;
- un **script de test API** pour démonstration rapide.

### Lint / sécurité

Les outils suivants sont intégrés :

- `ruff check`
- `ruff format`
- `bandit`

### CI

Le dépôt inclut des workflows GitHub Actions :

- `.github/workflows/ci.yml` : lint + sécurité + tests
- `.github/workflows/eval.yml` : reconstruction + évaluations relançables

---

## 13. Stratégie d'évaluation

Le projet utilise **deux niveaux d'évaluation complémentaires**.

### 13.1. Évaluation heuristique

Le script `scripts/evaluate_rag.py` compare les réponses générées à un jeu de test annoté simple :

- couverture de mots-clés attendus ;
- cohérence géographique des sources ;
- classification :
  - `correct`
  - `partially_correct`
  - `incorrect`

Cette méthode est :

- simple ;
- rapide ;
- relançable ;
- utile pour la non-régression.

### 13.2. Évaluation Ragas

Le script `scripts/evaluate_ragas.py` produit des métriques plus avancées :

- `faithfulness`
- `answer_relevancy`
- `context_precision`

L'évaluation Ragas est exécutée avec :

- **ChatMistralAI** comme LLM d'évaluation ;
- **MistralAIEmbeddings** pour les embeddings ;
- **sans dépendance à OpenAI**.

### Limite importante

Les scores élevés obtenus sur un petit jeu de test ne doivent pas être interprétés comme une preuve de perfection générale du système.  
Ils montrent surtout que le pipeline est fonctionnel et cohérent sur le périmètre testé. Une montée en robustesse nécessiterait :

- un jeu de test plus large ;
- une annotation humaine plus fine ;
- des cas négatifs plus variés ;
- une analyse détaillée des erreurs.

---

### 14. Choix techniques clés

### Pourquoi FAISS ?

- simple à intégrer ;
- très adapté à un POC local ;
- performant sur quelques centaines à quelques milliers de chunks ;
- facilement reconstruisible.

### Pourquoi Mistral pour embeddings et génération ?

- homégénéité de la stack ;
- simplicité d'intégration dans LangChain ;

### Pourquoi pas de reranker ?

Le reranking n était pas requis par le cahier des charges.  
Pour ce POC, le pipeline retrieval + génération couvre correctement les attentes. Le reranking constitue une **piste d'amélioration**, pas une condition de réussite.

### Pourquoi uv plutôt qu'un simple requirements.txt ?

`uv` fournit :

- un environnement reproductible ;
- un lockfile (`uv.lock`) ;
- une gestion claire des groupes de dépendances ;
- une meilleure ergonomie moderne.

Un `requirements.txt` exporté est toutefois fourni pour faciliter certaines installations.

---

## 15. Exécution avec Docker

### Build

```bash
docker build -t events-rag .
```

### Run

```bash
docker run --rm -p 8000:8000 --env-file .env events-rag
```

### Avec Docker Compose

```bash
docker compose up --build
```

### Remarque importante

Le conteneur permet d'exécuter localement l'API, mais la génération et les embeddings restent dépendants de l'API Mistral.  
Le système est donc **conteneurisé**, mais pas entièrement autonome hors ligne.

---

## 16. Exemples d'usage

### Via Swagger

- ouvrir `http://127.0.0.1:8000/docs`
- tester `/ask`
- tester `/rebuild`

### Via curl (Windows PowerShell)

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{"question":"Quels événements musicaux ont lieu à Montpellier ?","top_k":5}"
```

```bash
curl -X POST "http://127.0.0.1:8000/rebuild" ^
  -H "Content-Type: application/json" ^
  -d "{"token":"change_me_local_token"}"
```

---

## 17. Limites actuelles

- corpus limité à une zone géographique et une fenêtre temporelle configurées ;
- dépendance à Mistral pour les embeddings et la génération ;
- évaluation encore fondée sur un petit jeu de test ;
- absence d'historique conversationnel ;
- pas de reranking dédié ;
- pas d'interface front dédiée ;
- scores de source encore simples.

---

## 18. Pistes d'amélioration

- enrichir le jeu de test annoté ;
- intégrer davantage de cas négatifs et ambigus ;
- ajouter un reranker ou un système de re-scoring ;
- amiliorer les scores et justifications de sources ;
- filtrer plus finement par catégories d'événements ;
- ajouter une interface utilisateur même simple ;
- industrialiser davantage le monitoring et l'observabilité ;
- préparer un déploiement cloud plus robuste.

---

### 19. Démo recommandée

Trois scénarios simples et parlants :

1. **Quels événements musicaux ont lieu à Montpellier ?**
2. **Y a-t-il des événements gratuits à Montpellier ?**
3. **Quels événements pour les enfants sont prévus à Montpellier ?**

---

## 20. Résumé

Le projet livre un **POC RAG complet, testable et démontrable**, avec :

- ingestion de données ;
- preprocessing ;
- chunking ;
- embeddings ;
- indexation FAISS ;
- retrieval ;
- génération via Mistral ;
- API FastAPI ;
- endpoint `/rebuild` ;
- tests ;
- évaluation heuristique ;
- évaluation Ragas ;
- CI ;
- script de lancement local ;
- conteneurisation Docker.

Il répond au périmètre attendu pour un prototype métier crédible et constitue une base solide pour un élargissement futur.
