# AgriMind 🌱

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![CI](https://github.com/Teja2205/AgriMind/actions/workflows/deploy.yml/badge.svg)](https://github.com/Teja2205/AgriMind/actions)

AI-powered crop disease diagnosis service. Upload a crop photo and get instant disease identification, pesticide recommendations with exact dosages, and field-aware treatment plans — powered by GPT-4o Vision, RAG, and LangGraph agents.

---

## Architecture

```
User Request (image + crop)
      ↓
FastAPI + API Key Auth
      ↓
LangGraph Agent
      ├── check_image_node (GPT-4o-mini guardrail)
      ├── get_context_node
      │       ├── FarmOS MCP → weather + pest alerts + soil data
      │       └── ChromaDB RAG → disease knowledge base (cross-encoder reranked)
      └── diagnose_node (GPT-4o Vision)
      ↓
Structured Response (disease, severity, treatment, sources)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic |
| Agent | LangGraph (3-node state machine) |
| Vision | OpenAI GPT-4o Vision |
| RAG | ChromaDB + text-embedding-3-small |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Field Intel | FarmOS MCP (weather, pests, soil) |
| Guardrails | GPT-4o-mini image validation |
| Caching | In-memory with MD5 hashing |
| Evaluation | RAGAS (faithfulness: 0.18, relevancy: 0.62) |
| Deployment | Docker + GitHub Actions CI/CD |

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/Teja2205/AgriMind.git
cd AgriMind
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

**2. Create `.env` file**
```
OPENAI_API_KEY=sk-...
AGRIMIND_API_KEY=your-chosen-api-key
```

**3. Build knowledge base**
```bash
# Add crop disease PDFs to data/ folder
python ingest.py
```

**4. Run**
```bash
uvicorn main:app --reload
# Open http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/` | No | Health check |
| POST | `/diagnose` | API Key | Structured JSON diagnosis |
| POST | `/diagnose/stream` | API Key | Streaming plain-text diagnosis |

**Authentication** — pass your API key in the request header:
```
x-api-key: your-chosen-api-key
```

---

## Example Request

```bash
curl -X POST "http://localhost:8000/diagnose/stream" \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://your-crop-image.jpg", "crop": "tomato"}'
```

## Example Response

```json
{
  "disease_name": "Early Blight",
  "severity": 3,
  "description": "Dark spots with concentric rings on lower leaves, spreading upward.",
  "treatment_steps": [
    "Remove and destroy infected leaves.",
    "Apply chlorothalonil fungicide at 1.5 lbs/acre every 7-10 days.",
    "Ensure proper plant spacing for air circulation.",
    "Avoid overhead irrigation."
  ],
  "sources": ["data/Tomato_diseases.pdf"]
}
```

---

## Run with Docker

```bash
docker build -t agrimind .
docker run -p 8000:8000 --env-file .env agrimind
```

---

## Evaluation

RAGAS scores on 3-question golden dataset:

| Metric | Score |
|---|---|
| Faithfulness | 0.18 |
| Answer Relevancy | 0.62 |
| Context Precision | 0.33 |

Low faithfulness indicates the LLM relies on training data over retrieved chunks. Improving with larger golden dataset and better PDF coverage.

---

## Related

- [FarmOS MCP](https://github.com/Teja2205/farmos-mcp) — Agent infrastructure providing weather, pest, and soil data

---

## Author

Built by Teja Guduguntla as part of an AI engineering portfolio targeting FAANG roles in 2026.