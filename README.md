---
license: mit
title: Wealth Management Advisor
sdk: docker
emoji: 📈
colorFrom: gray
colorTo: green
sdk_version: 6.27.0
---

# Wealth Management Advisor API

FastAPI service that answers financial education questions using Gemini, a local FAISS index, and CPU sentence-transformer models. It is educational software, not a licensed financial adviser.

## Current features

- FastAPI health, readiness, and streaming chat endpoints
- Server-sent event responses from Gemini
- Hugging Face finance dataset ingestion on first start
- FAISS retrieval with local JSON document storage
- CPU-only embedding and reranking
- Basic request validation, rate limiting, CORS, and gzip support

## Runtime components

| Component | Technology |
|---|---|
| API | FastAPI |
| LLM | Google Gemini |
| Embeddings | all-MiniLM-L6-v2 |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| Vector store | FAISS files (`finance.index`, `finance_docs.json`) |
| Streaming | SSE |
| Dataset | Hugging Face Datasets |

## Project structure

```text
app.py                 FastAPI application
main.py                RAG, Gemini, and market-data logic
requirements.txt       Python dependencies
finance.index          Generated FAISS index
finance_docs.json      Generated document metadata
```

## Memory and 4 GB deployments

The service is configured for CPU execution and a small dataset slice (`train[:2000]`). A 4 GB machine may run it, but this is not a guarantee: PyTorch, the embedding model, reranker, Python process, and first-start dataset/index build can create a high memory peak. The first start is the riskiest because it downloads models and builds the index.

For a 4 GB host:

- Use one Uvicorn worker only.
- Do not run multiple replicas on the same host.
- Keep `DATASET_SPLIT` at `train[:2000]` or reduce it.
- Build `finance.index` and `finance_docs.json` once, then reuse them.
- Leave at least 1 GB of free disk space for model and dataset caches.
- Monitor memory during the first start; use a 6 to 8 GB host if it is killed by the platform.

The application does not currently use Redis, PostgreSQL, JWT authentication, or a database. Do not configure those services unless you add their implementation.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service metadata |
| GET | `/health` | Liveness response |
| GET | `/ready` | Ready after the RAG index is loaded |
| POST | `/chat` | SSE chat response |

Example request:

```json
{"query": "How can I plan for inflation?"}
```

# Run locally

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

Set the required secret before starting:

```powershell
$env:GEMINI_API_KEY = "your-key"
py -3 -m uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1
```

For Linux or a container, use `uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1`.

# Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | None | Google Gemini API key |
| `FRONTEND_URL` | No | `https://odunolazainab.github.io` | Comma-separated allowed origins |
| `REQUEST_TIMEOUT` | No | `60` | Maximum chat request duration in seconds |

Do not commit `.env` files or API keys. Configure secrets through the deployment platform.

# Deployment

For Hugging Face Spaces, use a Docker Space or another ASGI-capable deployment. A Gradio Space configuration is not appropriate for this FastAPI entry point. Configure the platform to expose port `7860`, provide `GEMINI_API_KEY`, and persist the generated FAISS files if possible.

```text
User -> FastAPI -> safety check -> FAISS retrieval -> Gemini -> SSE response
```

The responses provide financial education only and should not be treated as personalized investment advice.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
