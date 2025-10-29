# Minimal RAGFlow (Docker)

A tiny Retrieval-Augmented Generation API:
- `POST /ingest/text` - add raw text
- `POST /ingest/file` - upload a `.txt` file (line-split)
- `POST /ask` - ask questions grounded in your data

## 1) Prereqs
- Docker / Docker Compose
- An OpenAI API key

## 2) Run
```bash
export OPENAI_API_KEY=sk-xxxx
docker compose up --build
