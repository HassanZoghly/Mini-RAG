# Mini RAG

A modular Retrieval-Augmented Generation (RAG) app with:
- FastAPI backend for upload, processing, indexing, and Q&A
- PostgreSQL + PGVector/Qdrant support
- Streamlit web UI for PDF ingestion and chat

## 1) Environment Setup

1. Create and activate environment:
```bash
conda create -n mini-rag python=3.8 -y
conda activate mini-rag
```

2. Install dependencies:
```bash
cd src
pip install -r requirements.txt
```

3. Create backend env file:
```bash
cp .env.example .env
```

4. Fill `.env` with DB + model provider credentials.

## 2) Start Infrastructure (Docker)

From repository root:
```bash
cd docker
cp env/.env.example.postgres env/.env.postgres
cp env/.env.example.app env/.env.app
# optional: cp env/.env.example.grafana env/.env.grafana
# optional: cp env/.env.example.postgres-exporter env/.env.postgres-exporter
```

Update the copied env files, then start services:
```bash
docker compose up -d
```

## 3) Run Database Migrations

From `src/models/db_schemes/minirag`:
```bash
cp alembic.ini.example alembic.ini
```

Update `alembic.ini` (`sqlalchemy.url`), then run:
```bash
alembic upgrade head
```

## 4) Run FastAPI + Streamlit Concurrently

Open two terminals from `src`.

Terminal 1 (FastAPI):
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

Terminal 2 (Streamlit UI):
```bash
streamlit run ui/app.py --server.port 8501
```

Or run both with one command from repository root:
```bash
chmod +x run.sh
./run.sh
```

Optional: run migrations before startup in the same command:
```bash
RUN_MIGRATIONS=1 ./run.sh
```

## 5) Use the App

1. Open Streamlit: `http://localhost:8501`
2. In sidebar:
   - Set FastAPI URL (default `http://localhost:5000`)
   - Upload PDF
   - Click `Process + Index`
3. In main chat, ask questions about your indexed documents.

## API Endpoints Used by the UI

- `POST /v1/data/upload/{project_id}`
- `POST /v1/data/process/{project_id}`
- `POST /v1/nlp/index/push/{project_id}`
- `POST /v1/nlp/index/answer/{project_id}`
