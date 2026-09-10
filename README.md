# RAG Assistant with Role-Based Access

A Retrieval-Augmented Generation (RAG) assistant for answering questions over financial documents with **role-based access control, hybrid retrieval, citations, and evaluation**.

The application loads PDF documents, splits them into chunks, creates embeddings, retrieves relevant passages, and generates grounded answers with citations.

## Features

- Role-based access control for `general` and `admin` users
- JWT authentication
- PDF upload and ingestion
- Recursive text chunking
- Vector and keyword retrieval
- Metadata-based access filtering
- Grounded answers with citations
- Response caching
- Retrieval and answer-quality evaluation
- MLflow experiment tracking
- FastAPI backend
- Docker Compose


For more details about the architecture, implementation, experiments, and future work, see the [Technical Project Overview](docs/Technical_Project_Overview.md).

## Project Structure

```text
configs/              Application settings and prompts
data/                 Test questions, users, and documents
docs/                 Project documentation
evaluation/           Evaluation metrics and LLM-as-Judge
evaluation_results/   Experiment results
experiments/          Experiment scripts
scripts/              Administration scripts
src/                  FastAPI application and RAG implementation
tests/                Automated tests
```

## Setup

### 1. Configure Environment

Copy `.env.example` to `.env` and configure:

```dotenv
OPENAI_API_KEY=your-openai-api-key
JWT_SECRET_KEY=use-a-long-random-secret
```

Never commit `.env` to Git.

### 2. Create a User

```bash
docker compose build
docker compose run --rm rag-app python scripts/add_user.py
```

Available roles:

- `general` — access to general documents
- `admin` — access to all documents and administrative endpoints

### 3. Run the Application

```bash
docker compose up --build
```

Application:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

MLflow:

```text
http://localhost:5000
```

Stop the services:

```bash
docker compose down
```

## Main API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `POST /login` | User authentication |
| `POST /upload` | Admin document upload |
| `POST /ingest` | Document ingestion |
| `POST /ask` | RAG question answering |
| `/docs` | FastAPI documentation |

## Tests

Tests use mocks and do not require a real OpenAI API key.

```bash
docker compose run --rm rag-app pytest -q tests/
```

## Experiments

Run the RAG experiments with:

```bash
docker compose run --rm rag-app python experiments/run_baseline.py
docker compose run --rm rag-app python experiments/run_chunking_experiments.py
docker compose run --rm rag-app python experiments/run_prompt_experiments.py
docker compose run --rm rag-app python experiments/run_retrieval_experiments.py
```

Experiments evaluate different **chunking, retrieval, and prompt configurations**.

Results are saved in `evaluation_results/`, and experiments are tracked with **MLflow**.

## Evaluation

The project evaluates:

- **Precision@K** — relevance of retrieved chunks
- **Recall@K** — coverage of relevant information
- **MRR** — ranking of the first relevant result
- **Faithfulness** — whether answers are supported by retrieved context
- **Relevance** — whether answers address the question

## Future Improvements

- OpenSearch or Qdrant for scalable vector search
- BM25 + vector hybrid retrieval
- Cross-Encoder or LLM reranking
- Semantic chunking
- Redis caching
- PostgreSQL for persistent metadata
- LangGraph for more complex workflows
- Neo4j for deterministic knowledge-graph retrieval
- Cloud deployment and production monitoring

## Demo Data

The financial documents and test questions included in this repository are **synthetic and created for demonstration purposes**.

## Security

Do not commit local credentials or generated runtime data such as:

```text
.env
data/users.yaml
data/documents.db
vector_store/
mlruns/
```

Use `.gitignore` to keep these files out of the repository.
