# RAG Assistant: Technical Project Overview

## 1. Purpose and Scope

This project is a production-oriented Retrieval-Augmented Generation (RAG) assistant for answering questions over energy-industry technical documents. It combines document ingestion, access-controlled retrieval, LLM-based answer generation, citation validation, and offline evaluation in one application.

The system is designed around two user-facing capabilities:

- **Question answering:** an authenticated user asks a natural-language question and receives an answer grounded in retrieved document passages.
- **Document administration:** an administrator uploads, ingests, lists, deletes, and changes the access level of documents.

The application is implemented as a FastAPI service with a lightweight browser client. ChromaDB stores vectorized document chunks, SQLite stores document-management metadata, and OpenAI services provide embeddings and answer generation.

The architecture is shown in [00-architecture.png](00-architecture.png). The companion workflow and sequence diagrams are available in the `docs` directory.

## 2. Architectural View

The system is organized into the following layers:

1. **Application and API layer** - browser or API clients communicate with FastAPI over HTTP.
2. **Authentication and user-management layer** - JWT authentication and administrator authorization protect endpoints.
3. **Question orchestration layer** - a workflow applies caching and security checks, selects an answering agent, and validates the result.
4. **Ingestion and document-processing layer** - files are hashed, loaded, split into chunks, enriched with metadata, embedded, and indexed.
5. **Retrieval and generation layer** - vector, keyword, and optional reranker techniques select context for the LLM.
6. **Persistence layer** - ChromaDB, SQLite, a JSON manifest, YAML configuration, and process-local caches hold application state.
7. **Configuration and prompt layer** - typed settings and versioned YAML prompts control runtime behavior.
8. **Evaluation layer** - automated metrics, LLM judges, experiments, and MLflow track quality and performance.

A useful design principle in the code is the separation between **workflows** and **agents**. A workflow owns a predictable sequence of operations. An agent makes a decision based on an observed input or result. For example, ingestion is a workflow, while the RAG agent can choose whether a query needs hybrid retrieval.

## 3. Application and API Layer

### Main technologies

- FastAPI and Uvicorn
- Pydantic request and response schemas
- Static HTML and JavaScript browser client
- Docker and Docker Compose for deployment

[src/main.py](../src/main.py) creates the FastAPI application, mounts the API router, serves the browser interface, and exposes a health endpoint. [src/api/routes.py](../src/api/routes.py) defines the main HTTP endpoints:

| Endpoint | Purpose | Authorization |
|---|---|---|
| `POST /login` | Authenticate a user and issue a JWT | Public |
| `POST /ask` | Ask a question | Authenticated user |
| `POST /upload` | Upload a document | Administrator |
| `POST /ingest` | Run document ingestion | Administrator |
| `GET /documents` | List registered documents | Administrator |
| `DELETE /documents/{filename}` | Delete a document | Administrator |
| `PATCH /documents/{filename}/access` | Change document access | Administrator |

Request and response structures are defined in [src/schemas](../src/schemas). This keeps validation at the API boundary and prevents route handlers from receiving unstructured input.

## 4. Authentication and Role-Based Access Control

### Techniques used

- Password hashing with bcrypt
- Signed JSON Web Tokens (JWT)
- Bearer-token dependency injection in FastAPI
- Role-based endpoint authorization
- Retrieval-time metadata filtering
- Defense-in-depth authorization after retrieval

[src/services/auth_service.py](../src/services/auth_service.py) authenticates users from `data/users.yaml`. Passwords are stored as bcrypt hashes rather than plaintext. Successful login produces a signed JWT containing the user identity, role, and expiration time.

[src/api/dependencies.py](../src/api/dependencies.py) extracts and verifies the bearer token. Administrator-only operations use `require_admin`, while `/ask` accepts any authenticated user.

RBAC is applied before document text reaches the LLM. The user's role is mapped to allowed document access levels by [src/tools/metadata_filter.py](../src/tools/metadata_filter.py). The vector query applies those levels as a Chroma metadata filter, and the returned chunks are checked again before generation. This is stronger than asking the LLM to hide restricted information because unauthorized text is excluded from its context window.

The current implementation supports `general` and `admin` access behavior. Older project notes mention a `power` level, but that level is not currently represented in the active authorization mapping and should not be treated as implemented.

## 5. Question-Answering Flow

A question follows this sequence:

1. The client sends `POST /ask` with a bearer token.
2. The API dependency validates the JWT and obtains the verified user role.
3. [src/workflows/question_workflow.py](../src/workflows/question_workflow.py) checks the response cache.
4. [src/agents/security_agent.py](../src/agents/security_agent.py) checks for suspicious or prompt-injection-style questions.
5. [src/agents/router_agent.py](../src/agents/router_agent.py) selects an answering route.
6. The registered `rag` route dispatches to [src/agents/rag_agent.py](../src/agents/rag_agent.py).
7. The RAG agent retrieves authorized context, applies grounding and optional reranking, and calls the LLM.
8. Citations are filtered against retrieved metadata.
9. The answer and sources are returned and cached.

The router is intentionally extensible, but only the `rag` route is registered today. Consequently, the router can dispatch directly without spending an LLM call on a choice between multiple routes.

## 6. Ingestion and Document-Processing Layer

The ingestion path is implemented by [src/workflows/ingestion_workflow.py](../src/workflows/ingestion_workflow.py):

1. Scan `data/docs` for supported files.
2. Calculate a content hash for each file.
3. Compare hashes with `vector_store/ingestion_manifest.json`.
4. Skip unchanged documents and remove stale chunks for changed or deleted documents.
5. Register document metadata in SQLite.
6. Load PDF, Markdown, or text content through [src/tools/document_loader.py](../src/tools/document_loader.py).
7. Split content using [src/tools/chunker.py](../src/tools/chunker.py), based on LangChain's `RecursiveCharacterTextSplitter`.
8. Attach source, page, chunk, and access-level metadata.
9. Generate embeddings and write chunks to ChromaDB.
10. Rebuild the in-memory keyword index.
11. Save ingestion statistics and the updated manifest.

The ingestion hash is an incremental-processing technique: unchanged files do not need to be embedded again. Chunk identifiers preserve document and page identity, which supports reproducible retrieval and citation construction.

PDF pages with no extractable text are currently skipped. Markdown and plain-text files are treated as page `1`, so their citations cannot identify a more precise page location.

## 7. Retrieval and Generation Layer

### 7.1 Vector retrieval

[src/services/vector_store_service.py](../src/services/vector_store_service.py) uses OpenAI embeddings and persistent ChromaDB storage. The service queries by semantic similarity and converts Chroma distance into a similarity-style score using `1 - distance`. The access-level filter is applied inside the vector-store query.

Vector retrieval is effective for conceptual questions where related wording may differ between the question and the document.

### 7.2 Keyword retrieval and hybrid search

[src/services/keyword_search_service.py](../src/services/keyword_search_service.py) provides a lightweight keyword index using word-overlap scoring. The RAG agent uses keyword retrieval when a question contains signals such as quoted phrases, acronyms, model-number-like tokens, or weak vector results.

Hybrid retrieval merges or compares the semantic and lexical results. This improves exact-token matching, where embeddings alone may rank a conceptually similar passage above the passage containing the exact identifier.

The keyword implementation is intentionally simple and is not yet a production BM25 or full-text search engine.

### 7.3 Reranking

[src/services/reranker_service.py](../src/services/reranker_service.py) optionally applies the `cross-encoder/ms-marco-MiniLM-L-6-v2` model to a wider candidate set. A cross-encoder scores the question and candidate passage together, which can improve final ordering after initial retrieval.

Reranking is configurable and disabled by default. This keeps the baseline inexpensive and avoids adding model-download and inference cost unless evaluation shows a benefit.

### 7.4 Grounding and answer generation

The RAG agent refuses before calling the LLM when no authorized chunks are found or when the best retrieval score is below the configured grounding threshold. This provides a deterministic no-answer path instead of relying only on a prompt instruction.

[src/services/llm_service.py](../src/services/llm_service.py) loads a versioned YAML prompt, formats the retrieved context with source and page metadata, and calls the configured OpenAI chat model at temperature `0`. The selected `rag_v4` prompt instructs the model to:

- answer only from the supplied context;
- resist instructions embedded inside retrieved documents;
- decompose multi-part questions;
- answer supported parts while identifying unsupported parts;
- avoid treating related terms as equivalent without evidence; and
- verify that cited facts are present in the cited source.

This combines deterministic application checks with model-level grounding instructions.

## 8. Citation Validation and Response Caching

[src/utils/citations.py](../src/utils/citations.py) builds citations from retrieved metadata in the form `{source}, p.{page}`. The workflow does not blindly trust source names produced by the model: [src/agents/security_agent.py](../src/agents/security_agent.py) filters citations against the set of chunks actually retrieved for the request.

[src/services/response_cache.py](../src/services/response_cache.py) provides an in-memory TTL cache. Its key includes the normalized question, user role, and `top_k`, so a response generated for one role is not reused for another role. Upload, deletion, ingestion, and access-level changes clear the cache because those operations can change the available evidence.

The ingestion hash cache and response cache solve different problems:

- the ingestion manifest avoids repeated embedding work;
- the response cache avoids repeated retrieval and generation for repeated questions.

Both the response cache and keyword index are process-local. They are not shared automatically between multiple workers or application instances.

## 9. Persistence and External Services

| Component | Technology | Responsibility |
|---|---|---|
| Vector store | ChromaDB | Embeddings, chunk text, and retrieval metadata |
| Document registry | SQLite | Filename, access level, uploader, timestamp, and content hash |
| Ingestion manifest | JSON | File hashes and ingestion state |
| User registry | YAML | Usernames, bcrypt hashes, and roles |
| LLM and embeddings | OpenAI API | Embedding and answer-generation models |
| Optional reranker | Sentence Transformers | Cross-encoder candidate reranking |
| Experiment tracking | MLflow | Parameters, metrics, and artifacts |
| Runtime packaging | Docker Compose | Application and persistent volume setup |

[compose.yaml](../compose.yaml) mounts application data, the vector store, MLflow data, and evaluation results so state can survive container replacement. The `Dockerfile` packages the Python application and runs it through Uvicorn.

## 10. Configuration and Prompt Management

[configs/settings.py](../configs/settings.py) defines typed Pydantic settings loaded from environment variables and `.env`. Important parameters include:

- OpenAI API and model settings;
- ChromaDB location;
- chunk size and overlap;
- embedding batch and concurrency settings;
- retrieval `top_k`;
- reranking and hybrid-search switches;
- grounding threshold;
- JWT configuration; and
- prompt version.

Prompts are versioned under `configs/prompts`. This makes prompt changes experimentally comparable and allows the application to reproduce a selected configuration without embedding prompt text directly in Python code.

One operational detail should be resolved before deployment: the default prompt version in settings and the prompt version shown in the example environment configuration must be kept consistent, because an actual `.env` value takes precedence over the code default.

## 11. Evaluation and Experiments

The evaluation layer measures both retrieval and answer behavior:

- [evaluation/retrieval_metrics.py](../evaluation/retrieval_metrics.py) calculates precision@k, recall@k, and reciprocal rank.
- [evaluation/judges.py](../evaluation/judges.py) uses an LLM judge for faithfulness and relevance.
- [evaluation/evaluate_rag.py](../evaluation/evaluate_rag.py) measures answer quality, source retrieval, no-answer accuracy, and latency.
- The experiment scripts isolate baseline, chunking, retrieval, and prompt variables.
- MLflow records experiment parameters, metrics, and artifacts.

The existing experiments show that prompt design is particularly important for partial and unanswerable questions. The v4 prompt improved no-answer behavior by combining partial-answer decomposition with citation verification. The existing results also indicate that more context is not always better: a larger `top_k` can increase latency and introduce less-relevant passages.

Evaluation is therefore treated as part of the product architecture rather than as a one-time development activity. Changes to chunking, retrieval, prompts, or reranking should be compared against the same question set and metrics.

## 12. Testing Strategy

The tests cover the most important internal contracts:

- chunk boundaries and metadata preservation;
- citation formatting and filtering;
- grounding and refusal behavior;
- question workflow order, security rejection, caching, and citation validation;
- RBAC access-level rules; and
- retrieval, top-k, reranking, and hybrid-search behavior.

The main remaining testing gap is API-level integration coverage. The project should add tests that exercise the actual HTTP endpoints with JWTs, including login, protected routes, upload and ingestion lifecycle, document deletion, and access-level changes.

## 13. Future Work

### Security and identity

- Add login rate limiting, account lockout or abuse monitoring, and JWT revocation.
- Move browser authentication to a refresh-token design with an `httpOnly`, `Secure`, and appropriately scoped cookie.
- Apply the same filename sanitization and path validation to document deletion that is applied during upload.
- Add audit logging for document uploads, access-level changes, deletions, and administrative actions.

### Retrieval quality

- Replace word-overlap keyword scoring with BM25 or a persistent full-text index.
- Evaluate hybrid score calibration and retrieval fusion on a larger, more diverse question set.
- Tune the reranker using domain-specific evaluation data and measure its latency/cost trade-off.
- Add OCR or an image/document understanding path for scanned PDFs.
- Preserve section and page structure more accurately for Markdown, text, and complex PDFs.

### Data and orchestration

- Add a second structured-data agent for CSV, spreadsheet, or SQL questions and connect it to `RouterAgent`.
- Add query rewriting, multi-query retrieval, or conversation history only when evaluation demonstrates a measurable benefit.
- Add a lock or job queue around ingestion so simultaneous administrator requests cannot race.
- Move the keyword index and response cache to shared infrastructure when deploying multiple workers.

### Reliability and operations

- Add API integration tests and end-to-end tests for the browser workflow.
- Add structured logs, request identifiers, metrics, and tracing across retrieval and generation.
- Add health checks for ChromaDB, SQLite, and external model services.
- Define retry, timeout, and circuit-breaker behavior for OpenAI and reranker calls.
- Add document versioning and rollback for ingestion failures.

### Evaluation and governance

- Expand the test set with adversarial, multi-part, ambiguous, and access-control questions.
- Track cost, token usage, cache hit rate, and retrieval latency alongside answer metrics.
- Add regression gates so prompt or retrieval changes cannot reduce grounding, citation correctness, or RBAC isolation.
- Maintain separate evaluation sets for general-access and administrator-only documents.

## 14. Summary

The project implements a layered RAG system in which security, retrieval, generation, and evaluation are explicit components. Its strongest architectural choices are retrieval-time RBAC, deterministic grounding checks, hybrid retrieval for exact terms, citation validation based on retrieved metadata, incremental ingestion through content hashes, and repeatable experiment tracking.

The next stage is production hardening: shared state for multi-worker deployments, stronger authentication lifecycle controls, robust full-text retrieval, ingestion concurrency control, API integration tests, and broader evaluation coverage. These improvements can be added without changing the central separation between API routes, fixed workflows, decision-making agents, retrieval services, and persistence components.
