import os
import stat

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from configs.settings import settings
from src.schemas.request import QuestionRequest
from src.schemas.response import AnswerResponse
from src.schemas.auth import LoginRequest, TokenResponse, AccessLevelUpdate
from src.services.vector_store_service import VectorStoreService
from src.services.llm_service import LLMService
from src.services.reranker_service import RerankerService
from src.services.keyword_search_service import KeywordSearchService
from src.services.response_cache import ResponseCache
from src.services.document_registry_service import DocumentRegistryService
from src.services import auth_service
from src.agents.rag_agent import RAGAgent
from src.agents.security_agent import SecurityAgent
from src.agents.router_agent import RouterAgent
from src.tools.document_loader import SUPPORTED_EXTENSIONS
from src.tools.metadata_filter import ACCESS_LEVELS
from src.utils.hashing import calculate_file_hash
from src.workflows.ingestion_workflow import IngestionWorkflow
from src.workflows.question_workflow import QuestionWorkflow
from src.api.dependencies import get_current_user, require_admin
from src.utils.logger import get_logger

DOCS_DIR = "data/docs"

# ============================================================================
# FILE OVERVIEW: routes.py
# ============================================================================
# PURPOSE
#   Owns two things main.py deliberately doesn't: building every dependency
#   the app needs, and translating whatever goes wrong inside a workflow
#   into a clean HTTP error instead of a raw traceback reaching the client.
#
# GENERAL ALGORITHM
#   1. build every dependency, in the order each one needs the last:
#        1a. VectorStoreService  -> src/services/vector_store_service.py
#        1b. LLMService          -> src/services/llm_service.py
#        1c. RerankerService (optional, config-gated) and
#            KeywordSearchService -> src/services/*.py
#        1d. SecurityAgent       -> src/agents/security_agent.py
#        1e. RAGAgent (needs 1a-1c)        -> src/agents/rag_agent.py
#        1f. RouterAgent (needs 1b, 1e)    -> src/agents/router_agent.py
#        1g. IngestionWorkflow (needs 1a, 1c, and the document registry)
#            -> src/workflows/ingestion_workflow.py
#        1h. ResponseCache      -> src/services/response_cache.py
#        1i. QuestionWorkflow (needs 1d, 1f, 1h) -> src/workflows/question_workflow.py
#        1j. DocumentRegistryService (SQLite - filename, access_level,
#            uploaded_by, uploaded_at, file_hash) -> src/services/document_registry_service.py.
#            Replaces the stage-1 data/access_config.yaml.
#      This mirrors the dependency order in rag_agent.py's own file
#      overview and docs/architecture.md - nothing here is new logic, this
#      is just where the wiring happens.
#   2. define each endpoint as a thin wrapper:
#        a. call the one workflow/service method that does the real work
#        b. on success, return its result (FastAPI serializes it)
#        c. on any exception, log it (get_logger) and raise HTTPException
#           with a generic message - the client never sees a Python
#           traceback or internal file paths
#   3. endpoints: POST /login (issue a token), POST /upload (admin-only:
#      save a file + ingest it), POST /ingest (admin-only: re-sync
#      data/docs/), POST /ask (any logged-in user), GET /documents,
#      DELETE /documents/{filename}, PATCH /documents/{filename}/access
#      (all three admin-only)
#   4. auth: /upload, /ingest, and all /documents* endpoints depend on
#      require_admin; /ask depends on get_current_user - both from
#      src/api/dependencies.py. The role RAGAgent sees now comes from the
#      verified JWT's "role" claim, never from anything the client puts in
#      a request body.
#   5. export `router` (an APIRouter) for main.py to mount with
#      app.include_router(router) - main.py never sees a route body
#
# WHO CALLS THIS FILE
#   main.py imports `router` from here and mounts it. Nothing else does -
#   this file is the HTTP boundary, the last layer before a request reaches
#   question_workflow / ingestion_workflow.
# ============================================================================

logger = get_logger(__name__)

router = APIRouter()

vector_store = VectorStoreService(
    persist_dir=settings.vector_dir,
    collection_name="energy_docs_cosine_v1",
    embedding_model=settings.embedding_model,
    api_key=settings.openai_api_key,
    embedding_batch_size=settings.embedding_batch_size,
    embedding_max_concurrent_batches=settings.embedding_max_concurrent_batches,
)

llm_service = LLMService(
    api_key=settings.openai_api_key,
    model_name=settings.llm_model,
    prompt_path=f"configs/prompts/rag_{settings.prompt_version}.yaml",
)

# Off by default - see docs/future_improvements.md for why. Flip
# RERANKER_ENABLED=true in .env to turn it on.
reranker = RerankerService() if settings.reranker_enabled else None

keyword_search = KeywordSearchService()
security_agent = SecurityAgent()

agent = RAGAgent(
    vector_store,
    llm_service,
    keyword_search=keyword_search,
    reranker=reranker,
    top_k=settings.top_k,
    rerank_candidates=settings.rerank_top_k,
)

# Only "rag" is registered today. A future structured-data route
# (e.g. "sql", answered via Athena) gets added here as one more entry -
# router_agent, question_workflow, and this file's endpoints don't change.
router_agent = RouterAgent(llm_service, routes={"rag": agent})

document_registry = DocumentRegistryService(db_path="data/documents.db")

ingestion = IngestionWorkflow(
    vector_store,
    docs_dir=DOCS_DIR,
    document_registry=document_registry,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    keyword_search=keyword_search,
)

response_cache = ResponseCache(ttl_seconds=3600)
question_workflow = QuestionWorkflow(security_agent, router_agent, response_cache)


# -------------------------------------------------------------------------
# STEP: POST /login
# ALGORITHM:
#   1. CALLS auth_service.authenticate(username, password)
#      -> src/services/auth_service.py
#   2. if it returns None (no such user, or wrong password) -> raise
#      HTTPException(401) with a generic message - never say WHICH part
#      was wrong, that leaks which usernames exist
#   3. if it succeeds -> CALLS auth_service.create_access_token(username,
#      role) and return it as a TokenResponse
#   No error-logging try/except here, deliberately: a failed login is an
#   expected, routine outcome (typo'd password), not a system error worth
#   logging at ERROR level the way an ingestion crash is.
# -------------------------------------------------------------------------
@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    user = auth_service.authenticate(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = auth_service.create_access_token(user["username"], user["role"])
    return TokenResponse(access_token=token, role=user["role"])


# -------------------------------------------------------------------------
# STEP: POST /upload
# ALGORITHM:
#   1. reject the file if its extension isn't one document_loader can
#      actually read (SUPPORTED_EXTENSIONS, imported from document_loader.py
#      rather than duplicated here - one definition of "what we support")
#   2. sanitize the filename to just its base name (os.path.basename) -
#      blocks a filename like "../../etc/passwd" from writing outside
#      data/docs/ via a crafted path
#   3. write the uploaded bytes into data/docs/ under that sanitized name
#   4. CALLS ingestion.run() -> IngestionWorkflow, so the file is searchable
#      immediately, not just sitting on disk until someone remembers to hit
#      /ingest separately
#   5. return the ingestion stats
#
#   DELIBERATELY MISSING: no access_level field anywhere in this request,
#   even though it's now admin-only. Every upload lands at
#   DocumentRegistryService's DEFAULT_ACCESS_LEVEL ("general") - setting a
#   document's access level is PATCH /documents/{filename}/access's job,
#   kept as a separate endpoint/step rather than a field here, so upload
#   and access-granting are two distinct, individually auditable actions.
# -------------------------------------------------------------------------
@router.post("/upload")
async def upload(file: UploadFile = File(...), current_user: dict = Depends(require_admin)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    safe_name = os.path.basename(file.filename)
    os.makedirs(DOCS_DIR, exist_ok=True)
    dest_path = os.path.join(DOCS_DIR, safe_name)

    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)

        file_hash = calculate_file_hash(dest_path)
        document_registry.record_upload(safe_name, current_user["username"], file_hash)

        logger.info(f"Uploaded {safe_name!r} ({len(contents)} bytes) by {current_user['username']!r}, running ingestion.")
        stats = ingestion.run()
        response_cache.clear()
        return stats

    except Exception as e:
        logger.error(f"Upload failed for {file.filename!r}: {e}")
        raise HTTPException(status_code=500, detail="Upload failed.")


# -------------------------------------------------------------------------
# STEP: POST /ingest
# ALGORITHM:
#   1. CALLS ingestion.run() -> IngestionWorkflow (built above)
#   2. on success -> return its stats dict as-is
#   3. on any exception -> log it, raise HTTPException(500) with a generic
#      message (the real error goes to the log, not to the client)
#   Kept as its own endpoint (not just folded into /upload) so an admin can
#   re-sync data/docs/ after manually adding/removing/editing files on
#   disk, without needing to upload anything through the browser.
# -------------------------------------------------------------------------
@router.post("/ingest")
def ingest(current_user: dict = Depends(require_admin)):
    try:
        stats = ingestion.run()
        response_cache.clear()
        return stats
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed.")


# -------------------------------------------------------------------------
# STEP: POST /ask
# ALGORITHM:
#   1. requires a valid token (Depends(get_current_user)) - role comes
#      from current_user["role"], decoded from the verified JWT, NOT from
#      anything in the request body. QuestionRequest has no user_role
#      field for exactly this reason (see schemas/request.py).
#   2. CALLS question_workflow.run(question, role, top_k)
#      -> QuestionWorkflow (built above) -> (answer_text, sources, latency)
#   3. on success -> wrap the three values into an AnswerResponse and
#      return it
#   4. on any exception -> log it (including which question failed), raise
#      HTTPException(500) with a generic message
# -------------------------------------------------------------------------
@router.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    try:
        answer_text, sources, latency = question_workflow.run(
            request.question, current_user["role"], request.top_k
        )
        return AnswerResponse(answer=answer_text, sources=sources, latency=latency)
    except Exception as e:
        logger.error(f"Ask failed for question={request.question!r}: {e}")
        raise HTTPException(status_code=500, detail="Failed to answer question.")


# -------------------------------------------------------------------------
# STEP: GET /documents
# ALGORITHM:
#   1. CALLS document_registry.list_documents()
#      -> src/services/document_registry_service.py
#   2. return the list as-is - each entry is {filename, access_level,
#      uploaded_by, uploaded_at, file_hash}
#   Admin-only: the list reveals which documents exist at which access
#   level, which is exactly the kind of thing an unauthenticated or
#   non-admin caller shouldn't be able to enumerate.
# -------------------------------------------------------------------------
@router.get("/documents")
def list_documents(current_user: dict = Depends(require_admin)):
    return document_registry.list_documents()


# -------------------------------------------------------------------------
# STEP: DELETE /documents/{filename}
# ALGORITHM:
#   1. derive document_id from the filename (strip extension - same
#      convention document_loader.py and ingestion_workflow.py already use)
#   2. CALLS vector_store.delete_document(document_id) - removes its
#      chunks so answers stop citing it immediately
#   3. delete the file from data/docs/ on disk, if it's still there
#   4. CALLS document_registry.delete(filename) - removes its registry row
#   5. return a confirmation
#   NOT updated here: the ingestion manifest (ingestion_workflow's own
#   file, keyed by hash). The next /ingest run naturally reconciles this -
#   it'll see the file missing from data/docs/ and treat it as newly
#   "removed" (calling delete_document again, which is a harmless no-op on
#   an already-empty result set). Not worth coupling this endpoint to
#   ingestion_workflow's internal manifest format just to skip one no-op
#   call on the next run.
# -------------------------------------------------------------------------
@router.delete("/documents/{filename}")
def delete_document(filename: str, current_user: dict = Depends(require_admin)):
    path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(path):
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            os.remove(path)
        except PermissionError:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete {filename}; close the file in any other application and try again.",
            )

    document_id = os.path.splitext(filename)[0]
    vector_store.delete_document(document_id)

    document_registry.delete(filename)
    # Cached answers may reference the deleted document.
    response_cache.clear()
    logger.info(f"Deleted document {filename!r} (by {current_user['username']!r}).")

    return {"status": "deleted", "filename": filename}


# -------------------------------------------------------------------------
# STEP: PATCH /documents/{filename}/access
# ALGORITHM:
#   1. validate the requested access_level is one of ACCESS_LEVELS
#      (imported from src/tools/metadata_filter.py - one definition of
#      what a valid level is, not duplicated here) - reject with 400
#      otherwise
#   2. derive document_id from the filename, same convention as above
#   3. CALLS vector_store.update_access_level(document_id, new_level)
#      -> src/services/vector_store_service.py - patches every existing
#      chunk's stored metadata in place, no re-embedding
#   4. CALLS document_registry.set_access_level(filename, new_level)
#      -> src/services/document_registry_service.py - if this returns
#      False (no registry row for this filename), the file was never
#      ingested; raise 404 rather than silently succeeding
#   5. return a confirmation including how many chunks were updated
# -------------------------------------------------------------------------
@router.patch("/documents/{filename}/access")
def update_document_access(
    filename: str, request: AccessLevelUpdate, current_user: dict = Depends(require_admin)
):
    if request.access_level not in ACCESS_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"access_level must be one of {ACCESS_LEVELS}",
        )

    document_id = os.path.splitext(filename)[0]
    chunks_updated = vector_store.update_access_level(document_id, request.access_level)

    found = document_registry.set_access_level(filename, request.access_level)
    if not found:
        raise HTTPException(status_code=404, detail=f"No such document: {filename}")
    # Prevent answers cached under the previous permissions from being reused.
    response_cache.clear()
 
    logger.info(
        f"Access level for {filename!r} set to {request.access_level!r} "
        f"by {current_user['username']!r} ({chunks_updated} chunks updated)."
    )
    return {"filename": filename, "access_level": request.access_level, "chunks_updated": chunks_updated}
