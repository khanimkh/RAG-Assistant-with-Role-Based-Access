from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.api.routes import router

# ============================================================================
# FILE OVERVIEW: main.py
# ============================================================================
# PURPOSE
#   The application's entry point. Deliberately thin - this file never
#   builds a service, never touches a workflow, and never sees a request
#   handler's body for /upload, /ingest, or /ask. All of that lives in
#   src/api/routes.py.
#
# GENERAL ALGORITHM
#   1. create the FastAPI app instance
#   2. mount every endpoint defined in src/api/routes.py:
#        CALLS app.include_router(router) - this single call brings in
#        /upload, /ingest, and /ask, fully wired, with their own error
#        handling
#   3. serve the static frontend (src/static/index.html) at "/" - read
#      once at request time, returned as raw HTML; no template engine or
#      build step needed for a single static page
#   4. define /health directly here (not in routes.py) - it doesn't touch
#      any dependency, so it doesn't need to live behind that file
#
# WHY THE SPLIT
#   Keeping dependency construction and route bodies out of this file means
#   "what does the app expose" (this file) stays separate from "how is each
#   endpoint implemented" (routes.py) - a reviewer can read this file in
#   ten seconds and know exactly what's mounted, without scrolling through
#   wiring code to find it.
# ============================================================================

app = FastAPI(title="RAG Assistant")

app.include_router(router)

FRONTEND_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return FRONTEND_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}
