import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.services.auth_service import decode_access_token

# ============================================================================
# FILE OVERVIEW: dependencies.py
# ============================================================================
# PURPOSE
#   FastAPI "dependencies" - functions a route declares with Depends(...),
#   which FastAPI runs before the route body and either passes the result
#   in as an argument, or raises before the route body ever executes. This
#   is where "is this request even allowed to happen" gets decided, kept
#   separate from routes.py's actual endpoint logic.
#
# GENERAL ALGORITHM
#   get_current_user(request):
#     1. FastAPI's HTTPBearer security scheme extracts the
#        "Authorization: Bearer <token>" header automatically - if it's
#        missing entirely, FastAPI itself returns 401 before this
#        function's body even runs
#     2. CALLS decode_access_token(token) -> src/services/auth_service.py
#     3. if it raises (bad signature, expired, malformed) -> catch it here
#        and raise HTTPException(401) with a clean message
#     4. if it succeeds -> return {"username", "role"}, which FastAPI
#        injects into whichever route declared this dependency
#
#   require_admin(current_user):
#     1. depends on get_current_user ABOVE - so a request that fails
#        authentication never even reaches this check
#     2. if current_user["role"] != "admin" -> raise HTTPException(403)
#     3. otherwise -> return current_user unchanged, so a route using
#        require_admin still gets the same {"username", "role"} shape
#
# WHO CALLS THIS FILE
#   src/api/routes.py declares Depends(get_current_user) on /ask, and
#   Depends(require_admin) on /upload and /ingest.
# ============================================================================

bearer_scheme = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        return decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def require_admin(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user
