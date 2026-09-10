import bcrypt
import jwt
import yaml
from datetime import datetime, timedelta, timezone

from configs.settings import settings

USERS_PATH = "data/users.yaml"

# ============================================================================
# FILE OVERVIEW: auth_service.py
# ============================================================================
# PURPOSE
#   Everything about proving who someone is: checking a password against a
#   stored hash, and issuing/verifying the JWT that stands in for "logged
#   in as this user" on every request after login. Nothing here decides
#   what a role is ALLOWED to do - that's api/dependencies.py (which route
#   requires which role) and src/tools/metadata_filter.py (which document
#   access levels a role can see). This file only answers "who is this."
#
# GENERAL ALGORITHM
#   login flow:
#     1. authenticate(username, password)
#          1a. load_users() -> read data/users.yaml
#          1b. look up username; if missing -> return None (don't reveal
#              whether the username exists vs. the password was wrong -
#              same response either way)
#          1c. verify_password() against the stored bcrypt hash
#          1d. if it matches -> return {"username", "role"}; else None
#     2. create_access_token(username, role) -> a signed JWT containing
#        the username, role, and an expiry timestamp
#
#   per-request flow (used by api/dependencies.py, not this file):
#     3. decode_access_token(token) -> verifies the signature and expiry,
#        returns {"username", "role"} or raises if invalid/expired
#
# ROLE VALUES
#   Deliberately reuse the same strings as src/tools/metadata_filter.py's
#   ACCESS_LEVELS ("general", "admin") - so a role decoded from a
#   JWT can be passed straight into RAGAgent's user_role parameter with no
#   translation step in between.
# ============================================================================


def load_users():
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("users", {}) or {}


# ---------------------------------------------------------------------
# STEP: hash_password(password) / verify_password(password, password_hash)
# ALGORITHM:
#   hash_password:
#     1. encode the password to bytes
#     2. CALLS bcrypt.hashpw() with a freshly generated random salt
#        (bcrypt.gensalt()) - this is why the same password hashed twice
#        produces two different-looking hashes; both still verify
#     3. return the hash as a string, for storing in YAML/a database
#   verify_password:
#     1. encode both the given password and the stored hash to bytes
#     2. CALLS bcrypt.checkpw() - re-hashes the given password with the
#        salt embedded in the stored hash, compares the result
#     3. return True/False
# ---------------------------------------------------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password, password_hash):
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ---------------------------------------------------------------------
# STEP: authenticate(username, password)
# ALGORITHM:
#   1. CALLS load_users() -> read the user store
#   2. look up username; if not found -> return None
#   3. CALLS verify_password(password, stored_hash)
#   4. if it doesn't match -> return None
#   5. if it matches -> return {"username": username, "role": user["role"]}
#   Same None result for "no such user" and "wrong password" - a login
#   endpoint should never let a caller distinguish the two, since that
#   would let someone enumerate valid usernames by trial and error.
# ---------------------------------------------------------------------
def authenticate(username, password):
    users = load_users()
    user = users.get(username)
    if user is None:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {"username": username, "role": user["role"]}


# ---------------------------------------------------------------------
# STEP: create_access_token(username, role)
# ALGORITHM:
#   1. compute an expiry timestamp: now + settings.jwt_expiry_minutes
#   2. build the payload: {"sub": username, "role": role, "exp": expiry}
#      ("sub" = subject, the standard JWT claim name for "who this is about")
#   3. CALLS jwt.encode(payload, secret_key, algorithm) -> signs the
#      payload with settings.jwt_secret_key, returns the token string
#   The signature is what makes this trustworthy: anyone can READ a JWT's
#   contents (it's not encrypted, just base64), but only someone with
#   jwt_secret_key can produce a signature that verifies - so a client
#   can't forge a token claiming role="admin".
# ---------------------------------------------------------------------
def create_access_token(username, role):
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


# ---------------------------------------------------------------------
# STEP: decode_access_token(token)
# ALGORITHM:
#   1. CALLS jwt.decode(token, secret_key, algorithms=[...]) - verifies
#      the signature AND checks "exp" hasn't passed; raises
#      jwt.PyJWTError (or a subclass) if either check fails
#   2. return {"username": payload["sub"], "role": payload["role"]}
#   Raises rather than returning None on failure, so the caller
#   (api/dependencies.py) can't accidentally treat "invalid token" the
#   same as "valid token for a real user" by forgetting a None check.
# ---------------------------------------------------------------------
def decode_access_token(token):
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    return {"username": payload["sub"], "role": payload["role"]}
