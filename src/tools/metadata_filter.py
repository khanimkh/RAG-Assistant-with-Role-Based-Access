# Roles are ranked low to high. Adding a new tier later is one line here,
# not a new hand-written list for every existing role.
ACCESS_LEVELS = ["general", "admin"]


# -------------------------------------------------------------------------
# STEP: allowed_access_levels(user_role)
# ALGORITHM:
#   1. if user_role isn't a recognized level -> treat it as "general"
#      (unknown role gets the LOWEST access, never raises, never guesses
#      higher - a typo in a role name must never accidentally grant more)
#   2. find that role's position (rank) in ACCESS_LEVELS
#   3. return every level from position 0 up to and including that rank
#      (a role sees its own level plus everything below it)
# -------------------------------------------------------------------------
def allowed_access_levels(user_role):
    if user_role not in ACCESS_LEVELS:
        user_role = "general"
    rank = ACCESS_LEVELS.index(user_role)
    return ACCESS_LEVELS[: rank + 1]


# -------------------------------------------------------------------------
# STEP: build_metadata_filter(user_role)
# ALGORITHM:
#   1. get this role's allowed_access_levels()
#   2. wrap them in the shape Chroma's query "where" clause expects:
#      {"access_level": {"$in": [...]}}
#   Applied at retrieval time, before any chunk text reaches the LLM -
#   authorization happens at the database query, not by asking the model
#   to withhold information it already has.
# -------------------------------------------------------------------------
def build_metadata_filter(user_role):
    levels = allowed_access_levels(user_role)
    return {"access_level": {"$in": levels}}


# -------------------------------------------------------------------------
# STEP: is_role_authorized(user_role, chunk_access_level)
# ALGORITHM:
#   1. get this role's allowed_access_levels()
#   2. check whether chunk_access_level is one of them
#   A second, redundant check applied AFTER retrieval (in rag_agent) -
#   defense in depth, so a bug in the DB filter above doesn't silently
#   leak a chunk the filter should have blocked.
# -------------------------------------------------------------------------
def is_role_authorized(user_role, chunk_access_level):
    return chunk_access_level in allowed_access_levels(user_role)
