import time


class ResponseCache:
    """
    In-memory cache for full /ask responses. Deliberately simple, a dict
    with timestamps, no external cache server, since the goal is avoiding
    repeat LLM calls for repeated questions within a single running
    process, not building a distributed cache.
    """

    def __init__(self, ttl_seconds=3600):
        self.ttl_seconds = ttl_seconds
        self._store = {}

    # ---------------------------------------------------------------------
    # STEP: _make_key(question, user_role, top_k)
    # ALGORITHM:
    #   1. normalize the question (strip whitespace, lowercase) so trivial
    #      differences like extra spaces or casing don't cause cache misses
    #   2. combine it with user_role and top_k into a tuple - different
    #      role or top_k means a genuinely different request, so it needs
    #      its own cache entry
    # ---------------------------------------------------------------------
    def _make_key(self, question, user_role, top_k):
        return (question.strip().lower(), user_role, top_k)

    # ---------------------------------------------------------------------
    # STEP: get(question, user_role, top_k)
    # ALGORITHM:
    #   1. build the cache key
    #   2. if that key isn't in the store -> miss, return None
    #   3. if it IS in the store, but was stored longer ago than
    #      ttl_seconds -> treat it as stale: delete it and return None
    #      (a miss, not an error)
    #   4. otherwise -> return the stored (answer, sources) value
    # ---------------------------------------------------------------------
    def get(self, question, user_role, top_k):
        key = self._make_key(question, user_role, top_k)
        entry = self._store.get(key)
        if entry is None:
            return None

        stored_at, value = entry
        if time.time() - stored_at > self.ttl_seconds:
            del self._store[key]
            return None

        return value

    # ---------------------------------------------------------------------
    # STEP: set(question, user_role, top_k, value)
    # ALGORITHM:
    #   1. build the cache key
    #   2. store (current timestamp, value) under that key
    # ---------------------------------------------------------------------
    def set(self, question, user_role, top_k, value):
        key = self._make_key(question, user_role, top_k)
        self._store[key] = (time.time(), value)

    def clear(self):
        self._store.clear()
    