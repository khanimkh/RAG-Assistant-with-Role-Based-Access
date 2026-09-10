class KeywordSearchService:
    """
    A deliberately simple keyword search, the second half of hybrid
    retrieval, used only when RAGAgent decides a question needs exact-token
    matching (model numbers, acronyms, quoted phrases) rather than pure
    semantic similarity. Not BM25, not TF-IDF, plain word overlap. Good
    enough for a small document set, and simple enough to explain line by
    line.
    """

    def __init__(self):
        self.chunks = []

    # ---------------------------------------------------------------------
    # STEP: build_index(chunks)
    # ALGORITHM:
    #   1. store the given chunks in memory, replacing whatever was there
    #   "Indexing" here just means "keep them around" - the real scoring
    #   work happens per-query in search(), since the collection is small
    #   enough that scanning it each time is cheap.
    # ---------------------------------------------------------------------
    def build_index(self, chunks):
        self.chunks = chunks

    def _tokenize(self, text):
        return set(text.lower().split())

    # ---------------------------------------------------------------------
    # STEP: search(query, k, allowed_levels)
    # ALGORITHM:
    #   1. tokenize the query into a set of lowercase words
    #   2. for each stored chunk:
    #        a. if allowed_levels is given and this chunk's access_level
    #           isn't in it -> skip it (RBAC applies to keyword search too,
    #           not just vector search)
    #        b. tokenize the chunk's text
    #        c. overlap = words present in BOTH the query and the chunk
    #        d. if overlap is zero -> skip it (not a match at all)
    #        e. score = overlap / total words in the chunk (normalizes so
    #           a long chunk doesn't win just by being long and containing
    #           more words by chance)
    #   3. sort the surviving chunks by score, descending
    #   4. return the top k
    # ---------------------------------------------------------------------
    def search(self, query, k=5, allowed_levels=None):
        query_words = self._tokenize(query)
        scored = []

        for chunk in self.chunks:
            if allowed_levels is not None and chunk.get("access_level", "general") not in allowed_levels:
                continue
            chunk_words = self._tokenize(chunk["text"])
            if not chunk_words:
                continue
            overlap = len(query_words & chunk_words)
            if overlap == 0:
                continue
            score = overlap / len(chunk_words)
            scored.append({**chunk, "score": score})

        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:k]
