import re

from src.tools.metadata_filter import build_metadata_filter, allowed_access_levels, is_role_authorized
#from src.utils.citations import extract_answer_sources, validate_citations
from src.utils.citations import extract_sources
from src.services.llm_service import NO_ANSWER_MESSAGE

GROUNDING_THRESHOLD = 0.25
EXACT_MATCH_SIGNALS = re.compile(r'"[^"]+"|\b[A-Z]{2,}[-]?\d+\b|\b[A-Z]{3,}\b')
ENABLE_HYBRID_SEARCH = True
ENABLE_GROUNDING_CHECK = True


class RAGAgent:
    """Retrieve grounded document context and generate an answer."""

    def __init__(self, vector_store, llm_service,
                 keyword_search=None, reranker=None,
                 top_k=5, rerank_candidates=15):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.keyword_search = keyword_search
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates
    def _looks_like_exact_match_query(self, question):
        """Return whether the question contains an exact-match signal."""
        return bool(EXACT_MATCH_SIGNALS.search(question))

    # ---------------------------------------------------------------------
    # STEP: _retrieve(question, allowed_levels, where, k, use_hybrid)
    # ALGORITHM:
    #   1. always run vector search:
    #      CALLS self.vector_store.similarity_search(question, k, where)
    #      -> injected VectorStoreService (src/services/vector_store_service.py)
    #   2. if use_hybrid is False, or no keyword_search is configured ->
    #      return the vector results as-is, nothing more to do
    #   3. otherwise, also run keyword search:
    #      CALLS self.keyword_search.search(question, k, allowed_levels)
    #      -> injected KeywordSearchService (src/services/keyword_search_service.py)
    #   4. merge the two result lists:
    #        for each chunk (from either list), keep only the
    #        higher-scoring copy if the same chunk_id appears in both
    #   5. sort the merged chunks by score, descending
    #   6. return the top k
    # ---------------------------------------------------------------------
    def _retrieve(self, question, allowed_levels, where, k, use_hybrid):
        vector_chunks = self.vector_store.similarity_search(question, k=k, where=where)

        if not use_hybrid or self.keyword_search is None:
            return vector_chunks

        keyword_chunks = self.keyword_search.search(question, k=k, allowed_levels=allowed_levels)

        merged = {}
        for chunk in vector_chunks + keyword_chunks:
            existing = merged.get(chunk["chunk_id"])
            if existing is None or chunk["score"] > existing["score"]:
                merged[chunk["chunk_id"]] = chunk

        return sorted(merged.values(), key=lambda c: c["score"], reverse=True)[:k]

    # ---------------------------------------------------------------------
    # STEP: _is_weak(chunks)
    # ALGORITHM:
    #   1. if there are no chunks at all -> weak
    #   2. otherwise, compare the top chunk's score to GROUNDING_THRESHOLD
    #      (module-level constant defined at the top of this file)
    #   3. return True if there are no chunks, OR the top score is below
    #      the threshold
    #   No external calls.
    # ---------------------------------------------------------------------
    def _is_weak(self, chunks):
     print(f"DEBUG retrieved chunks: {len(chunks)}", flush=True)

     for index, chunk in enumerate(chunks[:5], start=1):
        print(
            f"DEBUG result {index}: "
            f"score={chunk.get('score')} | "
            f"source={chunk.get('source')} | "
            f"page={chunk.get('page')}",
            flush=True,
        )
        print(
            f"DEBUG text: {chunk.get('text', '')[:500]}",
            flush=True,
        )

     if not chunks:
        print("DEBUG retrieval returned no chunks", flush=True)
        return True

     is_weak = chunks[0]["score"] < GROUNDING_THRESHOLD
     print(
        f"DEBUG top score={chunks[0]['score']} | "
        f"threshold={GROUNDING_THRESHOLD} | "
        f"is_weak={is_weak}",
        flush=True,
     )
     return is_weak
    #  ---------------------------------------------------------------------
    # STEP: answer(question, user_role, top_k)
    # ALGORITHM:
    #   1. work out access:
    #        CALLS allowed_access_levels(user_role) and
    #        build_metadata_filter(user_role) -> src/tools/metadata_filter.py
    #   2. decide retrieval strategy:
    #        CALLS self._looks_like_exact_match_query(question) -> this file
    #   3. retrieve (ask for rerank_candidates up front instead of top_k if
    #      a reranker is configured, so we don't retrieve twice):
    #        CALLS self._retrieve(...) -> this file
    #        if CALLS self._is_weak(chunks) -> this file -> True, and we
    #        hadn't already tried hybrid -> CALLS self._retrieve(...) again
    #        with use_hybrid=True (the actual "agent" retry behavior)
    #   4. re-check access on whatever came back:
    #        CALLS is_role_authorized(user_role, level) per chunk
    #        -> src/tools/metadata_filter.py
    #   5. bail out if still weak:
    #        CALLS self._is_weak(chunks) -> this file; if True, return
    #        NO_ANSWER_MESSAGE (constant from src/services/llm_service.py)
    #        and skip step 6 - no LLM call made
    #   6. generate + cite:
    #        if a reranker is configured, CALLS self.reranker.rerank(...)
    #        -> injected RerankerService (src/services/reranker_service.py)
    #        CALLS self.llm_service.generate_answer(question, chunks)
    #        -> injected LLMService (src/services/llm_service.py)
    #        CALLS extract_sources(chunks) -> src/utils/citations.py
    #   7. return (answer_text, sources, chunks)
    # ---------------------------------------------------------------------
    def answer(self, question, user_role="general", top_k=None):
        k = top_k or self.top_k
        allowed_levels = allowed_access_levels(user_role)
        where = build_metadata_filter(user_role)

        # --- IMPROVEMENT 1: hybrid search decision (see toggle above) ---
        use_hybrid = ENABLE_HYBRID_SEARCH and self._looks_like_exact_match_query(question)
        # --- end IMPROVEMENT 1 ---

        retrieve_k = self.rerank_candidates if self.reranker else k
        chunks = self._retrieve(question, allowed_levels, where, retrieve_k, use_hybrid)

        # --- IMPROVEMENT 1: retry with hybrid if the first attempt is weak ---
        if ENABLE_HYBRID_SEARCH and self._is_weak(chunks) and not use_hybrid:
            chunks = self._retrieve(question, allowed_levels, where, retrieve_k, use_hybrid=True)
        # --- end IMPROVEMENT 1 ---

        chunks = [c for c in chunks if is_role_authorized(user_role, c.get("access_level", "general"))]

        # --- IMPROVEMENT 2: grounding threshold refusal (see toggle above) ---
        if ENABLE_GROUNDING_CHECK and self._is_weak(chunks):
            return NO_ANSWER_MESSAGE, [], chunks
        # --- end IMPROVEMENT 2 ---

        if self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_n=k)

        answer_text = self.llm_service.generate_answer(question, chunks)
        # Do not display retrieved but irrelevant sources for a no-answer response.
        if self.normalize(NO_ANSWER_MESSAGE) in self.normalize(answer_text):
          return NO_ANSWER_MESSAGE, [], chunks
        #claimed_sources = extract_answer_sources(answer_text)
        #sources = validate_citations(claimed_sources, chunks)
        sources = extract_sources(chunks)

        return answer_text, sources, chunks

    def normalize(self, text):
         return " ".join(text.strip().lower().split()).rstrip(".")
