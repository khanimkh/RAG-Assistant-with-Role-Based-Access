from sentence_transformers import CrossEncoder


class RerankerService:
    """
    Phase-2 improvement on top of baseline vector search. Vector similarity
    embeds the query and each chunk SEPARATELY, so it's fast but can't
    reason about how the two relate. A cross-encoder takes the (query,
    chunk) PAIR together and scores relevance directly, more accurate, but
    too slow to run over the whole collection. So: vector search gets a
    wide net of candidates, the cross-encoder re-scores just those.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    # ---------------------------------------------------------------------
    # STEP: rerank(query, chunks, top_n)
    # ALGORITHM:
    #   1. if there are no chunks, return as-is (nothing to rerank)
    #   2. build a (query, chunk_text) pair for every chunk
    #   3. score every pair in one batch call to the cross-encoder model
    #   4. attach each chunk's score onto it as "rerank_score"
    #   5. sort all chunks by that score, descending
    #   6. return the top_n
    # ---------------------------------------------------------------------
    def rerank(self, query, chunks, top_n=5):
        if not chunks:
            return chunks

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_n]
