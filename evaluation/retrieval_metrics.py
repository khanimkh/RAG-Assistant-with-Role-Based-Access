def precision_at_k(retrieved_sources, relevant_sources, k):
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for s in top_k if s in relevant_sources)
    return hits / len(top_k)


def recall_at_k(retrieved_sources, relevant_sources, k):
    if not relevant_sources:
        return None
    top_k = retrieved_sources[:k]
    hits = sum(1 for s in relevant_sources if s in top_k)
    return hits / len(relevant_sources)


def reciprocal_rank(retrieved_sources, relevant_sources):
    """1 / rank of the first relevant source found, or 0 if none appear."""
    for i, source in enumerate(retrieved_sources, start=1):
        if source in relevant_sources:
            return 1 / i
    return 0.0



