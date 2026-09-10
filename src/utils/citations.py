import re

# -------------------------------------------------------------------------
# STEP: build_citation(chunk)
# ALGORITHM:
#   1. take a chunk's "source" and "page" fields
#   2. format them as a single human-readable string: "{source}, p.{page}"
# -------------------------------------------------------------------------
def build_citation(chunk):
    return f"{chunk['source']}, p.{chunk['page']}"


# -------------------------------------------------------------------------
# STEP: extract_sources(chunks)
# ALGORITHM:
#   1. start an empty "seen" set and an empty result list
#   2. for each chunk, in order:
#        a. build its citation string
#        b. if that citation hasn't been added yet -> add it to both the
#           seen set and the result list (keeps first-seen order, drops
#           duplicates - the same document/page can appear from more than
#           one chunk if chunking overlap put the same page in two pieces)
#   3. return the result list
#   Sources come from retrieved chunk metadata only - never from the LLM's
#   own text - so the model has no path to invent a citation.
# -------------------------------------------------------------------------
def extract_sources(chunks):
    seen = set()
    sources = []
    for c in chunks:
        citation = build_citation(c)
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)
    return sources


# -------------------------------------------------------------------------
# STEP: validate_citations(cited_sources, retrieved_chunks)
# ALGORITHM:
#   1. build the set of citations that could legitimately be claimed - one
#      per retrieved chunk
#   2. keep only the entries in cited_sources that are actually in that set
#   3. return the filtered list
#   A cheap backstop: if something downstream claims a source that was
#   never retrieved, it gets silently dropped rather than shown to the user.
# -------------------------------------------------------------------------
def validate_citations(cited_sources, retrieved_chunks):
    retrieved_citations = {build_citation(c) for c in retrieved_chunks}
    return [c for c in cited_sources if c in retrieved_citations]

