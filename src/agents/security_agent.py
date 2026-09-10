import re

from src.utils.citations import validate_citations

# Phrases that indicate the question itself is a probing/injection attempt,
# not a real request for document content. Checked before any retrieval
# happens, so a probing question never even reaches the vector store.
SUSPICIOUS_PATTERNS = [
    r"ignore (all|any|previous) instructions",
    r"reveal (your|the) (system prompt|instructions)",
    r"what (documents|files) (do you have|exist)",
    r"list (all|every) (document|source|file)",
]

REFUSAL_MESSAGE = "This question can't be answered by this assistant."


class SecurityAgent:
    """
    Owns two checks that are generic across whichever agent ends up
    answering a question, they don't belong to rag_agent specifically, so
    they're called once, from question_workflow, before/after routing.
    """

    # ---------------------------------------------------------------------
    # STEP: check_question(question)
    # ALGORITHM:
    #   1. lowercase the question
    #   2. for each pattern in SUSPICIOUS_PATTERNS:
    #        if it matches the question -> reject: return
    #        (False, a fixed refusal message)
    #   3. if nothing matched -> allow: return (True, None)
    #   Runs BEFORE routing/retrieval - a rejected question never reaches
    #   router_agent, rag_agent, or the LLM at all.
    # ---------------------------------------------------------------------
    def check_question(self, question):
        lowered = question.lower()
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, lowered):
                return False, REFUSAL_MESSAGE
        return True, None

    # ---------------------------------------------------------------------
    # STEP: filter_citations(cited_sources, retrieved_chunks)
    # ALGORITHM:
    #   1. delegate to utils/citations.validate_citations() - keep only
    #      citations that match something actually retrieved
    #   Runs AFTER generation, on whatever agent answered the question -
    #   applies uniformly, not just to rag_agent.
    # ---------------------------------------------------------------------
    def filter_citations(self, cited_sources, retrieved_chunks):
        return validate_citations(cited_sources, retrieved_chunks)
