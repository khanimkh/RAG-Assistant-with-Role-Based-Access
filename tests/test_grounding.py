from unittest.mock import MagicMock

from src.agents.rag_agent import RAGAgent
from src.services.llm_service import LLMService, NO_ANSWER_MESSAGE


def test_no_retrieved_chunks_gives_no_answer_message():
    llm_service = LLMService.__new__(LLMService)  # It skips __init__, So, there is no need to an API key.
    result = llm_service.generate_answer("some question", [])
    assert result == NO_ANSWER_MESSAGE


def test_agent_returns_no_answer_when_nothing_retrieved():
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = []
    llm_service = MagicMock()

    agent = RAGAgent(vector_store, llm_service, top_k=5)
    answer, sources, _ = agent.answer("question with no matching documents")

    assert answer == NO_ANSWER_MESSAGE
    assert sources == []
    llm_service.generate_answer.assert_not_called()


def test_weak_top_score_is_treated_as_no_answer():
    # top chunk technically "matched" but far below the grounding threshold,
    # this is the case a naive "did we get any chunks back" check would miss.
    weak_chunk = {
        "chunk_id": "doc-p1", "text": "barely related text", "source": "doc.pdf",
        "page": 1, "document_id": "doc", "access_level": "general", "score": 0.1,
    }
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [weak_chunk]
    keyword_search = MagicMock()
    keyword_search.search.return_value = []
    llm_service = MagicMock()

    agent = RAGAgent(vector_store, llm_service, keyword_search=keyword_search, top_k=5)
    answer, sources, _ = agent.answer("an unrelated question")

    assert answer == NO_ANSWER_MESSAGE
    llm_service.generate_answer.assert_not_called()
