from unittest.mock import MagicMock

from src.agents.rag_agent import RAGAgent

# vector_store and llm_service are mocked here because they call the OpenAI
# API - unit tests shouldn't depend on a live network call or a real API key.
# Question-level security checks and citation filtering now live in
# question_workflow.py, not here - see test_question_workflow.py.


def make_chunk(source, page, text="some retrieved text", score=0.9, access_level="general"):
    return {
        "chunk_id": f"{source}-p{page}",
        "text": text,
        "source": source,
        "page": page,
        "document_id": source,
        "access_level": access_level,
        "score": score,
    }


def test_top_k_is_respected():
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [make_chunk("doc.pdf", i) for i in range(5)]
    llm_service = MagicMock()
    llm_service.generate_answer.return_value = "an answer"

    agent = RAGAgent(vector_store, llm_service, top_k=5)
    _, _, chunks = agent.answer("some ordinary question")

    assert len(chunks) <= 5
    vector_store.similarity_search.assert_called_once()
    assert vector_store.similarity_search.call_args.kwargs["k"] == 5


def test_retrieved_chunks_carry_metadata():
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [make_chunk("doc.pdf", 3)]
    llm_service = MagicMock()
    llm_service.generate_answer.return_value = "an answer"

    agent = RAGAgent(vector_store, llm_service, top_k=5)
    _, sources, chunks = agent.answer("some ordinary question")

    assert chunks[0]["source"] == "doc.pdf"
    assert chunks[0]["page"] == 3
    assert sources == ["doc.pdf, p.3"]


def test_reranker_is_used_when_configured():
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [make_chunk("doc.pdf", i) for i in range(15)]
    reranker = MagicMock()
    reranker.rerank.return_value = [make_chunk("doc.pdf", 1)]
    llm_service = MagicMock()
    llm_service.generate_answer.return_value = "an answer"

    agent = RAGAgent(vector_store, llm_service,
                      reranker=reranker, top_k=5, rerank_candidates=15)
    agent.answer("some ordinary question")

    assert vector_store.similarity_search.call_args.kwargs["k"] == 15
    reranker.rerank.assert_called_once()


def test_role_that_cannot_see_a_chunk_never_gets_it():
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [make_chunk("secret.pdf", 1, access_level="admin")]
    llm_service = MagicMock()

    agent = RAGAgent(vector_store, llm_service, top_k=5)
    answer, sources, chunks = agent.answer("some ordinary question", user_role="general")

    # chunk was retrieved by the mock DB call, but filtered out by the
    # post-retrieval RBAC check, so it never reaches generation
    assert chunks == []
    llm_service.generate_answer.assert_not_called()


def test_exact_match_question_triggers_hybrid_search():
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [make_chunk("doc.pdf", 1, score=0.9)]
    keyword_search = MagicMock()
    keyword_search.search.return_value = []
    llm_service = MagicMock()
    llm_service.generate_answer.return_value = "an answer"

    agent = RAGAgent(vector_store, llm_service,
                      keyword_search=keyword_search, top_k=5)
    agent.answer('What does the spec say about "AB-1234"?')

    keyword_search.search.assert_called_once()
