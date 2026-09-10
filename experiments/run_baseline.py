"""
Baseline configuration, deliberately simple:
    chunk_size=500, chunk_overlap=50, top_k=3, prompt=v4, reranker=off

Every later experiment (chunking, retrieval, prompt) changes exactly one of
these and compares back to this run.
"""
from configs.settings import settings
from src.services.vector_store_service import VectorStoreService
from src.services.llm_service import LLMService
from src.services.document_registry_service import DocumentRegistryService
from src.agents.rag_agent import RAGAgent, ENABLE_HYBRID_SEARCH, ENABLE_GROUNDING_CHECK
from src.workflows.ingestion_workflow import IngestionWorkflow
from evaluation.judges import RAGJudge
from evaluation.evaluate_rag import load_evaluation_questions, evaluate_dataset, save_results

import mlflow


def run_baseline():
    vector_store = VectorStoreService(
        persist_dir="vector_store_baseline",
        collection_name="energy_docs_baseline",
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    # Scoped alongside this run's own vector store, same reasoning as the
    # ingestion manifest's scoping: a shared registry path would leak
    # access-level state between this experiment and the real app / other
    # experiments.
    document_registry = DocumentRegistryService(db_path="vector_store_baseline/documents.db")

    ingestion = IngestionWorkflow(
        vector_store, "data/docs", document_registry,
        chunk_size=500, chunk_overlap=50,
    )
    stats = ingestion.run()
    print("Ingestion stats:", stats)

    llm_service = LLMService(settings.openai_api_key, settings.llm_model, "configs/prompts/rag_v4.yaml")
    agent = RAGAgent(vector_store, llm_service, reranker=None, top_k=3)
    judge = RAGJudge(settings.openai_api_key, settings.llm_model, "configs/prompts/judge.yaml")

    questions = load_evaluation_questions()

    mlflow.set_experiment("rag-case-study")
    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params({
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k": 3,
            "prompt_version": "v4",
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "reranker_enabled": False,
            "hybrid_search_enabled": ENABLE_HYBRID_SEARCH,
            "grounding_check_enabled": ENABLE_GROUNDING_CHECK,
        })

        results = evaluate_dataset(agent, judge, questions, top_k=3)

        mlflow.log_metrics({k: v for k, v in results["aggregate"].items() if v is not None})
        save_results(results, "evaluation_results/baseline_results.json")
        mlflow.log_artifact("evaluation_results/baseline_results.json")

    print("Baseline aggregate metrics:", results["aggregate"])
    return results


if __name__ == "__main__":
    run_baseline()
