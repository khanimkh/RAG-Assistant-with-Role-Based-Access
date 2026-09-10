"""
Changing chunk size/overlap changes what text ends up in each chunk, so unlike
top_k or prompt changes, this requires rebuilding the vector index from scratch
for every configuration tested.
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

CHUNK_CONFIGS = [
    (500, 50),
    (800, 100),
    (1000, 150),
]


def run_chunking_experiments():
    questions = load_evaluation_questions()
    llm_service = LLMService(settings.openai_api_key, settings.llm_model, "configs/prompts/rag_v4.yaml")
    judge = RAGJudge(settings.openai_api_key, settings.llm_model, "configs/prompts/judge.yaml")

    mlflow.set_experiment("rag-case-study")

    all_results = {}
    for chunk_size, chunk_overlap in CHUNK_CONFIGS:
        label = f"chunk_{chunk_size}_{chunk_overlap}"
        print(f"Running {label} ...")

        vector_store = VectorStoreService(
            persist_dir=f"vector_store_{label}",
            collection_name=f"energy_docs_{label}",
            embedding_model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        # Scoped alongside this run's own vector store - same reasoning as
        # in run_baseline.py.
        document_registry = DocumentRegistryService(db_path=f"vector_store_{label}/documents.db")
        ingestion = IngestionWorkflow(
            vector_store, "data/docs", document_registry,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        ingestion.run()

        agent = RAGAgent(vector_store, llm_service, reranker=None, top_k=3)

        with mlflow.start_run(run_name=label):
            mlflow.log_params({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": 3,
                "prompt_version": "v4",
                "embedding_model": settings.embedding_model,
                "hybrid_search_enabled": ENABLE_HYBRID_SEARCH,
                "grounding_check_enabled": ENABLE_GROUNDING_CHECK,
            })
            results = evaluate_dataset(agent, judge, questions, top_k=3)
            mlflow.log_metrics({k: v for k, v in results["aggregate"].items() if v is not None})

        all_results[label] = results["aggregate"]
        print(label, results["aggregate"])

    save_results(all_results, "evaluation_results/chunking_experiment_results.json")
    with mlflow.start_run(run_name="chunking_experiments_summary"):
        mlflow.log_artifact("evaluation_results/chunking_experiment_results.json")
    return all_results


if __name__ == "__main__":
    run_chunking_experiments()
