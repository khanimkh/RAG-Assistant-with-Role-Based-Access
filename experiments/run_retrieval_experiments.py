"""
top_k is a query-time parameter, not an indexing decision, so this reuses the
baseline vector index instead of re-ingesting for every value tested.
"""
from configs.settings import settings
from src.services.vector_store_service import VectorStoreService
from src.services.llm_service import LLMService
from src.agents.rag_agent import RAGAgent, ENABLE_HYBRID_SEARCH, ENABLE_GROUNDING_CHECK
from evaluation.judges import RAGJudge
from evaluation.evaluate_rag import load_evaluation_questions, evaluate_dataset, save_results

import mlflow

TOP_K_VALUES = [3, 5, 8]


def run_retrieval_experiments():
    vector_store = VectorStoreService(
        persist_dir="vector_store_baseline",
        collection_name="energy_docs_baseline",
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    llm_service = LLMService(settings.openai_api_key, settings.llm_model, "configs/prompts/rag_v4.yaml")
    judge = RAGJudge(settings.openai_api_key, settings.llm_model, "configs/prompts/judge.yaml")
    questions = load_evaluation_questions()

    mlflow.set_experiment("rag-case-study")

    all_results = {}
    for top_k in TOP_K_VALUES:
        label = f"top_k_{top_k}"
        print(f"Running {label} ...")
        agent = RAGAgent(vector_store, llm_service, reranker=None, top_k=top_k)

        with mlflow.start_run(run_name=label):
            mlflow.log_params({
                "chunk_size": 500,  # reusing the baseline index - unchanged
                "chunk_overlap": 50,
                "top_k": top_k,
                "prompt_version": "v4",
                "embedding_model": settings.embedding_model,
                "hybrid_search_enabled": ENABLE_HYBRID_SEARCH,
                "grounding_check_enabled": ENABLE_GROUNDING_CHECK,
            })
            results = evaluate_dataset(agent, judge, questions, top_k=top_k)
            mlflow.log_metrics({k: v for k, v in results["aggregate"].items() if v is not None})

        all_results[label] = results["aggregate"]
        print(label, results["aggregate"])

    save_results(all_results, "evaluation_results/retrieval_experiment_results.json")
    with mlflow.start_run(run_name="retrieval_experiments_summary"):
        mlflow.log_artifact("evaluation_results/retrieval_experiment_results.json")
    return all_results


if __name__ == "__main__":
    run_retrieval_experiments()
