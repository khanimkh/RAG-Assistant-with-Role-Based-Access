"""
Same chunks, same vector index, same retrieval settings - only the system
prompt changes. Isolates the effect of the grounding/injection-resistance
rules added in v2 (see configs/prompts/rag_v2.yaml).
"""
from configs.settings import settings
from src.services.vector_store_service import VectorStoreService
from src.services.llm_service import LLMService
from src.agents.rag_agent import RAGAgent, ENABLE_HYBRID_SEARCH, ENABLE_GROUNDING_CHECK
from evaluation.judges import RAGJudge
from evaluation.evaluate_rag import load_evaluation_questions, evaluate_dataset, save_results

import mlflow

PROMPT_VERSIONS = ["v1", "v2", "v3", "v4"]


def run_prompt_experiments():
    vector_store = VectorStoreService(
        persist_dir="vector_store_baseline",
        collection_name="energy_docs_baseline",
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    judge = RAGJudge(settings.openai_api_key, settings.llm_model, "configs/prompts/judge.yaml")
    questions = load_evaluation_questions()

    mlflow.set_experiment("rag-case-study")

    all_results = {}
    for version in PROMPT_VERSIONS:
        print(f"Running prompt {version} ...")
        llm_service = LLMService(
            settings.openai_api_key, settings.llm_model, f"configs/prompts/rag_{version}.yaml"
        )
        agent = RAGAgent(vector_store, llm_service, reranker=None, top_k=3)

        with mlflow.start_run(run_name=f"prompt_{version}"):
            mlflow.log_params({
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k": 3,
                "prompt_version": version,
                "embedding_model": settings.embedding_model,
                "hybrid_search_enabled": ENABLE_HYBRID_SEARCH,
                "grounding_check_enabled": ENABLE_GROUNDING_CHECK,
            })
            results = evaluate_dataset(agent, judge, questions, top_k=3)
            mlflow.log_metrics({k: v for k, v in results["aggregate"].items() if v is not None})

        all_results[version] = results["aggregate"]
        print(version, results["aggregate"])

    save_results(all_results, "evaluation_results/prompt_experiment_results.json")
    with mlflow.start_run(run_name="prompt_experiments_summary"):
        mlflow.log_artifact("evaluation_results/prompt_experiment_results.json")
    return all_results


if __name__ == "__main__":
    run_prompt_experiments()
