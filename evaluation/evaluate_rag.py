import json
import os
import time
import yaml

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.services.llm_service import NO_ANSWER_MESSAGE

def normalize_text(text):
    return " ".join(text.strip().lower().split()).rstrip(".")

def load_evaluation_questions(path="data/test_questions.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def evaluate_question(agent, judge, question_item, top_k=5):
    start = time.time()

    answer_text, sources, chunks = agent.answer(
        question_item["question"],
        user_role=question_item.get("role", "general"),
        top_k=top_k,
    )

    latency = time.time() - start

    retrieved_sources = [c["source"] for c in chunks]
    expected_sources = question_item.get("expected_sources", [])
    expected_answerable = question_item.get("answerable", True)

    context = "\n\n".join(c["text"] for c in chunks)

    judge_scores = judge.evaluate_answer(
        question_item["question"],
        context,
        answer_text,
    )

    is_no_answer = (
        normalize_text(answer_text)
        == normalize_text(NO_ANSWER_MESSAGE)
    )

    if expected_answerable:
        no_answer_correct = not is_no_answer
    else:
        no_answer_correct = is_no_answer

    return {
        "id": question_item["id"],
        "question": question_item["question"],
        "role": question_item.get("role", "general"),
        "answerable": expected_answerable,
        "answer": answer_text,
        "sources": sources,
        "retrieved_sources": retrieved_sources,
        "latency": latency,

        "precision_at_k": (
            precision_at_k(
                retrieved_sources,
                expected_sources,
                top_k,
            )
            if expected_sources
            else None
        ),

        "recall_at_k": (
            recall_at_k(
                retrieved_sources,
                expected_sources,
                top_k,
            )
            if expected_sources
            else None
        ),

        "reciprocal_rank": (
            reciprocal_rank(
                retrieved_sources,
                expected_sources,
            )
            if expected_sources
            else None
        ),

        "is_no_answer": is_no_answer,
        "no_answer_correct": float(no_answer_correct),

        "faithfulness": judge_scores.get("faithfulness"),
        "relevance": judge_scores.get("relevance"),
        "judge_reason": judge_scores.get("reason"),
    }


def evaluate_dataset(agent, judge, questions, top_k=5):
    results = [evaluate_question(agent, judge, q, top_k) for q in questions]

    def avg(key):
        values = [r[key] for r in results if r[key] is not None]
        return sum(values) / len(values) if values else None

    aggregate = {
      "faithfulness": avg("faithfulness"),
      "relevance": avg("relevance"),
      "precision_at_k": avg("precision_at_k"),
      "recall_at_k": avg("recall_at_k"),
      "mrr": avg("reciprocal_rank"),
      "no_answer_accuracy": avg("no_answer_correct"),
      "avg_latency": avg("latency"),
    }   
    return {"aggregate": aggregate, "per_question": results}


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
