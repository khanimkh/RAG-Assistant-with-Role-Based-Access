import json
import yaml
from openai import OpenAI


class RAGJudge:
    """Asks a separate LLM call to score an answer against its retrieved
    context. Faithfulness = is the answer supported by the context.
    Relevance = was the retrieved context actually useful for the question."""

    def __init__(self, api_key, model_name, prompt_path):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = yaml.safe_load(f)["system_prompt"]

    def evaluate_answer(self, question, context, answer):
        user_message = (
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"faithfulness": None, "relevance": None, "reason": "could not parse judge output"}
