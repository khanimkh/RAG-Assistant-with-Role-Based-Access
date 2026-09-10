import yaml
from openai import OpenAI

NO_ANSWER_MESSAGE = (
    "I don't have enough information in the provided documents "
    "to answer this question."
)


class LLMService:
    """
    Everything about turning (question, retrieved chunks) into a grounded
    answer: loading the prompt, formatting context, and calling the model.
    """

    def __init__(self, api_key, model_name, prompt_path):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.system_prompt = self._load_prompt(prompt_path)

    # ---------------------------------------------------------------------
    # STEP: _load_prompt(prompt_path)
    # ALGORITHM:
    #   1. read the given YAML file
    #   2. pull out its "system_prompt" key
    #   3. return that string
    #   Prompts live in versioned YAML files (rag_v1.yaml, rag_v2.yaml),
    #   not hardcoded strings in this file, so a prompt change is a normal,
    #   reviewable git diff rather than a code change.
    # ---------------------------------------------------------------------
    def _load_prompt(self, prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config["system_prompt"]

    # ---------------------------------------------------------------------
    # STEP: build_context(chunks)
    # ALGORITHM:
    #   1. for each chunk, format it as:
    #      "[Source: {source}, Page: {page}]\n{text}"
    #   2. join all the formatted chunks with a blank line between them
    #   3. return the combined string
    # ---------------------------------------------------------------------
    def build_context(self, chunks):
     parts = [
        f"[Source: {c['source']}, Page: {c['page']}]\n{c['text']}"
        for c in chunks
     ]
     return "\n\n".join(parts)

    # ---------------------------------------------------------------------
    # STEP: generate_answer(question, chunks)
    # ALGORITHM:
    #   1. build the context string from the retrieved chunks
    #   2. if that context is empty -> return the fixed NO_ANSWER_MESSAGE
    #      immediately, WITHOUT calling the LLM (nothing to ground an
    #      answer in, so there's nothing useful for the model to do)
    #   3. otherwise, build the user message: context + the question
    #   4. call the chat model with:
    #        system message = this instance's loaded system_prompt
    #        user message    = context + question
    #        temperature = 0 (deterministic, not creative, for a QA task)
    #   5. return the model's reply text
    # ---------------------------------------------------------------------
    def generate_answer(self, question, chunks):
        context = self.build_context(chunks)

        if not context.strip():
            return NO_ANSWER_MESSAGE

        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    # ---------------------------------------------------------------------
    # STEP: generate_raw(system_prompt, user_prompt, max_tokens, temperature)
    # ALGORITHM:
    #   1. call the chat model with the GIVEN system_prompt (not this
    #      instance's own loaded RAG prompt) and the given user_prompt
    #   2. return the reply text, stripped of leading/trailing whitespace
    #   A small, generic completion call for short structured outputs -
    #   currently used by router_agent to pick a route name. Kept separate
    #   from generate_answer() because that method is specifically the RAG
    #   answer path and always uses its own loaded grounding prompt.
    # ---------------------------------------------------------------------
    def generate_raw(self, system_prompt, user_prompt, max_tokens=20, temperature=0):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
