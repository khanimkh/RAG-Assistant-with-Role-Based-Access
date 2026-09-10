import yaml

DEFAULT_ROUTE = "rag"


class RouterAgent:
    """
    Decides which registered agent answers a question, and hands off to it.
    `routes` is a plain dict, dispatch is a dict lookup, no graph library,
    since a single linear "pick one, then run it" flow doesn't need one.

    Only "rag" is registered today. Adding a future route (e.g. "sql" for
    structured data via Athena) means: write that agent with an
    answer(question, user_role, top_k) method matching RAGAgent's, and add
    one line to the `routes` dict passed in from main.py, not a change to
    this file.
    """

    def __init__(self, llm_service, routes, prompt_path="configs/prompts/router.yaml"):
        self.llm_service = llm_service
        self.routes = routes  # {route_name: agent}
        self.route_names = list(routes.keys())
        self.system_prompt = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["system_prompt"]

    # ---------------------------------------------------------------------
    # STEP: route(question)
    # ALGORITHM:
    #   1. if only one route is registered:
    #        -> nothing to decide - return its name directly, and skip the
    #           LLM call entirely (a call to pick between one option is
    #           pure waste)
    #   2. otherwise:
    #        a. build a prompt listing the registered route names + the
    #           question
    #        b. ask the LLM for a short completion (max 10 tokens - it
    #           should return just a route name)
    #        c. lowercase/strip the response
    #        d. if it ISN'T one of the registered route names -> fall back
    #           to DEFAULT_ROUTE (or the only route available, if
    #           DEFAULT_ROUTE isn't registered) - a bad pick must never
    #           crash the pipeline
    #        e. return the picked (or fallback) name
    # ---------------------------------------------------------------------
    def route(self, question):
        if len(self.route_names) == 1:
            return self.route_names[0]

        user_prompt = f"Available routes: {', '.join(self.route_names)}\n\nQuestion: {question}"
        raw = self.llm_service.generate_raw(self.system_prompt, user_prompt, max_tokens=10, temperature=0)
        picked = raw.strip().lower()

        if picked not in self.routes:
            return DEFAULT_ROUTE if DEFAULT_ROUTE in self.routes else self.route_names[0]

        return picked

    # ---------------------------------------------------------------------
    # STEP: dispatch(route_name)
    # ALGORITHM:
    #   1. look up route_name in the routes dict
    #   2. return the agent registered under it
    # ---------------------------------------------------------------------
    def dispatch(self, route_name):
        return self.routes[route_name]
