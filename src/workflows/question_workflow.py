import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QuestionWorkflow:
    """
    Fixed sequence for handling one question, sitting above router_agent:
        cache check -> security check -> route -> dispatch -> filter citations -> cache
    No step here makes a judgment call about the question's content - that's
    router_agent's and rag_agent's job. This file owns the order they run
    in, plus the two generic concerns (caching, citation validation) that
    apply no matter which agent ends up answering.
    """

    def __init__(self, security_agent, router_agent, response_cache):
        self.security_agent = security_agent
        self.router_agent = router_agent
        self.response_cache = response_cache

    # ---------------------------------------------------------------------
    # STEP: run(question, user_role, top_k)
    # ALGORITHM:
    #   1. check response_cache for (question, user_role, top_k)
    #      -> hit: return the cached (answer, sources) immediately, nothing
    #         below this line runs
    #   2. security_agent.check_question(question)
    #      -> rejected: return the fixed refusal immediately, router_agent
    #         never runs
    #   3. router_agent.route(question) picks a route name
    #   4. router_agent.dispatch(route_name) hands back the agent
    #      registered for that route
    #   5. call that agent's answer(question, user_role, top_k) ->
    #      (answer_text, sources, chunks)
    #   6. security_agent.filter_citations(sources, chunks) drops any
    #      citation that doesn't match something actually retrieved -
    #      applied here so it covers whichever agent answered, not just RAG
    #   7. store (answer_text, filtered sources) in the cache
    #   8. return (answer_text, sources, elapsed time)
    # ---------------------------------------------------------------------
    def run(self, question, user_role="general", top_k=None):
        start = time.time()

        cached = self.response_cache.get(question, user_role, top_k)
        if cached is not None:
            answer_text, sources = cached
            logger.info(f"cache_hit role={user_role!r} latency={time.time() - start:.3f}s")
            return answer_text, sources, time.time() - start

        ok, refusal = self.security_agent.check_question(question)
        if not ok:
            logger.warning(f"question_rejected role={user_role!r} question={question!r}")
            return refusal, [], time.time() - start

        route_name = self.router_agent.route(question)
        handler = self.router_agent.dispatch(route_name)
        answer_text, sources, chunks = handler.answer(question, user_role, top_k)
        sources = self.security_agent.filter_citations(sources, chunks)

        self.response_cache.set(question, user_role, top_k, (answer_text, sources))

        latency = time.time() - start
        logger.info(
            f"answered route={route_name!r} role={user_role!r} "
            f"chunks_used={len(chunks)} latency={latency:.3f}s"
        )
        return answer_text, sources, latency
