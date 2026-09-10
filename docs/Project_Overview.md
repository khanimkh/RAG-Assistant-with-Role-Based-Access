**E Source RAG Assistant**

Project Overview & Design Decisions

1\. What This Project Does

This project is a Retrieval-Augmented Generation (RAG) assistant that
answers natural-language questions over E Source's energy industry
technical documentation. Engineers and analysts ask a question in plain
English, the system retrieves the relevant passages from the ingested
PDFs, generates a grounded answer with citations, and refuses to answer
when the documents genuinely do not contain the information.

Beyond the core pipeline (ingest, chunk, embed, retrieve, generate), the
system implements role-based access control, authentication, an admin
document-management interface, response caching, and a full offline
evaluation harness with tracked experiments, covering all three goals of
the case study, not just the baseline pipeline.

2\. Architecture & Key Design Decisions

The codebase is organized around one deliberate rule, applied
consistently everywhere: a component is called a workflow if it always
runs the same fixed sequence of steps, and an agent only if an observed
result genuinely changes what happens next. This distinction shaped
every structural decision below.

2.1 Agents vs. Workflows

- RAGAgent (agent): decides whether a question needs hybrid (vector +
  keyword) retrieval based on its shape, and retries with the other
  strategy if the first attempt scores weakly. Both are real decisions
  driven by an observed result, which is what makes it an agent.

- SecurityAgent (agent): decides whether a question is a probing or
  prompt-injection attempt before any retrieval happens, and filters
  generated citations down to sources that were actually retrieved.

- RouterAgent (agent): decides which downstream agent answers a
  question. Today only a “rag” route is registered, so it skips the LLM
  call entirely, since there is nothing to decide between one option.
  The seam is built for a future structured-data route without requiring
  any restructuring.

- IngestionWorkflow and QuestionWorkflow (workflows): fixed sequences,
  hash then load then chunk then embed for ingestion, and cache check
  then security check then route then dispatch then filter citations for
  a question. Neither makes a judgment call, they own ordering and
  cross-cutting concerns (caching, citation validation) so those
  concerns are not duplicated inside every agent.

2.2 Access Control (RBAC)

Access is enforced at retrieval time, not by asking the model to
withhold information. Each document is tagged with an access level
(general, power, or admin), a user’s role is resolved into the levels it
is allowed to see, and that list is applied as a database filter before
any chunk is retrieved. A restricted chunk is never in the LLM’s context
window in the first place. A second, redundant check re-verifies each
returned chunk’s access level independently of the database filter, so a
misconfigured filter cannot silently leak a chunk.

Document access levels are tracked in a small SQLite registry (filename, access level, uploader, upload time, content hash).

2.3 Authentication

Login issues a signed JWT containing the user’s role. Every protected
endpoint reads its authorization from that verified token, never from a
client-supplied field in the request body. 

2.4 Grounding & Hybrid Retrieval

Two agent-level behaviors are the project’s main quality levers, and
both are implemented as toggles (ENABLE_HYBRID_SEARCH,
ENABLE_GROUNDING_CHECK) so a baseline run can be reproduced for
comparison without deleting any code:

- Grounding threshold: if the top retrieved chunk’s similarity score
  falls below a threshold, the system returns a fixed “insufficient
  information” response instead of calling the LLM, directly answering
  the case study’s requirement that the system identify when it lacks
  enough information to answer.

- Hybrid retrieval: questions containing an exact token, such as a
  quoted phrase, or a model-number-shaped string,
  are retried with keyword search merged into the vector results, since
  plain vector similarity is weaker on exact-token matching than on
  conceptual matching.

2.5 Caching

Two independent caches keep the system cheap to iterate on: an
ingestion-time content hash skips re-embedding any file whose content
has not changed since the last run, and an in-memory, TTL-based response
cache skips retrieval and generation entirely for a repeated question.

3\. Evaluation Methodology

Evaluation follows the pipeline the case study asks for: for each test
question, run the full RAG pipeline, score retrieval quality
(precision@k, recall@k, MRR) against expected sources, and score answer
quality via an LLM-as-judge (faithfulness, relevance). A
no-answer-accuracy metric was added on top of this to specifically
measure whether the system correctly refused truly unanswerable
questions and correctly answered questions that were only partially
covered by the documents. This turned out to be the most diagnostic
metric in the whole evaluation, as the results below show.

Four experiments were run and tracked in MLflow, each isolating exactly
one variable while holding the rest constant:

4\. Experiment Results

4.1 Baseline

Configuration: chunk_size=500, chunk_overlap=50, top_k=3, prompt v4,
reranker off.

| **Baseline** | **Faithfulness** | **Relevance** | **Precision@K** | **No-Answer Acc.** | **Avg Latency (s)** |
|----|----|----|----|----|----|
| baseline | 1.00 | 1.00 | 1.00 | 1.00 | 2.56 |

4.2 Chunking Strategy

Chunk size and overlap changed, everything else held constant. This
required rebuilding the vector index from scratch for each
configuration, since different chunk boundaries invalidate the previous
embeddings.

| **Configuration** | **Faithfulness** | **Relevance** | **Precision@K** | **No-Answer Acc.** | **Avg Latency (s)** |
|----|----|----|----|----|----|
| chunk_500 / overlap_50 | 0.96 | 0.94 | 1.00 | 1.00 | 2.74 |
| chunk_800 / overlap_100 | 1.00 | 1.00 | 1.00 | 1.00 | 2.69 |
| chunk_1000 / overlap_150 | 1.00 | 1.00 | 1.00 | 1.00 | 2.66 |

*Larger chunks (800 to 1000 characters) slightly outperformed the
smaller baseline chunk size on both faithfulness and relevance, likely
because more context survives intact within a single chunk for this
document set.*

4.3 Retrieval (top_k)

Same index, same prompt, only the number of retrieved chunks changed.
This does not require re-embedding, only a different query-time
parameter.

| **top_k** | **Faithfulness** | **Relevance** | **Precision@K** | **No-Answer Acc.** | **Avg Latency (s)** |
|----|----|----|----|----|----|
| top_k = 3 | 1.00 | 0.80 | 1.00 | 1.00 | 2.86 |
| top_k = 5 | 1.00 | 1.00 | 1.00 | 1.00 | 3.67 |
| top_k = 8 | 0.92 | 0.94 | 0.95 | 1.00 | 4.04 |

*top_k = 5 was the strongest overall result, and top_k = 8 introduced
enough marginal, less-relevant context to slightly reduce faithfulness
and precision while increasing latency, so more retrieved context is not
free.*

4.4 Prompt Versions, the Headline Finding

Four prompt versions were tested, holding chunking and retrieval
constant. This experiment produced the most significant result in the
whole evaluation.

| **Prompt** | **Faithfulness** | **Relevance** | **Precision@K** | **No-Answer Acc.** | **Avg Latency (s)** |
|----|----|----|----|----|----|
| v1 (baseline) | 1.00 | 1.00 | 1.00 | 0.40 | 1.44 |
| v2 (+ grounding & injection rules) | 1.00 | 1.00 | 1.00 | 0.40 | 1.46 |
| v3 (+ partial-answer decomposition) | 0.98 | 0.80 | 1.00 | 1.00 | 2.40 |
| v4 (+ citation verification) | 1.00 | 1.00 | 1.00 | 1.00 | 2.60 |

**Observed weakness:** v1 and v2 scored 1.00 on faithfulness and
relevance, but only 0.40 on no-answer accuracy. Both prompts told the
model to refuse when “the context does not contain enough information,”
but in practice, this caused the model to refuse the entire question
whenever any part of it lacked full support, even when other parts of
the question were genuinely answerable from the retrieved context.

**Fix:** v3 introduced an explicit instruction to decompose a multi-part
question, answer every part the context supports, and only apply the
full refusal when no part of the question is supported at all, plus an
explicit rule against treating a related-but-not-identical term, such as
a generic utility rebate program, as equivalent to a specific
classification, such as a DSM program, unless the source text says so
explicitly. This alone took no-answer accuracy from 0.40 to 1.00.

**v4 chosen configuration:** v4 kept v3’s decomposition logic and added
an evidence-verification rule. The model must confirm a fact actually
appears on the page it is about to cite before citing it, and must
describe an answer as partial rather than implying full unanswerability.
v4 is the prompt shipped in the final configuration: full accuracy
across every metric, with an acceptable latency cost (2.60s vs. 1.44s
baseline) for the added faithfulness guarantee.

5\. Known Limitations & Future Work

- No token revocation or rate limiting on /login yet.

- Session token is kept in memory only in the frontend, not browser
  storage, so a page refresh logs the user out. The real fix is a
  refresh-token flow or an httpOnly cookie.

- The ingestion manifest has no lock, so two admins triggering ingestion
  at the same moment could race.

- Keyword search uses simple word-overlap scoring, not real BM25,
  sufficient for this document set but not yet production-grade.

- No structured-data (spreadsheet/CSV) route yet. The RouterAgent seam
  is built for this, but a second agent has not been added.

