import chromadb
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor


class VectorStoreService:
    """
    Wraps embedding + storage + similarity search behind one interface.
    If OpenAI embeddings or Chroma are ever swapped for something else,
    only this file changes, nothing upstream (retrieval, agents, API)
    needs to know.
    """

    def __init__(self, persist_dir, collection_name, embedding_model, api_key,
                 embedding_batch_size=500, embedding_max_concurrent_batches=5):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        self.embedding_model = embedding_model
        self.openai_client = OpenAI(api_key=api_key)
        # OpenAI's embeddings endpoint caps how many items/tokens fit in one
        # request - a document with enough chunks would otherwise get
        # rejected outright rather than gracefully split. batch_size keeps
        # each request under that limit; max_concurrent_batches caps how
        # many batches run at once so we don't trade "one huge request" for
        # "so many simultaneous requests we get rate-limited."
        self.embedding_batch_size = embedding_batch_size
        self.embedding_max_concurrent_batches = embedding_max_concurrent_batches

    # ---------------------------------------------------------------------
    # STEP: _embed_batch(texts)
    # ALGORITHM:
    #   1. send this one batch (already known to be under the size limit)
    #      to the OpenAI embeddings API
    #   2. pull the vector out of each returned item, in the same order
    #      the API returned them (which matches input order within a batch)
    #   3. return the list of vectors for just this batch
    # ---------------------------------------------------------------------
    def _embed_batch(self, texts):
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # ---------------------------------------------------------------------
    # STEP: embed(texts)
    # ALGORITHM:
    #   1. if texts fits in one batch (<= embedding_batch_size) -> call
    #      _embed_batch() once directly, no thread pool needed for a
    #      single call
    #   2. otherwise, split texts into chunks of embedding_batch_size
    #   3. run _embed_batch() on each chunk CONCURRENTLY, via a thread
    #      pool capped at embedding_max_concurrent_batches - each batch
    #      call is I/O-bound (waiting on a network response), so nothing
    #      about batch 2 depends on batch 1 finishing first; running them
    #      concurrently means the total wait is closer to the slowest
    #      single batch than to the sum of all of them
    #   4. executor.map() returns results in the SAME ORDER as the input
    #      batches, even though the batches may finish out of order - this
    #      is what keeps step 5 correct without extra bookkeeping
    #   5. flatten the per-batch vector lists back into one list, in the
    #      same order as the original `texts`
    #   Documents and queries MUST go through this same method/model -
    #   otherwise their vectors live in different spaces and similarity
    #   scores between them become meaningless.
    # ---------------------------------------------------------------------
    def embed(self, texts):
        if len(texts) <= self.embedding_batch_size:
            return self._embed_batch(texts)

        batches = [
            texts[i:i + self.embedding_batch_size]
            for i in range(0, len(texts), self.embedding_batch_size)
        ]

        with ThreadPoolExecutor(max_workers=self.embedding_max_concurrent_batches) as executor:
            batch_results = list(executor.map(self._embed_batch, batches))

        vectors = []
        for batch_vectors in batch_results:
            vectors.extend(batch_vectors)
        return vectors

    # ---------------------------------------------------------------------
    # STEP: add_chunks(chunks)
    # ALGORITHM:
    #   1. if there are no chunks, do nothing (avoid an empty API call)
    #   2. embed every chunk's text - embed() itself decides whether that's
    #      one call or several concurrent batched calls (see its own
    #      algorithm comment)
    #   3. store each chunk in Chroma: its id, vector, raw text, and
    #      metadata (source, page, document_id, access_level)
    # ---------------------------------------------------------------------
    def add_chunks(self, chunks):
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        vectors = self.embed(texts)

        self.collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "source": c["source"],
                    "page": c["page"],
                    "document_id": c["document_id"],
                    "access_level": c.get("access_level", "general"),
                }
                for c in chunks
            ],
        )

    # ---------------------------------------------------------------------
    # STEP: similarity_search(query, k, where)
    # ALGORITHM:
    #   1. embed the query with the same embed() used for documents
    #   2. ask Chroma for the k nearest stored vectors, restricted to
    #      whatever `where` filter was passed (e.g. RBAC access levels)
    #   3. Chroma returns parallel lists (ids, texts, metadatas, distances)
    #      for the matches - reassemble them into one list of chunk dicts
    #   4. convert each distance into a similarity score (1 - distance),
    #      so "higher = more similar" everywhere downstream, rather than
    #      mixing "lower distance = closer" and "higher score = better"
    #   5. return the list of chunk dicts, each carrying a "score"
    # ---------------------------------------------------------------------
    def similarity_search(self, query, k=5, where=None):
        query_vector = self.embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
        )

        chunks = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            chunks.append({
                "chunk_id": ids[i],
                "text": docs[i],
                "source": metas[i]["source"],
                "page": metas[i]["page"],
                "document_id": metas[i]["document_id"],
                "access_level": metas[i]["access_level"],
                "score": 1 - distances[i],
            })
        return chunks

    def count(self):
        return self.collection.count()

    def has_document(self, document_id):
        result = self.collection.get(where={"document_id": document_id}, limit=1)
        return bool(result["ids"])

    # ---------------------------------------------------------------------
    # STEP: update_access_level(document_id, new_level)
    # ALGORITHM:
    #   1. fetch every stored chunk whose metadata.document_id matches,
    #      including their current metadata (ids + metadatas)
    #   2. if none found -> return 0, nothing to update
    #   3. for each chunk's metadata dict, overwrite just its access_level
    #      field (Chroma's update() replaces a chunk's whole metadata dict,
    #      not a partial merge - so the existing metadata has to be read
    #      first and only one field changed, not built from scratch)
    #   4. write all the updated metadata dicts back in one call
    #   5. return how many chunks were updated
    #   Deliberately does NOT call embed() or add_chunks() - changing who's
    #   allowed to see a document doesn't change the document's content, so
    #   there's no reason to pay for a new embeddings API call. This is
    #   what makes PATCH /documents/{filename}/access cheap.
    # ---------------------------------------------------------------------
    def update_access_level(self, document_id, new_level):
        result = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
        ids = result["ids"]
        if not ids:
            return 0

        metadatas = result["metadatas"]
        for metadata in metadatas:
            metadata["access_level"] = new_level

        self.collection.update(ids=ids, metadatas=metadatas)
        return len(ids)

    # ---------------------------------------------------------------------
    # STEP: delete_document(document_id)
    # ALGORITHM:
    #   1. delete every stored chunk whose metadata.document_id matches
    #   Lets ingestion remove just one changed/removed file's chunks
    #   without wiping the whole collection.
    # ---------------------------------------------------------------------
    def delete_document(self, document_id):
        self.collection.delete(where={"document_id": document_id})

    # ---------------------------------------------------------------------
    # STEP: reset()
    # ALGORITHM:
    #   1. delete the entire collection by name
    #   2. immediately recreate an empty collection with the same name
    #   Used for a full rebuild, distinct from delete_document()'s
    #   single-file cleanup.
    # ---------------------------------------------------------------------
    def reset(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name,metadata={"hnsw:space": "cosine"})
