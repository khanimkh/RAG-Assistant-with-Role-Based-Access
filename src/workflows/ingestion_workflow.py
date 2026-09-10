import json
import os

from src.tools.document_loader import load_document, load_directory
from src.tools.chunker import chunk_documents
from src.utils.hashing import calculate_file_hash
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


class IngestionWorkflow:
    """
    Ingestion is a fixed, multi-step sequence (load -> chunk -> tag ->
    embed -> store), not a decision, so it's a workflow rather than an
    agent. It skips the expensive step (embedding) for files that haven't
    changed since the last run.
    """

    def __init__(self, vector_store, docs_dir, document_registry,
                 chunk_size=800, chunk_overlap=100, keyword_search=None,
                 manifest_path=None):
        self.vector_store = vector_store
        self.docs_dir = docs_dir
        self.document_registry = document_registry
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keyword_search = keyword_search
        # Scoped to this vector store's own data, not a shared global path -
        # otherwise a fresh, empty collection (like each experiment script
        # builds) could see "file unchanged" from a DIFFERENT collection's
        # manifest and wrongly skip embedding into itself.
        self.manifest_path = manifest_path or os.path.join(vector_store.persist_dir, "ingestion_manifest.json")

    # -------------------------------------------------------------------
    # STEP: _load_manifest() / _save_manifest(manifest)
    # ALGORITHM (load):
    #   1. if no manifest file exists yet (first ever run) -> empty dict
    #   2. otherwise read the JSON {filename: file_hash} written last run
    # ALGORITHM (save):
    #   1. make sure the manifest's parent folder exists
    #   2. write the given {filename: file_hash} dict as JSON
    # -------------------------------------------------------------------
    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _save_manifest(self, manifest):
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # -------------------------------------------------------------------
    # STEP: _changed_files()
    # ALGORITHM:
    #   1. load last run's manifest (filename -> hash it had then)
    #   2. start a fresh, empty new_manifest and an empty changed[] list
    #   3. for each supported file currently in data/docs/:
    #        a. compute its hash right now
    #        b. record that hash in new_manifest (this becomes next run's
    #           "last run" reference, regardless of whether it changed)
    #        c. CALLS document_registry.register_if_missing(filename, hash)
    #           -> src/services/document_registry_service.py - ensures
    #           every file on disk has a registry row, even one dropped in
    #           manually rather than uploaded through /upload. Does NOT
    #           touch access_level if a row already exists (see that
    #           method's own algorithm comment for why).
    #        d. if this file is new, OR its hash differs from what the old
    #           manifest had -> add its filename to changed[]
    #   4. removed = any filename that was in the OLD manifest but isn't in
    #      new_manifest - it existed last run and doesn't exist now, so it
    #      was deleted from data/docs/ since then
    #   5. return (changed[], removed[], new_manifest)
    # -------------------------------------------------------------------
    def _changed_files(self):
        old_manifest = self._load_manifest()
        new_manifest = {}
        changed = []

        for filename in sorted(os.listdir(self.docs_dir)):
            path = os.path.join(self.docs_dir, filename)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue

            file_hash = calculate_file_hash(path)
            new_manifest[filename] = file_hash
            self.document_registry.register_if_missing(filename, file_hash)
            document_id = os.path.splitext(filename)[0]
            if old_manifest.get(filename) != file_hash or not self.vector_store.has_document(document_id):
                changed.append(filename)

        removed = [filename for filename in old_manifest if filename not in new_manifest]

        return changed, removed, new_manifest

    # -------------------------------------------------------------------
    # STEP: run()
    # ALGORITHM:
    #   1. figure out which files are new/changed and which were removed
    #      since the last run, and build this run's manifest
    #      (_changed_files - this also registers every on-disk file in the
    #      document registry, see that method's own algorithm comment)
    #   2. for every removed file:
    #        delete its chunks from the vector store - it's gone from
    #        data/docs/, so answers should stop citing it
    #   3. for every changed file:
    #        a. delete its OLD chunks from the vector store, if any existed
    #        b. load that one file -> page records
    #        c. split those pages into chunks
    #        d. CALLS document_registry.get_access_level(filename) for each
    #           chunk's access_level -> src/services/document_registry_service.py
    #        e. embed + store those chunks; count them
    #   4. save the new manifest, so next run knows what "unchanged" means
    #   5. rebuild the keyword search index from EVERY current document,
    #      not just the changed ones - loading + chunking is cheap/local,
    #      so this always runs; only the costly embedding step in (3e) was
    #      skipped for unchanged files
    #   6. return a stats dict: files seen, files changed, files removed,
    #      chunks embedded
    # -------------------------------------------------------------------
    def run(self):
        changed_files, removed_files, new_manifest = self._changed_files()

        for filename in removed_files:
            document_id = os.path.splitext(filename)[0]
            self.vector_store.delete_document(document_id)

        newly_embedded_chunks = 0
        for filename in changed_files:
            document_id = os.path.splitext(filename)[0]
            self.vector_store.delete_document(document_id)

            path = os.path.join(self.docs_dir, filename)
            pages = load_document(path)
            chunks = chunk_documents(pages, self.chunk_size, self.chunk_overlap)
            access_level = self.document_registry.get_access_level(filename)
            for chunk in chunks:
                chunk["access_level"] = access_level

            self.vector_store.add_chunks(chunks)
            newly_embedded_chunks += len(chunks)

        self._save_manifest(new_manifest)

        if self.keyword_search is not None:
            all_documents = load_directory(self.docs_dir)
            all_chunks = chunk_documents(all_documents, self.chunk_size, self.chunk_overlap)
            for chunk in all_chunks:
                chunk["access_level"] = self.document_registry.get_access_level(chunk["source"])
            self.keyword_search.build_index(all_chunks)

        stats = {
            "files_seen": len(new_manifest),
            "files_changed": len(changed_files),
            "files_removed": len(removed_files),
            "chunks_embedded": newly_embedded_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        logger.info(
            f"ingestion_complete files_seen={stats['files_seen']} "
            f"changed={stats['files_changed']} removed={stats['files_removed']} "
            f"chunks_embedded={stats['chunks_embedded']}"
        )
        return stats
