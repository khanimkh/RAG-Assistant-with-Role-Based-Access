import sqlite3
from datetime import datetime, timezone

DEFAULT_ACCESS_LEVEL = "general"

class DocumentRegistryService:
    # Initialize the SQLite document registry.
    def __init__(self, db_path="data/documents.db"):
        self.db_path = db_path
        self._init_db()

    # Open a SQLite connection with dictionary-like rows.
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # Create the documents table when it does not exist.
    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    filename TEXT PRIMARY KEY,
                    access_level TEXT NOT NULL,
                    uploaded_by TEXT,
                    uploaded_at TEXT NOT NULL,
                    file_hash TEXT
                )
                """
            )

            # Register a document only when it is not already tracked.
    def register_if_missing(self, filename, file_hash):
        uploaded_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO documents
                    (filename, access_level, uploaded_by, uploaded_at, file_hash)
                VALUES (?, ?, NULL, ?, ?)
                """,
                (filename, DEFAULT_ACCESS_LEVEL, uploaded_at, file_hash),
            )

    # Record an upload while preserving an existing access level.
    def record_upload(self, filename, uploaded_by, file_hash):
        uploaded_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (filename, access_level, uploaded_by, uploaded_at, file_hash)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    uploaded_by = excluded.uploaded_by,
                    uploaded_at = excluded.uploaded_at,
                    file_hash = excluded.file_hash
                """,
                (filename, DEFAULT_ACCESS_LEVEL, uploaded_by, uploaded_at, file_hash),
            )

    # Update the access level for a tracked document.
    def set_access_level(self, filename, access_level):
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE documents SET access_level = ? WHERE filename = ?",
                (access_level, filename),
            )
            return cursor.rowcount > 0

            # Return a document's access level or the default level.
    def get_access_level(self, filename):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT access_level FROM documents WHERE filename = ?", (filename,)
            ).fetchone()
        return row["access_level"] if row else DEFAULT_ACCESS_LEVEL

    # Return metadata for one document.
    def get(self, filename):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE filename = ?", (filename,)
            ).fetchone()
        return dict(row) if row else None

    # Return all tracked documents ordered by filename.
    def list_documents(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY filename").fetchall()
        return [dict(row) for row in rows]

    # Delete one document from the registry.
    def delete(self, filename):
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
