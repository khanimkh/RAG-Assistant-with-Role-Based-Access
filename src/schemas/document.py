from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Internal representation of a chunk as it moves through loading, chunking,
    embedding and retrieval. Not used directly by the API - the API only sees
    AnswerResponse."""
    chunk_id: str
    text: str
    source: str
    page: int
    document_id: str
    access_level: str = "general"
