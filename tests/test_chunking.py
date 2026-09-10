from src.tools.chunker import chunk_documents


def make_doc(text, source="doc.pdf", page=1, document_id="doc"):
    return {"text": text, "source": source, "page": page, "document_id": document_id}


def test_chunks_generated():
    doc = make_doc("A " * 1000)
    chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1


def test_chunk_metadata_preserved():
    doc = make_doc("Some text about DER hosting capacity.", page=3)
    chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
    assert chunks[0]["source"] == "doc.pdf"
    assert chunks[0]["page"] == 3
    assert chunks[0]["chunk_id"].startswith("doc-p3-c")


def test_chunk_size_reasonable():
    doc = make_doc("A " * 1000)
    chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
    for chunk in chunks:
        assert len(chunk["text"]) <= 600  # small buffer over chunk_size
