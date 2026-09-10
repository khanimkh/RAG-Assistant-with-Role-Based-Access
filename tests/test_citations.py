from src.utils.citations import build_citation, extract_sources, validate_citations


def test_build_citation():
    chunk = {"source": "doc.pdf", "page": 5}
    assert build_citation(chunk) == "doc.pdf, p.5"


def test_extract_sources_deduplicates_and_keeps_order():
    chunks = [
        {"source": "doc.pdf", "page": 5},
        {"source": "doc.pdf", "page": 5},
        {"source": "doc.pdf", "page": 7},
    ]
    assert extract_sources(chunks) == ["doc.pdf, p.5", "doc.pdf, p.7"]


def test_validate_citations_rejects_source_that_was_never_retrieved():
    retrieved = [{"source": "doc.pdf", "page": 5}]
    cited = ["doc.pdf, p.5", "made_up.pdf, p.1"]
    assert validate_citations(cited, retrieved) == ["doc.pdf, p.5"]
