import os
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


# -------------------------------------------------------------------------
# STEP: load_pdf(path)
# ALGORITHM:
#   1. open the PDF, derive a document_id from its filename (name minus
#      extension - this is what ties chunks back to a document later, and
#      what ingestion_workflow uses to delete a document's stale chunks)
#   2. for each page in the PDF, in order:
#        a. extract its text
#        b. if that text is empty (e.g. a scanned image page with no
#           extractable text) -> skip it, don't record an empty record
#        c. otherwise record {text, source, page, document_id}
#           (page number is 1-indexed here, matching how a human would
#           reference "page 3", not a 0-indexed array position)
#   3. return the list of page records
# -------------------------------------------------------------------------
def load_pdf(path):
    reader = PdfReader(path)
    filename = os.path.basename(path)
    document_id = os.path.splitext(filename)[0]

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        pages.append({
            "text": text,
            "source": filename,
            "page": page_number,
            "document_id": document_id,
        })
    return pages


# -------------------------------------------------------------------------
# STEP: load_markdown(path)
# ALGORITHM:
#   1. derive document_id from the filename, same as load_pdf
#   2. read the whole file as one block of text
#   3. return it as a single page record with page = 1
#      (markdown/txt files have no natural page boundary, so the whole
#      file is treated as "page 1" - citations for these files will always
#      say "p.1", which is an accepted limitation, not a bug)
# -------------------------------------------------------------------------
def load_markdown(path):
    filename = os.path.basename(path)
    document_id = os.path.splitext(filename)[0]

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{
        "text": text,
        "source": filename,
        "page": 1,
        "document_id": document_id,
    }]


# -------------------------------------------------------------------------
# STEP: load_document(path)
# ALGORITHM:
#   1. look at the file's extension
#   2. dispatch to the matching loader:
#        .pdf              -> load_pdf()
#        .md / .markdown / .txt -> load_markdown()
#        anything else     -> raise, so an unsupported file is never
#                             silently skipped when loaded directly
#   This is a lookup, not a judgment call - see docs/future_improvements.md
#   for how this dispatch would generalize to more file types.
# -------------------------------------------------------------------------
def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".md", ".markdown", ".txt"):
        return load_markdown(path)
    raise ValueError(f"Unsupported file type: {ext}")


# -------------------------------------------------------------------------
# STEP: load_directory(directory)
# ALGORITHM:
#   1. list every entry in the directory, sorted (deterministic order,
#      useful for reproducible tests and debugging)
#   2. for each entry:
#        a. skip it if it isn't a regular file (e.g. a subfolder)
#        b. skip it if its extension isn't in SUPPORTED_EXTENSIONS
#           (unlike load_document(), this does NOT raise - a folder scan
#           is expected to contain files we don't handle, e.g. a README)
#        c. otherwise load_document() it and add its pages to the result
#   3. return every page record collected across all files
# -------------------------------------------------------------------------
def load_directory(directory):
    all_pages = []
    for filename in sorted(os.listdir(directory)):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        all_pages.extend(load_document(path))
    return all_pages
