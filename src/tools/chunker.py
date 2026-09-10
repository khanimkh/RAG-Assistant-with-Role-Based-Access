from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------------------------------------------------
# STEP: create_text_splitter(chunk_size, chunk_overlap)
# ALGORITHM:
#   1. build a RecursiveCharacterTextSplitter configured with:
#        - chunk_size / chunk_overlap as given
#        - an ordered list of separators to try, from most to least
#          "natural": paragraph break -> line break -> sentence end ->
#          space -> (last resort) split anywhere
#   2. return it
#   At split time, the splitter tries the first separator; if a piece is
#   still too big, it tries the next one down the list. This means it
#   prefers to cut at a paragraph or sentence boundary, and only falls
#   back to a hard character cutoff if nothing more natural fits.
# -------------------------------------------------------------------------
def create_text_splitter(chunk_size=800, chunk_overlap=100):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# -------------------------------------------------------------------------
# STEP: chunk_documents(documents, chunk_size, chunk_overlap)
# ALGORITHM:
#   1. build one splitter, reused for every document
#   2. for each page record in `documents` (from document_loader):
#        a. split its text into pieces using the splitter
#        b. for each piece, in order:
#             - build a chunk_id: "{document_id}-p{page}-c{index}"
#               (readable and traceable back to an exact page - e.g.
#               "AMI_deployment-p12-c3" is page 12, third chunk on it)
#             - carry the page's source/page/document_id onto the chunk,
#               unchanged, so citations still work after chunking
#   3. return the full list of chunks across every document
# -------------------------------------------------------------------------
def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = []

    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunk_id = f"{doc['document_id']}-p{doc['page']}-c{i}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": piece,
                "source": doc["source"],
                "page": doc["page"],
                "document_id": doc["document_id"],
            })

    return chunks
