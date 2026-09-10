import hashlib


# -------------------------------------------------------------------------
# STEP: calculate_file_hash(path)
# ALGORITHM:
#   1. open the file in binary mode, read all its bytes
#   2. run SHA-256 over those bytes
#   3. return the hex digest (a fixed-length string that changes if even
#      one byte of the file changes)
#   Used to decide whether a document has actually changed since the last
#   ingestion run - same hash means same content, so skip re-embedding it.
# -------------------------------------------------------------------------
def calculate_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# -------------------------------------------------------------------------
# STEP: calculate_text_hash(text)
# ALGORITHM:
#   1. encode the string to bytes (UTF-8)
#   2. run SHA-256 over those bytes
#   3. return the hex digest
#   Same idea as calculate_file_hash, for text already in memory rather
#   than a path on disk.
# -------------------------------------------------------------------------
def calculate_text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
