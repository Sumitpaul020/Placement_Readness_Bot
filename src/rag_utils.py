"""
rag_utils.py
------------
Builds (or loads) a FAISS vectorstore from the placement policy document and
exposes a `search_policy_raw(query)` function that:
  - retrieves the most relevant chunks
  - returns an honest fallback if nothing relevant enough is found

Uses free, local HuggingFace sentence-transformer embeddings, so no API key
is needed for this part (only the LLM/agent needs an API key).
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

POLICY_PATH = "policy_docs/placement_policy.txt"
INDEX_DIR = "model/policy_faiss_index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Similarity distance above this = "not confident enough" -> honest fallback.
# (FAISS default returns L2 distance; lower = more similar)
DISTANCE_THRESHOLD = 1.1

_vectorstore = None
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return _embeddings


def build_or_load_index():
    """Build the FAISS index from the policy doc, or load it if it already
    exists on disk, and cache it in memory."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = _get_embeddings()

    if os.path.exists(INDEX_DIR):
        _vectorstore = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        return _vectorstore

    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(raw_text)

    _vectorstore = FAISS.from_texts(chunks, embeddings)
    os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
    _vectorstore.save_local(INDEX_DIR)
    return _vectorstore


def search_policy_raw(query: str, k: int = 3) -> str:
    """Search the policy document. Returns retrieved text, or an honest
    fallback string if nothing is relevant enough."""
    vs = build_or_load_index()
    results = vs.similarity_search_with_score(query, k=k)

    if not results:
        return "NOT_FOUND"

    # Keep only reasonably close matches
    good_matches = [doc.page_content for doc, score in results if score <= DISTANCE_THRESHOLD]

    if not good_matches:
        return "NOT_FOUND"

    return "\n---\n".join(good_matches)


if __name__ == "__main__":
    # Quick manual test
    build_or_load_index()
    print(search_policy_raw("What is the minimum CGPA required for placements?"))
    print("=====")
    print(search_policy_raw("What is the stipend for internships at Google?"))
