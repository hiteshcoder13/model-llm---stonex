# db/chroma_client.py
import chromadb
from config.settings import CHROMA_DIR, CHROMA_COLOR_COLLECTION, CHROMA_EMBEDDING_COLLECTION

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client

def get_color_collection():
    return _get_client().get_or_create_collection(
        name=CHROMA_COLOR_COLLECTION,
        metadata={"hnsw:space": "l2"},
    )

def get_embedding_collection():
    return _get_client().get_or_create_collection(
        name=CHROMA_EMBEDDING_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

# Legacy alias so old code importing get_chroma_collection still works
def get_chroma_collection():
    return get_color_collection()

def collection_count(collection) -> int:
    try:
        return collection.count()
    except Exception:
        return 0