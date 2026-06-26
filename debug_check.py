import hashlib
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "./chroma_store"
title = "Lewis Hamilton"

h = hashlib.md5(title.lower().encode()).hexdigest()[:16]
name = f"wiki_{h}"
print(f"Looking for collection: {name}")

client = chromadb.PersistentClient(path=CHROMA_DIR)

print("\nAll collections currently in chroma_store:")
for c in client.list_collections():
    print(" -", c.name, "| count:", c.count())

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

try:
    collection = client.get_collection(name=name, embedding_function=embedding_fn)
    print(f"\nCollection '{name}' exists. Document count: {collection.count()}")

    result = collection.query(query_texts=["Who is Lewis Hamilton?"], n_results=4)
    docs = result.get("documents", [[]])[0]
    print(f"\nRetrieved {len(docs)} chunks:")
    for i, d in enumerate(docs):
        print(f"\n--- Chunk {i+1} ({len(d)} chars) ---")
        print(d[:300])
except Exception as e:
    print(f"\nCollection does not exist or errored: {e}")
