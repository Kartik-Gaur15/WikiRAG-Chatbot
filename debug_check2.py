import sys
sys.path.insert(0, ".")
from app.rag import WikipediaRAG

rag = WikipediaRAG()

test_queries = [
    "lewis hamilton",
    "iphone 15",
    "water molecule",
    "capital of japan",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print('='*60)
    result = rag.query(q)
    print("ANSWER:", result["answer"][:300])
    print("SOURCES:", result["sources"])
    print("IMAGE:", result["image"])
