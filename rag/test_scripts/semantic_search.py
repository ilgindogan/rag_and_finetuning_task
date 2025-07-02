from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

# Qdrant connect
client = QdrantClient(host="localhost", port=6333)
collection_name = "otranto_chunks"


model = SentenceTransformer("BAAI/bge-small-en-v1.5")


with open("../data/otranto_hierarchical_chunks.jsonl", "r") as f:
    parent_raw = [json.loads(line) for line in f]
    parent_map = {p["parent_id"]: p["parent_text"] for p in parent_raw}


while True:
    question = input("Q: ").strip()
    if not question:
        break

    # BGE formats vector
    query_vector = model.encode(f"passage question: {question}", normalize_embeddings=True).tolist()

    
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        with_payload=True
    )


    for i, hit in enumerate(hits, 1):
        child_text = hit.payload.get("text", "").strip()
        parent_id = hit.payload.get("parent_id")
        parent_text = parent_map.get(parent_id, "").strip()

        print(f"\n#{i} | score: {hit.score:.4f}")
        print(f" Child:\n{child_text}\n")
        print(f" Parent:\n{parent_text[:500]}{'...' if len(parent_text) > 500 else ''}")
        print("-" * 60)
