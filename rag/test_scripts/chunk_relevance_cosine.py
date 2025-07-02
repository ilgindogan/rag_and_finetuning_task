import json
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from tqdm import tqdm
import torch

# definition
qa_path = "../data/the_castle_of_otranto_qa_test.json"
chunk_path = "../data/otranto_semantic_chunks.jsonl"
collection_name = "otranto_chunks_sem"
embedding_model_name = "intfloat/e5-base-v2"
top_k = 1
cosine_threshold = 0.8  

# qdrant connect
embed_model = SentenceTransformer(embedding_model_name)
qclient = QdrantClient(host="localhost", port=6333)


with open(qa_path) as f:
    qa_pairs = [json.loads(line) for line in f]

with open(chunk_path) as f:
    parents = [json.loads(line) for line in f]


chunk_embedding_map = {}
for p in parents:
    for c in p["child_chunks"]:
        chunk_embedding_map[c["child_id"]] = c["embedding"]

# Test
relevant_count = 0
total = 0
failures = []


for item in tqdm(qa_pairs, desc="Cosine calculate"):
    question = item["question"]
    if not question.strip():
        continue

    query_text = f"query: {question}"
    query_vector = embed_model.encode(query_text, normalize_embeddings=True)

    hits = qclient.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
        with_vectors=True
    )

    found_relevant = False

    for hit in hits:
        chunk_vector = hit.vector
        similarity = util.cos_sim(torch.tensor(query_vector), torch.tensor(chunk_vector))[0][0].item()

        print(f"\n Question: {question}")
        print(f"Chunk: {hit.payload['text'][:200]}...")
        print(f"Cosine similarity: {similarity:.3f}")

        if similarity >= cosine_threshold:
            found_relevant = True
            break

    if found_relevant:
        relevant_count += 1
    else:
        failures.append({
            "question": question,
            "top_chunks": [hit.payload["text"] for hit in hits]
        })

    total += 1

accuracy = relevant_count / total if total > 0 else 0.0
print(f"\n Cosine Relevance (Hit@{top_k}) = {accuracy:.2%} ({relevant_count}/{total})")

# logs irrevelant results
with open("irrelevant_questions_cosine.json", "w", encoding="utf-8") as f:
    for fail in failures:
        f.write(json.dumps(fail, ensure_ascii=False) + "\n")

print("Completed")
