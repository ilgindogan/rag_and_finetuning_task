import json
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
import torch
from tqdm import tqdm

# definition
qa_path = "../data/the_castle_of_otranto_qa_test.json"
chunk_path = "../data/otranto_hierarchical_chunks.jsonl"
collection_name = "otranto_chunks"
embedding_model_name = "intfloat/e5-base-v2"
top_k = 10
rouge_threshold = 0.1 


rouge = Rouge()
embed_model = SentenceTransformer(embedding_model_name)
qclient = QdrantClient(host="localhost", port=6333)


with open(qa_path) as f:
    qa_pairs = [json.loads(line) for line in f]

with open(chunk_path) as f:
    parents = [json.loads(line) for line in f]

# chunks to parent child structure
chunk_text_map = {}
for p in parents:
    for c in p["child_chunks"]:
        chunk_text_map[c["child_id"]] = c["text"]

relevant_count = 0
total = 0
failures = []



for item in tqdm(qa_pairs, desc="processing"):
    question = item["question"]
    if not question.strip():
        continue

    query_text = f"query: {question}"
    query_vector = embed_model.encode(query_text, normalize_embeddings=True)

    hits = qclient.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )

    found_relevant = False

    for hit in hits:
        chunk_text = hit.payload["text"]
        print(f"\n Question: {question}")
        print(f" Chunk: {chunk_text[:200]}...")
 
        try:
            rouge_score = rouge.get_scores(chunk_text, question)[0]['rouge-l']['f']
            print(f" ROUGE-L score: {rouge_score}")
        except:
            rouge_score = 0.0

        if rouge_score >= rouge_threshold:
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
print(f"\n Chunk Relevance (Hit@{top_k}) = {accuracy:.2%} ({relevant_count}/{total})")

with open("irrelevant_questions.json", "w", encoding="utf-8") as f:
    for fail in failures:
        f.write(json.dumps(fail, ensure_ascii=False) + "\n")

print("completed.")
