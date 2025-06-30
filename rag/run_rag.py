# run_rag.py (güncellenmiş E5 + parent chunk prompt)
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import re

# QA verisini yükle
with open("./data/the_castle_of_otranto_qa_test.json", "r") as f:
    qa_pairs = [json.loads(line) for line in f]

# Qdrant ve embedding model (E5-base)
qclient = QdrantClient(host="localhost", port=6333)
embed_model = SentenceTransformer("intfloat/e5-base-v2")

# Batch encode all questions
queries = [f"query: {q['question']}" for q in qa_pairs]
query_vectors = embed_model.encode(queries, normalize_embeddings=True)

# Gemma LLM yükle
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto")
gemma = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=False)

results = []

# Parent chunk'ları (ve child'ları) yükle
with open("./data/otranto_hierarchical_chunks.jsonl", "r") as pf:
    parent_raw = [json.loads(line) for line in pf]
    parent_map = {p["parent_id"]: p for p in parent_raw}

# Soru üzerinde dön
for idx, item in enumerate(qa_pairs):
    print("girdim")
    question = item["question"]
    answer = item["answer1"]
    query_vec = query_vectors[idx].tolist()

    hits = qclient.search(
        collection_name="otranto_chunks",
        query_vector=query_vec,
        limit=10,
        with_payload=True
    )

    parent_ids = set(hit.payload["parent_id"] for hit in hits)
    context_chunks = [parent_map[pid]["parent_text"] for pid in parent_ids if pid in parent_map]
    context = "\n".join(context_chunks)

    prompt = f"""
You are an assistant answering questions based strictly on the provided context below.
Use only the context. Do not guess or hallucinate. If unsure, say you don't know.

Context:
{context}

Question:
{question}

Answer (1-2 sentences):
"""

    response = gemma(prompt, max_new_tokens=128)[0]["generated_text"]
    match = re.search(r"Answer\s*\(.*?\):\s*(.*)", response, re.DOTALL)
    generated_answer = match.group(1).strip() if match else response.strip()

    results.append({
        "question": question,
        "ground_truth": answer,
        "generated": generated_answer
    })

with open("rag_answers.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("✅ RAG tamamlandı.")