from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time
from tqdm import tqdm

# QA data load
with open("../data/the_castle_of_otranto_qa_test.json", "r") as f:
    qa_pairs = [json.loads(line) for line in f]

# Qdrant connect and embedding model
qclient = QdrantClient(host="localhost", port=6333)
embed_model = SentenceTransformer("intfloat/e5-base-v2")

# e5 Format query 
queries = [f"query: {q['question']}" for q in qa_pairs]
query_vectors = embed_model.encode(queries, normalize_embeddings=True)

# LLM model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# Load chunks
with open("../data/otranto_semantic_chunks.jsonl", "r") as pf:
    parents = [json.loads(line) for line in pf]
    parent_map = {p["parent_id"]: p for p in parents}

results = []
start_time = time.time()


for idx, (item, q_vec) in enumerate(tqdm(zip(qa_pairs, query_vectors), total=len(qa_pairs), desc="Processing")):
    question = item['question']
    ground_truth = item['answer1']

    hits = qclient.search(
        collection_name="otranto_chunks_sem",
        query_vector=q_vec.tolist(),
        limit=10,
        with_payload=True
    )

    # Cosine filter
    filtered_hits = [h for h in hits if h.score and h.score >= 0.8]
    if not filtered_hits:
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated": "I don't know"
        })
        continue

    # Parent context
    parent_ids = {hit.payload["parent_id"] for hit in filtered_hits if "parent_id" in hit.payload}
    parent_contexts = [parent_map[pid]["parent_text"] for pid in parent_ids if pid in parent_map]
    context = "\n\n".join(parent_contexts)

    # Token max length
    tokens = tokenizer.encode(context, return_tensors="pt")[0]
    max_input_tokens = 1024 - 64  
    if len(tokens) > max_input_tokens:
        context = tokenizer.decode(tokens[:max_input_tokens])

    # Prompt
    prompt = f"""You are a question answering assistant for a historical novel. Given a question and the context, answer clearly, briefly, and accurately.

    - Only use the information in the context.
    - Do not guess or hallucinate.
    - If the answer is not clearly present, reply: "I don't know."
    - Most answers are short: a name, place, or simple phrase.
    - Do not confuse similar characters (e.g., Jerome vs Frederic).
    - If the question asks for time, action, or location, answer directly.

    Context:
    {context}

    Question: {question}
    Answer:""".lstrip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

    gen = tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    gen = gen.split('\n')[0].strip()

    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "generated": gen
    })

# time consumption calculation
end_time = time.time()
total_sec = end_time - start_time
avg_sec = total_sec / len(qa_pairs)

print(f"\n Total time: {total_sec:.2f} seconds ({avg_sec:.2f} seconds/questions)")

# save jsonl for results
with open("../results/rag_answers_1.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(" All RAG result is completed")
