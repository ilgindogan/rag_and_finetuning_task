import json
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm
from tabulate import tabulate

# DEfine embed models
EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2"
]
LLM_MODEL_NAME = "google/gemma-3-1b-it"
TOP_K = 5
MAX_CONTEXT_CHAR = 1500

smoothie = SmoothingFunction().method4
rouge = Rouge()


with open("./data/the_castle_of_otranto_qa_test.json") as f:
    qa_pairs = [json.loads(line) for line in f]

# load chunks
with open("./data/otranto_hierarchical_chunks.jsonl", "r") as f:
    parents = [json.loads(line) for line in f]
all_chunks = [c["text"] for p in parents for c in p["child_chunks"]]

# 
print(f"\n LLM: {LLM_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()


results = []

for emb_name in EMBEDDING_MODELS:
    print(f"\n Testing embed models: {emb_name}")
    emb_model = SentenceTransformer(emb_name)
    emb_model.eval()

    chunk_vectors = emb_model.encode(all_chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    rouge_l, bleu, em_hits = [], [], []

    for item in tqdm(qa_pairs, desc="process"):
        q = item["question"]
        gt = item["answer1"]

        if not q.strip() or not gt.strip():
            continue

        q_text = f"query: {q}" if "bge" in emb_name or "e5" in emb_name else q
        q_vec = emb_model.encode(q_text, normalize_embeddings=True)

        sims = util.cos_sim(q_vec, chunk_vectors)[0]
        top_idx = torch.topk(sims, k=TOP_K).indices.tolist()
        retrieved = [all_chunks[i] for i in top_idx]
        context = "\n\n".join(retrieved)[:MAX_CONTEXT_CHAR]

       
        prompt = f"""
You are an assistant. Answer ONLY based on the context. Do not make anything up.

Context:
{context}

Q: {q}
A:""".strip()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        gen = gen.split('\n')[0].strip()

        try:
            rouge_score = rouge.get_scores(gen, gt)[0]["rouge-l"]["f"]
            bleu_score = sentence_bleu([gt.split()], gen.split(), smoothing_function=smoothie)
            rouge_l.append(rouge_score)
            bleu.append(bleu_score)
            em_hits.append(gen.strip().lower() == gt.strip().lower())
        except:
            continue

    results.append({
        "embedding_model": emb_name,
        "ROUGE-L": round(sum(rouge_l) / len(rouge_l), 4) if rouge_l else 0.0,
        "BLEU": round(sum(bleu) / len(bleu), 4) if bleu else 0.0,
        "Exact Match": round(sum(em_hits) / len(em_hits), 4) if em_hits else 0.0
    })

# tabulate for table results
print("\n Results:\n")
print(tabulate(results, headers="keys", tablefmt="github"))
