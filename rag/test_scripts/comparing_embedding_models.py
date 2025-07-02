import json
import torch
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm


EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
]


TOP_K = 5
MAX_CONTEXT_CHAR = 1500
smoothie = SmoothingFunction().method4
rouge = Rouge()


with open("../data/the_castle_of_otranto_qa_test.json") as f:
    qa_pairs = [json.loads(line) for line in f]

with open("../data/otranto_hierarchical_chunks.jsonl", "r") as f:
    parents = [json.loads(line) for line in f]


all_chunks = []
for parent in parents:
    for child in parent["child_chunks"]:
        all_chunks.append(child["text"])

print(f" Total QA: {len(qa_pairs)}, Total Chunks: {len(all_chunks)}")


results_summary = []

for model_name in EMBEDDING_MODELS:
    print(f"\n testing {model_name}")
    model = SentenceTransformer(model_name)
    model.eval()

    chunk_vectors = model.encode(all_chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    rouge_l = []
    bleu = []

    for item in tqdm(qa_pairs, desc="running"):
        question = item["question"]
        ground_truth = item["answer1"]

        if not question.strip() or not ground_truth.strip():
            continue


        query_embed = model.encode(f"query: {question}" if "bge" in model_name or "e5" in model_name else question,
                                   normalize_embeddings=True)


        sims = util.cos_sim(query_embed, chunk_vectors)[0]
        top_k_idx = torch.topk(sims, k=TOP_K).indices.tolist()
        retrieved_chunks = [all_chunks[i] for i in top_k_idx]


        context = "\n\n".join(retrieved_chunks)[:MAX_CONTEXT_CHAR]
        generated = f"(Simulated Answer) {context}" 

        try:
            rouge_score = rouge.get_scores(generated, ground_truth)[0]["rouge-l"]["f"]
            bleu_score = sentence_bleu([ground_truth.split()], generated.split(), smoothing_function=smoothie)
            rouge_l.append(rouge_score)
            bleu.append(bleu_score)
        except:
            continue

    avg_rouge = sum(rouge_l) / len(rouge_l) if rouge_l else 0.0
    avg_bleu = sum(bleu) / len(bleu) if bleu else 0.0

    results_summary.append({
        "model": model_name,
        "ROUGE-L": round(avg_rouge, 4),
        "BLEU": round(avg_bleu, 4)
    })


from tabulate import tabulate
print("\n Final Results:\n")
print(tabulate(results_summary, headers="keys", tablefmt="github"))
