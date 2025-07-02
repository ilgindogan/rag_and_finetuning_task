import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# 
answer_file = "../results/no_rag_answers.jsonl"  #  "rag_answer.jsonl"
model_name = "intfloat/e5-base-v2"  


model = SentenceTransformer(model_name)


with open(answer_file, "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

cosine_scores = []



for r in tqdm(results):
    ref = r["ground_truth"].strip()
    hyp = r["generated"].strip()

    if not ref or not hyp:
        continue


    ref_embed = model.encode(f"passage: {ref}", normalize_embeddings=True)
    hyp_embed = model.encode(f"passage: {hyp}", normalize_embeddings=True)

    score = util.cos_sim(torch.tensor(ref_embed), torch.tensor(hyp_embed)).item()
    cosine_scores.append(score)


avg_score = sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0
print(f"\n Average Cosine Similarity: {avg_score:.4f}")
