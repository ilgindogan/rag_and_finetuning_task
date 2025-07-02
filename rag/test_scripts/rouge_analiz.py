import json
import torch
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util

qa_path = "../data/the_castle_of_otranto_qa_test.json"
answer_path = "../rag_answers.jsonl"
threshold_cosine = 0.80
threshold_rouge = 0.1

embed_model = SentenceTransformer("intfloat/e5-base-v2")
rouge = Rouge()
smoothie = SmoothingFunction().method4

with open(qa_path) as f:
    qa_pairs = [json.loads(l) for l in f]
with open(answer_path) as f:
    predictions = [json.loads(l) for l in f]

assert len(qa_pairs) == len(predictions)

low_rouge_high_cosine = []

for item, pred in tqdm(zip(qa_pairs, predictions), total=len(qa_pairs), desc="process"):
    ref = item["answer1"].strip()
    hyp = pred["generated"].strip()
    q = item["question"]

    if not ref or not hyp:
        continue

    try:
        rouge_l = rouge.get_scores(hyp, ref)[0]["rouge-l"]["f"]
    except:
        rouge_l = 0.0

    try:
        ref_emb = embed_model.encode(f"passage: {ref}", normalize_embeddings=True, convert_to_tensor=True)
        hyp_emb = embed_model.encode(f"passage: {hyp}", normalize_embeddings=True, convert_to_tensor=True)
        cosine = util.cos_sim(ref_emb, hyp_emb).item()
    except:
        cosine = 0.0

    if cosine >= threshold_cosine and rouge_l <= threshold_rouge:
        low_rouge_high_cosine.append({
            "question": q,
            "ground_truth": ref,
            "generated": hyp,
            "rouge_l": round(rouge_l, 4),
            "cosine": round(cosine, 4)
        })


with open("low_rouge_high_cosine.jsonl", "w", encoding="utf-8") as f:
    for item in low_rouge_high_cosine:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n Total {len(low_rouge_high_cosine)} finds (cosine >= {threshold_cosine}, ROUGE-L <= {threshold_rouge}).")

