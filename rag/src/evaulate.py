import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

rouge = Rouge()
smoothie = SmoothingFunction().method4

# Load both results
with open("../results/rag_answers.jsonl") as f:
    rag = [json.loads(l) for l in f]
with open("../results/no_rag_answers.jsonl") as f:
    no_rag = [json.loads(l) for l in f]

print("\n Evaluation Results")

for label, data in zip(["RAG", "No-RAG"], [rag, no_rag]):
    rouge_l = []
    bleu = []
    for item in data:
        ref = item["ground_truth"]
        hyp = item["generated"]
        if not ref.strip() or not hyp.strip():
            continue
        try:
            rouge_l.append(rouge.get_scores(hyp, ref)[0]["rouge-l"]["f"])
            bleu.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie))
        except:
            continue

    rouge_score = sum(rouge_l) / len(rouge_l) if rouge_l else 0.0
    bleu_score = sum(bleu) / len(bleu) if bleu else 0.0

    print(f"\n {label} Results:")
    print(f"ROUGE-L: {rouge_score:.4f}")
    print(f"BLEU:     {bleu_score:.4f}")
