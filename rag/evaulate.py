import json
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

rouge = Rouge()

# Load both results
with open("rag_answers.jsonl") as f:
    rag = [json.loads(l) for l in f]
with open("no_rag_answers.jsonl") as f:
    no_rag = [json.loads(l) for l in f]

print("\n📊 Evaluation Results")

for label, data in zip(["RAG", "No-RAG"], [rag, no_rag]):
    rouge_l = []
    bleu = []
    for item in data:
        ref = item["ground_truth"]
        hyp = item["generated"]
        try:
            rouge_l.append(rouge.get_scores(hyp, ref)[0]["rouge-l"]["f"])
            bleu.append(sentence_bleu([ref.split()], hyp.split()))
        except:
            continue

    print(f"\n🔹 {label} Results:")
    print(f"ROUGE-L: {sum(rouge_l)/len(rouge_l):.4f}")
    print(f"BLEU:     {sum(bleu)/len(bleu):.4f}")