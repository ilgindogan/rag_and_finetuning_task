from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

with open("../data/the_castle_of_otranto_qa_test.json", "r") as f:
    qa_pairs = [json.loads(line) for line in f]

# Init Gemma
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto")
gemma = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=False)

results = []

for item in qa_pairs:
    question = item["question"]
    answer = item["answer1"]

    prompt = f"Question: {question}\nAnswer:"
    response = gemma(prompt, max_new_tokens=40)[0]["generated_text"]
    generated_answer = response.split("Answer:")[-1].strip()

    results.append({
        "question": question,
        "ground_truth": answer,
        "generated": generated_answer
    })

with open("../results/no_rag_answers.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("No-RAG completed")