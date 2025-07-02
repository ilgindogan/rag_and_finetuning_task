import nltk
import uuid
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# NLTK tokenizer control step
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# path 
input_path = "../data/the_castle_of_otranto_cleaned.txt"
output_path = "../data/otranto_hierarchical_chunks.jsonl"

child_chunk_sent_count = 5        # each child chunk includes
child_overlap = 2                 # overlaping child chunk
parent_group_size = 5             # each parent includes child
min_chunk_word_count = 10         # not include small chunk

# embedding model
model = SentenceTransformer("intfloat/e5-base-v2")

# read txt file and tokenize
with open(input_path, "r", encoding="utf-8") as f:
    full_text = f.read()
sentences = nltk.sent_tokenize(full_text)

# Generating child chunk
def generate_child_chunks(sentences, chunk_size, overlap):
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i + chunk_size])
        if len(chunk.split()) >= min_chunk_word_count:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

child_chunks = generate_child_chunks(sentences, child_chunk_sent_count, child_overlap)
print(f" Total child chunk: {len(child_chunks)}")

# Parent - child structure
parents = []
for i in range(0, len(child_chunks), parent_group_size):
    group = child_chunks[i:i + parent_group_size]
    if not group:
        continue

    parent_id = f"p_{uuid.uuid4().hex[:8]}"
    parent_text = " ".join(group)

    children = []
    for chunk in group:
        child_id = f"c_{uuid.uuid4().hex[:8]}"
        children.append({
            "child_id": child_id,
            "text": chunk
        })

    parents.append({
        "parent_id": parent_id,
        "parent_text": parent_text,
        "child_chunks": children
    })

print(f" Total parent chunk: {len(parents)}")

#  E5 model input format "passage: ..."

for parent in tqdm(parents, desc="Calculating embedding..."):
    for child in parent["child_chunks"]:
        input_text = f"passage: {child['text']}"
        embedding = model.encode(input_text, normalize_embeddings=True)
        child["embedding"] = embedding.tolist()

    parent_input = f"passage: {parent['parent_text']}"
    parent_embedding = model.encode(parent_input, normalize_embeddings=True)
    parent["embedding"] = parent_embedding.tolist()

# Save jsonl file
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for parent in parents:
        f.write(json.dumps(parent, ensure_ascii=False) + "\n")

print(f"saved jsonl file: {output_path}")
