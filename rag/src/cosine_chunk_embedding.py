import nltk
import uuid
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from tqdm import tqdm

# NLTK tokenizer
nltk.download("punkt")

# Path
input_path = "../data/the_castle_of_otranto_cleaned.txt"
output_path = "../data/otranto_semantic_chunks.jsonl"

similarity_threshold = 0.65         # Cosine similiratiy threshold
max_chunk_sentences = 5             
parent_group_size = 4              
min_chunk_word_count = 15          

# Embedding Model
model = SentenceTransformer("intfloat/e5-base-v2")

# read and tokenize
with open(input_path, "r", encoding="utf-8") as f:
    full_text = f.read()
sentences = nltk.sent_tokenize(full_text)

# Embedding creating
sentence_embeddings = model.encode(sentences, normalize_embeddings=True)

# Sentence filter rule
def is_valid_sentence(sent):
    words = sent.strip().split()
    return len(words) > 4 and not sent.strip().endswith("?")

# Creating semantic chunks
chunks = []
current_chunk = []
current_chunk_embeds = []

for i, sent in enumerate(sentences):
    if not is_valid_sentence(sent):
        continue

    emb = sentence_embeddings[i]

    if not current_chunk:
        current_chunk = [sent]
        current_chunk_embeds = [emb]
        continue

    avg_embed = np.mean(current_chunk_embeds, axis=0)
    sim = util.cos_sim(emb, avg_embed).item()

    if sim >= similarity_threshold and len(current_chunk) < max_chunk_sentences:
        current_chunk.append(sent)
        current_chunk_embeds.append(emb)
    else:
        if len(" ".join(current_chunk).split()) >= min_chunk_word_count:
            chunks.append(" ".join(current_chunk))
        current_chunk = [sent]
        current_chunk_embeds = [emb]

# Append last chunk
if current_chunk and len(" ".join(current_chunk).split()) >= min_chunk_word_count:
    chunks.append(" ".join(current_chunk))

print(f"Total semantic child chunk: {len(chunks)}")

# Create Parent-Child Structure
parents = []
for i in range(0, len(chunks), parent_group_size):
    group = chunks[i:i + parent_group_size]
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

print(f"Total parent chunk: {len(parents)}")

# Calculate embedding
for parent in tqdm(parents, desc="Calculating embeddings..."):
    for child in parent["child_chunks"]:
        input_text = f"passage: {child['text']}"
        child["embedding"] = model.encode(input_text, normalize_embeddings=True).tolist()

    parent_input = f"passage: {parent['parent_text']}"
    parent["embedding"] = model.encode(parent_input, normalize_embeddings=True).tolist()

# Save jsonl
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for parent in parents:
        f.write(json.dumps(parent, ensure_ascii=False) + "\n")

print(f" Semantic chunking saved: {output_path}")
