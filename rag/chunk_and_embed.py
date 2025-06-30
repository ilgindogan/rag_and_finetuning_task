from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import uuid
import json

# Dosya yolları
input_path = "./data/the_castle_of_otranto_cleaned.txt"
output_path = "./data/otranto_hierarchical_chunks.jsonl"

# Model ve tokenizer
model_id = "intfloat/e5-base-v2"
model = SentenceTransformer(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Chunking parametreleri
chunk_token_size = 256
chunk_overlap = 64
parent_group_size = 3

# Metni oku
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Token bazlı chunk fonksiyonu
def tokenize_chunks(text, tokenizer, chunk_size, overlap):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0].tolist()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_ids = tokens[i:i + chunk_size]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if len(chunk.split()) > 10:
            chunks.append(chunk)
    return chunks

# Child chunk'ları oluştur
child_chunks = tokenize_chunks(text, tokenizer, chunk_token_size, chunk_overlap)
print(f"Toplam child chunk sayısı: {len(child_chunks)}")

# Parent-child yapısını kur
parents = []
for i in range(0, len(child_chunks), parent_group_size):
    group = child_chunks[i:i + parent_group_size]
    parent_id = f"p_{uuid.uuid4().hex[:8]}"
    parent_text = " ".join(group)
    children = []
    for chunk in group:
        child_id = f"c_{uuid.uuid4().hex[:8]}"
        children.append({"child_id": child_id, "text": chunk})
    parents.append({
        "parent_id": parent_id,
        "parent_text": parent_text,
        "child_chunks": children
    })

# Embedding hesapla (E5 için prefix kullan)
for parent in parents:
    for child in parent["child_chunks"]:
        input_text = f"passage: {child['text']}"
        child["embedding"] = model.encode(input_text, normalize_embeddings=True).tolist()

# JSONL olarak kaydet
with open(output_path, "w", encoding="utf-8") as f:
    for parent in parents:
        f.write(json.dumps(parent) + "\n")

print(f"✅ Kaydedildi: {output_path}")
