# qdrant_indexing.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import json
from tqdm import tqdm

# Qdrant bağlantısı
client = QdrantClient(host="localhost", port=6333)
collection_name = "otranto_chunks"
vector_dim = 768  # E5-base-v2 model embedding size

# Koleksiyon varsa sil ve yeniden oluştur
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
)

# Veri dosyasını aç
input_path = "./data/otranto_hierarchical_chunks.jsonl"
points = []

# Child chunk embedding'lerini indexle (ID = integer, child_id payload'da tutulur)
with open(input_path, "r", encoding="utf-8") as f:
    global_idx = 0
    for line in tqdm(f, desc="Indexing chunks"):
        parent = json.loads(line)
        for child in parent["child_chunks"]:
            point = PointStruct(
                id=global_idx,
                vector=child["embedding"],
                payload={
                    "text": child["text"],
                    "parent_id": parent["parent_id"],
                    "child_id": child["child_id"]
                }
            )
            points.append(point)
            global_idx += 1

# Qdrant'a yükle
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"✅ {len(points)} adet chunk başarıyla Qdrant’a yüklendi.")