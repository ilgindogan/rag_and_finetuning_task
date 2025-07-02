

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import json
from tqdm import tqdm
from pathlib import Path

# Definition
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "otranto_chunks_sem"
VECTOR_DIM = 768 
INPUT_PATH = "../data/otranto_semantic_chunks.jsonl"

# Qdrant client connection
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# If collection exist remove
if client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"'{COLLECTION_NAME}' is exists")
    client.delete_collection(collection_name=COLLECTION_NAME)

# Create new collection
print(f"'{COLLECTION_NAME}' creating")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
)

# Load embedding
input_path = Path(INPUT_PATH)
if not input_path.exists():
    raise FileNotFoundError(f"Error no find folders: {input_path}")

points = []
global_idx = 0

with open(input_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Chunks are added qdrant"):
        parent = json.loads(line)
        parent_id = parent.get("parent_id", "unknown")

        for child in parent.get("child_chunks", []):
            embedding = child.get("embedding")
            text = child.get("text")

            if embedding is None or text is None:
                continue 

            point = PointStruct(
                id=global_idx,
                vector=embedding,
                payload={
                    "text": text,
                    "parent_id": parent_id,
                    "child_id": child.get("child_id")
                }
            )
            points.append(point)
            global_idx += 1

# Upsert to qdrant collection
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f" Total {len(points)} chunks are added to qdrant")
