from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Bağlantı
client = QdrantClient(host="localhost", port=6333)
collection_name = "otranto_chunks"

# Embed modeli
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Soru al
query = input("Soru: ").strip()

# Embed işlemi
query_vector = model.encode(query).tolist()

# Arama (top 3 benzer)
results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3
)

# Sonuçları göster
print("\n🔎 En yakın parçalar:")
for i, hit in enumerate(results, 1):
    print(f"\n#{i} - Skor: {hit.score:.4f}")
    print(hit.payload["text"])
    print("-" * 60)
