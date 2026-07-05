import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a local ChromaDB (saves to disk in ./chroma_db/)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_notes")

# Your documents — imagine these are chunks from a PDF
documents = [
    "Python was created by Guido van Rossum in 1991.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "ChromaDB is a vector database used to store and search embeddings.",
    "The Eiffel Tower is located in Paris, France.",
    "LangChain helps developers build applications powered by language models.",
]

# Embed and store
embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Stored {collection.count()} documents in ChromaDB\n")

# Query — ask a question, find the most relevant chunks
query = "What is ChromaDB used for?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print(f"Query: '{query}'\n")
print("Top matches:")
for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"  Score: {1 - distance:.3f} — {doc}")