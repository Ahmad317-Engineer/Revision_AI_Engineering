from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# RAG setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./rag_chroma_db")
collection = chroma.get_or_create_collection("documents")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

class Question(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "RAG Backend running!"}

@app.post("/ask")
def ask(question: Question):
    # 1. Embed
    query_embedding = embedder.encode([question.text]).tolist()
    
    # 2. ChromaDB se chunks dhoondho
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    chunks = results["documents"][0]
    
    # LLM ke baghair — sirf relevant chunks return karo
    return {
        "question": question.text,
        "answer": f"Most relevant content found:\n\n" + "\n\n".join(chunks),
        "sources": chunks
    }