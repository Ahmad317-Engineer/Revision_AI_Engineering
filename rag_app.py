# === STEP 1: Libraries import karo ===
# Yeh waise hi hai jaise school bag mein se books nikalte ho

import chromadb                          # Yeh hamara "filing cabinet" hai — chunks store karta hai
from sentence_transformers import SentenceTransformer  # Yeh words ko numbers mein badalta hai
from openai import OpenAI               # Yeh LLM se baat karne ka rasta hai
from dotenv import load_dotenv          # Yeh .env file se secret key uthata hai
import os                               # Yeh computer ke folders dekhne deta hai

# === STEP 2: Secret key load karo ===
# .env file mein tumhari API key chupi hai — yahan uthate hain
load_dotenv()

# === STEP 3: Teen cheezein banao ===

# 1. LLM client — OpenRouter ke through
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")  # .env se key uthao
)

# 2. Embedder — words ko numbers mein badalne wala
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# Yeh model already tumhare computer mein hai (Step 5 mein download hua tha)

# 3. ChromaDB — hamara filing cabinet
chroma = chromadb.PersistentClient(path="./rag_chroma_db")
# PersistentClient matlab: data computer par save rehta hai, band karne ke baad bhi
collection = chroma.get_or_create_collection("documents")
# "documents" naam ka folder banao andar

# === STEP 4: Function — folder se files load karo ===
def load_folder(folder_path):
    """Ek poore folder ki saari .txt files padho aur ChromaDB mein daalo"""
    
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    # Folder mein se sirf .txt files nikalo
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    total_chunks = 0
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)  # poora rasta banao
        
        # File padho
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Text ko tukdon mein kaat do (paragraphs se)
        # Agar tukda 30 characters se chhota hai toh skip karo
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 30]
        
        # ChromaDB mein kitne chunks pehle se hain? (duplicate IDs se bachne ke liye)
        existing = collection.count()
        
        # Har chunk ko numbers mein badlo (embed karo)
        embeddings = embedder.encode(chunks).tolist()
        
        # ChromaDB mein daalo
        collection.add(
            documents=chunks,           # asli text
            embeddings=embeddings,      # numbers wala version
            metadatas=[{"source": filename} for _ in chunks],  # kaunsi file se aaya
            ids=[f"chunk_{existing + i}" for i in range(len(chunks))]  # unique ID
        )
        
        total_chunks += len(chunks)
        print(f"  {len(chunks)} chunks load hue: {filename}")
    
    print(f"\nTotal {total_chunks} chunks store ho gaye!\n")


# === STEP 5: Function — sawaal puchho ===
def ask(question):
    """Sawaal lo, milte julte chunks dhoondo, LLM se jawab lo"""
    
    # 1. Sawaal ko bhi numbers mein badlo
    query_embedding = embedder.encode([question]).tolist()
    
    # 2. ChromaDB mein dhoondo — top 3 milte julte chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    chunks = results["documents"][0]      # top 3 chunks ki text
    metadatas = results["metadatas"][0]   # kaunsi file se aaye
    
    # 3. Context banao LLM ke liye
    # [1], [2], [3] laga ke taake LLM cite kar sake
    context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])
    
    # 4. LLM se jawab lo
    response = client.chat.completions.create(
        model="liquid/lfm-2.5-1.2b-instruct:free",
        messages=[
            {
                "role": "system",
                "content": f"""Tum ek document assistant ho.
Sirf neeche diye context se jawab do. Sources [1], [2], [3] cite karo.
Agar answer context mein nahi hai toh kaho: "Mujhe documents mein yeh nahi mila."

CONTEXT:
{context}"""
            },
            {
                "role": "user", 
                "content": question
            }
        ]
    )
    
    # 5. Print karo
    answer = response.choices[0].message.content
    print(f"\nSawaal: {question}")
    print(f"\nJawab: {answer}")
    print("\nSources (kahan se mila):")
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        print(f"  [{i+1}] File: {meta.get('source', '?')} — {chunk[:70]}...")


# === STEP 6: Program chalaao ===
print("Documents load ho rahe hain...\n")
load_folder("docs")   # "docs" folder se files uthao

# Loop — baar baar sawaal puchh sako
while True:
    question = input("\nKoi sawaal puchho (ya 'quit' likho bahar nikalne ke liye): ")
    if question.lower() == "quit":
        print("Khuda hafiz!")
        break
    ask(question)