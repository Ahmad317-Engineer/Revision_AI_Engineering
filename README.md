# RAG Document Assistant

AI-powered document Q&A system built with FastAPI + ChromaDB + Streamlit.

## Live Demo
https://revisionaiengineering-production.up.railway.app/docs

## Stack
- FastAPI (backend)
- ChromaDB (vector database)
- SentenceTransformers (embeddings)
- Streamlit (frontend)
- OpenRouter (LLM API)
- Railway (deployment)

## How to run locally
uvicorn backend:app --reload
streamlit run frontend.py