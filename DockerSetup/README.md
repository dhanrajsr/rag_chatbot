# Production-Grade Multi-Agent RAG System for DevOps Support
​An enterprise-ready, containerized Multi-Agent Retrieval-Augmented Generation (RAG) system built with FastAPI, LangGraph, and LangChain. Powered by Qwen 27B (via Groq API), this system ingests internal IT PDF runbooks, provides autonomous problem resolution through dynamic cyclic workflows, enforces security reviews, supports real-time streaming via Server-Sent Events (SSE), and incorporates Human-in-the-Loop (HITL) approval before executing changes.
# Key Features
## Multi-Agent Orchestration: 
### Modular workflow managed by LangGraph:
#### ​RAG Search Agent: Retrieves relevant context from indexed PDF runbooks.
#### ​Drafting Agent: Generates actionable, step-by-step resolution plans.
#### ​Security Reviewer Agent: Evaluates proposed solutions against enterprise security standards.
#### ​Human-in-the-Loop (HITL): Interrupts state execution before final admin approval using dynamic LangGraph state interrupts.
#### ​Persistent State & Memory: Uses LangGraph MemorySaver checkpointers to maintain multi-turn context per thread_id.
#### ​Fast Vector Indexing: Local vector search powered by FAISS with automatic disk persistence (save_local/load_local) to avoid re-indexing on container restarts.
#### ​Real-time SSE Streaming: Exposes /chat/stream endpoints via FastAPI StreamingResponse for live updates on agent status.
#### ​Containerized Deployment: Fully Dockerized with persistent volume mappings for both PDF docum.ents and vector storage.

​# Tech Stack
### ​Backend Framework: FastAPI, Uvicorn
​### Agent Graph & Workflow: LangGraph, LangChain
​### LLM Orchestration: Groq API (qwen/qwen3.8-27b)
​### .Embeddings & Vector Store: HuggingFace (all-MiniLM-L6-v2), FAISS (CPU)
### ​Document Processing: PyPDF (PyPDFDirectoryLoader), RecursiveCharacterTextSplitter
​### Containerization: Docker, Docker Compose

# Repository Structure



# Steps 


Setup Checklist (What we have):

main.py: The main file integrating FastAPI, LangGraph (Search, Drafting, Security Review, Admin Approval), Checkpointing (MemorySaver), SSE Streaming, and the Qwen 27B LLM.

Dockerfile: To build the container with the Python environment and dependencies.

requirements.txt: A list of all required Python packages including FastAPI, LangChain, LangGraph, FAISS, and PyPDF.

docker-compose.yml: To manage the container and mount the local PDF folder (runbooks_pdf) using Volume Mounting.

runbooks_pdf/: The local folder where you need to place your PDF runbooks.

# docker compose run
docker-compose up --build

# URL
http://localhost:8000
