# Production-Grade Multi-Agent RAG System for DevOps Support

An enterprise-ready, containerized Multi-Agent Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, and **LangChain**. Powered by **Qwen 27B** (via Groq API), this system ingests internal IT PDF runbooks, provides autonomous problem resolution through dynamic cyclic workflows, enforces security reviews, supports real-time streaming via Server-Sent Events (SSE), and incorporates Human-in-the-Loop (HITL) approval before executing changes.

---

## 🌟 Key Features

* **Multi-Agent Orchestration**: Modular workflow managed by **LangGraph**:
  * **RAG Search Agent**: Retrieves relevant context from indexed PDF runbooks.
  * **Drafting Agent**: Generates actionable, step-by-step resolution plans.
  * **Security Reviewer Agent**: Evaluates proposed solutions against enterprise security standards.
* **Human-in-the-Loop (HITL)**: Interrupts state execution before final admin approval using dynamic LangGraph state interrupts.
* **Persistent State & Memory**: Uses LangGraph `MemorySaver` checkpointers to maintain multi-turn context per `thread_id`.
* **Fast Vector Indexing**: Local vector search powered by **FAISS** with automatic disk persistence (`save_local`/`load_local`) to avoid re-indexing on container restarts.
* **Real-time SSE Streaming**: Exposes `/chat/stream` endpoints via FastAPI `StreamingResponse` for live updates on agent status.
* **Containerized Deployment**: Fully Dockerized with persistent volume mappings for both PDF documents and vector storage.

---

## 🛠️ Tech Stack

* **Backend Framework**: FastAPI, Uvicorn
* **Agent Graph & Workflow**: LangGraph, LangChain
* **LLM Orchestration**: Groq API (`qwen/qwen3.8-27b`)
* **Embeddings & Vector Store**: HuggingFace (`all-MiniLM-L6-v2`), FAISS (CPU)
* **Document Processing**: PyPDF (`PyPDFDirectoryLoader`), RecursiveCharacterTextSplitter
* **Containerization**: Docker, Docker Compose

---

## 📁 Repository Structure

```text
.
├── main.py              # Core application entrypoint (FastAPI + LangGraph)
├── Dockerfile           # Python 3.9 container definition
├── docker-compose.yml   # Multi-container orchestration and volume management
├── requirements.txt     # Python dependency lockfile
├── .env                 # Environment secrets file
├── runbooks_pdf/        # Volume directory for source PDF runbooks
└── faiss_index/         # Persisted local vector store data
```

---

## 🚀 Quickstart Guide

### Prerequisites

* [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.
* A valid **Groq API Key**.

### 1. Project Configuration

Clone the repository and set up your environment variables:

```bash
# Create the environment file
echo "GROQ_API_KEY=gsk_your_groq_api_key_here" > .env
```

Ensure your source PDF runbooks are placed inside the `runbooks_pdf` folder:

```bash
mkdir -p runbooks_pdf faiss_index
# Add your PDF files into ./runbooks_pdf/
```

### 2. Launch with Docker Compose

Build and run the containerized application:

```bash
docker-compose up --build
```

The FastAPI application will start at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

---

## 🔗 API Endpoints & Workflows

### 1. Real-time Streaming (`POST /chat/stream`)

Streams agent updates (search $\rightarrow$ drafting $\rightarrow$ security review) using Server-Sent Events (SSE). Pauses when human approval is required.

**Request:**
```bash
curl -N -X POST "http://localhost:8000/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How do I fix a Kubernetes OOMKilled error?",
       "thread_id": "session_001"
     }'
```

**Response (SSE Stream):**
```text
data: {"node": "rag_search", "output": {...}}

data: {"node": "draft_solution", "output": {...}}

data: {"node": "security_review", "output": {...}}

data: {"status": "PAUSED_FOR_APPROVAL", "message": "Pending admin review."}
```

---

### 2. Human Approval (`POST /approve`)

Resumes graph execution once an administrator evaluates the proposed solution.

**Request (Approval):**
```bash
curl -X POST "http://localhost:8000/approve" \
     -H "Content-Type: application/json" \
     -d '{
       "thread_id": "session_001",
       "approved": true
     }'
```

**Response:**
```json
{
  "status": "APPROVED_AND_COMPLETED",
  "message": "Admin approved the resolution plan.",
  "solution": "1. Inspect pod resource limits...\n2. Increase memory requests..."
}
```

---

## 🛡️ Operational Workflow Diagram

```text
[User Query] 
     │
     ▼
┌──────────────┐
│  RAG Search  │ ◄── (Retrieves context from persisted FAISS vector index)
└──────┬───────┘
       │
       ▼
┌──────────────┐     Review Failed / Rewrite Needed
│Draft Solution│ ◄────────────────────────────────────┐
└──────┬───────┘                                      │
       │                                              │
       ▼                                              │
┌─────────────────┐                                   │
│ Security Review ├───────────────────────────────────┘
└──────┬──────────┘  Review Approved
       │
       ▼
┌──────────────────┐
│  Admin Approval  │ (Interrupt Node: Pauses execution)
└──────┬───────────┘
       │
       ├───────────────► Approved   ──────► [Execute & END]
       └───────────────► Rejected   ──────► [Abort]
```

---


## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

 ✅ User Registration
 ✅ User Login (JWT)
 ✅ Password Hashing (bcrypt)
 ✅ User-specific Chat Sessions
 ✅ LangGraph Memory
 ✅ PostgreSQL Chat History
 ✅ RAG Search from PDF Runbooks
 ✅ FastAPI Backend

# Steps 

Setup Checklist (What we have):

main.py: The main file integrating FastAPI, LangGraph (Search, Drafting, Security Review, Admin Approval), Checkpointing (MemorySaver), SSE Streaming, and the Qwen 27B LLM.

Dockerfile: To build the container with the Python environment and dependencies.

requirements.txt: A list of all required Python packages including FastAPI, LangChain, LangGraph, FAISS, and PyPDF.

docker-compose.yml: To manage the container and mount the local PDF folder (runbooks_pdf) using Volume Mounting.

runbooks_pdf/: The local folder where you need to place your PDF runbooks.

# docker compose run
Make sure you have main.py (FastAPI) and app.py (Streamlit) in your project root.

docker-compose up --build

# URL
http://localhost:8000
