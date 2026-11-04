import os
import json
import requests
from typing import List, TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

# 1. API Keys & Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "dgsd_76ghgjgh***")
EMBEDDING_API_URL = "https://api.init.asphere.ai/v1/embeddings"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is missing!")

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.1, max_tokens=500, api_key=GROQ_API_KEY)


# 2. Custom Enterprise Embeddings Wrapper (Dedicated class for your company's K8s Endpoints)
class AsphereCustomEmbeddings(Embeddings):
    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings_list = []
        for text in texts:
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "encoding_format": "float",
                "input": text,
                "model": self.model_name,
                "user": "rag-system-service"
            }
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            # OpenAI standard response format: data[0]['embedding']
            embeddings_list.append(data["data"][0]["embedding"])
        return embeddings_list

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Initialize Custom Embeddings
embeddings = AsphereCustomEmbeddings(
    api_url=EMBEDDING_API_URL,
    api_key=EMBEDDING_API_KEY,
    model_name="Qwen/Qwen3-VL-Embedding-8B"
)


# 3. FAISS Vector DB Setup with Persistence Check
pdf_folder_path = "./runbooks_pdf"
faiss_index_path = "./faiss_index"
os.makedirs(pdf_folder_path, exist_ok=True)
os.makedirs(faiss_index_path, exist_ok=True)

if os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
    print("📂 Loading saved FAISS index from disk...")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("⚙️ Creating new FAISS index (via Enterprise Embeddings)...")
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    raw_documents = loader.load()
    
    if len(raw_documents) > 0:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(raw_documents)
    else:
        docs = [Document(page_content="No PDF runbooks found.", metadata={"source": "empty"})]
        
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)
    print("✅ FAISS index successfully saved!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# 4. State Definition
class RAGAgentState(TypedDict):
    query: str
    retrieved_docs: str
    solution: str
    review_comment: str
    approval_status: str
    iteration_count: int


# 5. Nodes (Agents)
def rag_search_node(state: RAGAgentState):
    query = state["query"]
    retrieved = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved])
    return {"retrieved_docs": context}


def draft_solution_node(state: RAGAgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    review = state.get("review_comment", "")
    count = state.get("iteration_count", 0)
    
    prompt = f"Problem: {query}\nDocument: {docs}\nWrite a secure solution."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"solution": response.content, "iteration_count": count + 1}


def security_review_node(state: RAGAgentState):
    solution = state["solution"]
    prompt = f"Check the following solution. If it is safe, reply with 'APPROVED':\n\n{solution}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"review_comment": response.content}


def admin_approval_node(state: RAGAgentState):
    return {"approval_status": "WAITING_FOR_ADMIN"}


# 6. Conditional Edge Logic
def evaluate_review(state: RAGAgentState):
    review = state["review_comment"]
    count = state["iteration_count"]
    if "APPROVED" in review.upper() or count >= 2:
        return "proceed_to_approval"
    return "rewrite"


# 7. Workflow Compilation
workflow = StateGraph(RAGAgentState)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("draft_solution", draft_solution_node)
workflow.add_node("security_review", security_review_node)
workflow.add_node("admin_approval", admin_approval_node)

workflow.add_edge(START, "rag_search")
workflow.add_edge("rag_search", "draft_solution")
workflow.add_edge("draft_solution", "security_review")

workflow.add_conditional_edges(
    "security_review",
    evaluate_review,
    {"rewrite": "draft_solution", "proceed_to_approval": "admin_approval"}
)

workflow.add_edge("admin_approval", END)

memory = MemorySaver()
rag_multi_agent_app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["admin_approval"]
)


# 8. FastAPI Setup
app = FastAPI(
    title="Enterprise Multi-Agent RAG API",
    description="Custom Enterprise Embeddings, FAISS, Streaming & Human Approval",
    version="6.0"
)

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "prod_session_1"

class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


# 9. Streaming Endpoint (SSE)
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def event_generator():
        initial_input = {
            "query": request.query,
            "retrieved_docs": "",
            "solution": "",
            "review_comment": "",
            "approval_status": "PENDING",
            "iteration_count": 0
        }
        config = {"configurable": {"thread_id": request.thread_id}}
        
        async for event in rag_multi_agent_app.astream(initial_input, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                yield f"data: {json.dumps({'node': node_name, 'output': node_output}, ensure_ascii=False)}\n\n"
        
        current_state = rag_multi_agent_app.get_state(config)
        if current_state.next:
            yield f"data: {json.dumps({'status': 'PAUSED_FOR_APPROVAL', 'message': 'Waiting for admin approval.'}, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'COMPLETED', 'message': 'Task completed successfully.'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 10. Approval Endpoint
@app.post("/approve")
async def approve_endpoint(request: ApprovalRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        current_state = rag_multi_agent_app.get_state(config)
        
        if not current_state.next:
            return {"message": "No pending tasks waiting for approval in this thread."}
            
        if request.approved:
            output = rag_multi_agent_app.invoke(None, config=config)
            return {
                "status": "APPROVED_AND_COMPLETED",
                "message": "Admin approved. Solution implemented!",
                "solution": current_state.values.get("solution")
            }
        else:
            return {"status": "REJECTED", "message": "Solution rejected."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
