# New Workflow with Approval
import os
from typing import TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. API Keys & LLM/Embeddings Initialization
os.environ["GROQ_API_KEY"] = "xxxxxx"

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.1, max_tokens=500)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Read PDF files and create FAISS Vector DB
pdf_folder_path = "./runbooks_pdf"
os.makedirs(pdf_folder_path, exist_ok=True)

loader = PyPDFDirectoryLoader(pdf_folder_path)
raw_documents = loader.load()

if len(raw_documents) > 0:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(raw_documents)
else:
    docs = [Document(page_content="No PDF runbooks found.", metadata={"source": "empty"})]

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# 3. State
class RAGAgentState(TypedDict):
    query: str
    retrieved_docs: str
    solution: str
    review_comment: str
    approval_status: str
    iteration_count: int


# 4. Nodes (Agents)
def rag_search_node(state: RAGAgentState):
    query = state["query"]
    print(f"\n🔍 [RAG Search Agent]: Searching for information related to '{query}'...")
    retrieved = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved])
    return {"retrieved_docs": context}


def draft_solution_node(state: RAGAgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    review = state.get("review_comment", "")
    count = state.get("iteration_count", 0)
    
    print(f"\n📝 [Drafting Agent]: Drafting solution (Iteration: {count + 1})...")
    prompt = f"Problem: {query}\nDocument: {docs}\nWrite a secure solution."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"solution": response.content, "iteration_count": count + 1}


def security_review_node(state: RAGAgentState):
    solution = state["solution"]
    print("\n🛡️ [Security Reviewer Agent]: Evaluating the solution...")
    prompt = f"Check the following solution. If it is safe, reply with 'APPROVED':\n\n{solution}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"review_comment": response.content}


def admin_approval_node(state: RAGAgentState):
    print("\n⏳ [Admin Approval Node]: Waiting for human approval...")
    return {"approval_status": "WAITING_FOR_ADMIN"}


# 5. Conditional Edge
def evaluate_review(state: RAGAgentState):
    review = state["review_comment"]
    count = state["iteration_count"]
    if "APPROVED" in review.upper() or count >= 2:
        return "proceed_to_approval"
    return "rewrite"


# 6. Workflow Compilation with Interrupt & Checkpointer
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


# 7. FastAPI Setup
app = FastAPI(
    title="Qwen Multi-Agent RAG API with Human Approval",
    description="IT agent application powered by Qwen 3.8-27b and LangGraph",
    version="3.1"
)

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "admin_session_1"

class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


# 8. Endpoint 1: Send Query (Runs until Interrupt)
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        initial_input = {
            "query": request.query,
            "retrieved_docs": "",
            "solution": "",
            "review_comment": "",
            "approval_status": "PENDING",
            "iteration_count": 0
        }
        config = {"configurable": {"thread_id": request.thread_id}}
        
        output = rag_multi_agent_app.invoke(initial_input, config=config)
        current_state = rag_multi_agent_app.get_state(config)
        
        # Clean up newline representations if present in solution
        solution_text = current_state.values.get("solution")
        if solution_text:
            solution_text = solution_text.replace("\\n", "\n")

        if current_state.next:
            return {
                "status": "PAUSED_FOR_APPROVAL",
                "message": "Solution is ready. Waiting for admin approval.",
                "thread_id": request.thread_id,
                "solution": solution_text,
                "review_comment": current_state.values.get("review_comment")
            }
        else:
            return {"status": "COMPLETED", "output": current_state.values}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 9. Endpoint 2: Human Approval (Resumes Graph)
@app.post("/approve")
async def approve_endpoint(request: ApprovalRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        current_state = rag_multi_agent_app.get_state(config)
        
        if not current_state.next:
            return {"message": "No pending tasks waiting for approval in this thread."}
            
        if request.approved:
            output = rag_multi_agent_app.invoke(None, config=config)
            solution_text = current_state.values.get("solution")
            if solution_text:
                solution_text = solution_text.replace("\\n", "\n")
                
            return {
                "status": "APPROVED_AND_COMPLETED",
                "message": "Admin approved. Solution successfully implemented!",
                "solution": solution_text
            }
        else:
            return {
                "status": "REJECTED",
                "message": "Admin rejected this solution."
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
