# . Checkpointing & State Persistence (நினைவகப் பாதுகாப்பு & தொடர்ச்சி)

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
from langgraph.checkpoint.memory import MemorySaver  # Checkpoint memory import

# 1. API Keys & LLM/Embeddings Initialization
os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. PDF ஃபைல்களைப் படித்து FAISS Vector DB உருவாக்குதல்
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
    iteration_count: int


# 4. Nodes (Agents)
def rag_search_node(state: RAGAgentState):
    query = state["query"]
    print(f"\n🔍 [RAG Search Agent]: '{query}' தொடர்பான தகவல்களைத் தேடுகிறது...")
    retrieved = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved])
    return {"retrieved_docs": context}


def draft_solution_node(state: RAGAgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    review = state.get("review_comment", "")
    count = state.get("iteration_count", 0)
    
    print(f"\n📝 [Drafting Agent]: தீர்வை உருவாக்குகிறது (சுற்று: {count + 1})...")
    
    if review:
        prompt = f"பிரச்சனை: {query}\nடாக்குமெண்ட்: {docs}\nமறுஆய்வு கருத்து: {review}\nபாதுகாப்பான புதிய தீர்வை எழுதவும்."
    else:
        prompt = f"பிரச்சனை: {query}\nகீழ்க்காணும் டாக்குமெண்ட் தகவலைப் பயன்படுத்தி தீர்வை எழுதவும்:\n{docs}"
        
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"solution": response.content, "iteration_count": count + 1}


def security_review_node(state: RAGAgentState):
    solution = state["solution"]
    print("\n🛡️ [Security Reviewer Agent]: தீர்வை மதிப்பீடு செய்கிறது...")
    
    prompt = f"கீழ்க்கண்ட தீர்வைச் சரிபார்க்கவும். பாதுகாப்பாக இருந்தால் 'APPROVED' எனக் கூறவும். இல்லையெனில் திருத்தத்தைக் கூறவும்:\n\n{solution}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"review_comment": response.content}


# 5. Conditional Edge
def evaluate_review(state: RAGAgentState):
    review = state["review_comment"]
    count = state["iteration_count"]
    
    if "APPROVED" in review.upper() or count >= 2:
        return "finish"
    return "rewrite"


# 6. Workflow Compilation with Checkpointer (MEMORY SAVER)
workflow = StateGraph(RAGAgentState)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("draft_solution", draft_solution_node)
workflow.add_node("security_review", security_review_node)

workflow.add_edge(START, "rag_search")
workflow.add_edge("rag_search", "draft_solution")
workflow.add_edge.add_edge if hasattr(workflow.add_edge, 'add_edge') else None # syntax placeholder
workflow.add_edge("draft_solution", "security_review")

workflow.add_conditional_edges(
    "security_review",
    evaluate_review,
    {"rewrite": "draft_solution", "finish": END}
)

# 💡 MemorySaver-ஐ இணைத்தல் (இதுதான் Checkpointing செய்கிறது)
memory = MemorySaver()
rag_multi_agent_app = workflow.compile(checkpointer=memory)


# 7. FastAPI Setup
app = FastAPI(
    title="Checkpoint-enabled IT Support Multi-Agent RAG API",
    description="LangGraph Checkpointer உடன் இயக்கப்படும் ஐடி ஏஜெண்ட் செயலி",
    version="2.0"
)

# Request & Response Models with thread_id
class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default_user_session"  # ஒவ்வொரு செஷனுக்கும் தனித்தனி ID

class ChatResponse(BaseModel):
    query: str
    thread_id: str
    solution: str
    review_comment: str
    iteration_count: int


# 8. POST /chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        initial_input = {
            "query": request.query,
            "retrieved_docs": "",
            "solution": "",
            "review_comment": "",
            "iteration_count": 0
        }
        
        # 💡 thread_id அடிப்படையிலான Config உருவாக்கம்
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # LangGraph செயலியை config உடன் இயக்குதல்
        output = rag_multi_agent_app.invoke(initial_input, config=config)
        
        return {
            "query": request.query,
            "thread_id": request.thread_id,
            "solution": output["solution"],
            "review_comment": output["review_comment"],
            "iteration_count": output["iteration_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
