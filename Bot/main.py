import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Any

# Initialize FastAPI app
app = FastAPI(title="IT Support RAG Agent API")

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from environment variables.")

# Initialize LLM & Local Embeddings
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

FAISS_INDEX_PATH = "faiss_index"

def get_vectorstore():
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# Define State with message history support
class AgentState(TypedDict):
    messages: List[Any]
    context: str

# Define Graph Nodes
def retrieve_node(state: AgentState):
    messages = state["messages"]
    latest_question = messages[-1].content if messages else ""
    
    vectorstore = get_vectorstore()
    if not vectorstore:
        return {"context": "No runbooks indexed yet."}
    
    docs = vectorstore.similarity_search(latest_question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generate_node(state: AgentState):
    messages = state["messages"]
    context = state["context"]
    
    # Strict prompt forcing the model to rely only on context or fallback
    system_prompt = (
        "You are an expert IT/DevOps support assistant. "
        "You must answer the user's question using **only** the provided context from the runbooks. "
        "If the answer cannot be found directly within the provided context, you must respond with: "
        "\"I don't have information on that in the runbooks.\"\n\n"
        f"Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    return {"messages": messages + [AIMessage(content=response.content)]}

# Build LangGraph workflow with Memory Checkpointer
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    session_id: str
    answer: str

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        input_state = {
            "messages": [HumanMessage(content=request.question)],
            "context": ""
        }
        
        result = app_graph.invoke(input_state, config=config)
        ai_answer = result["messages"][-1].content
        
        return {
            "session_id": request.session_id,
            "answer": ai_answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
