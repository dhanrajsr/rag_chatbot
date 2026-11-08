import os
from typing import TypedDict, Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ==========================================================
# LLM & Embeddings
# ==========================================================

os.environ["GROQ_API_KEY"] = "YOUR_GROQ_KEY"

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.1,
    max_tokens=500
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ==========================================================
# Load PDFs
# ==========================================================

pdf_folder_path = "./runbooks_pdf"

os.makedirs(pdf_folder_path, exist_ok=True)

loader = PyPDFDirectoryLoader(pdf_folder_path)

raw_documents = loader.load()

if raw_documents:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(raw_documents)

else:
    docs = [
        Document(
            page_content="No runbooks found.",
            metadata={"source": "empty"}
        )
    ]

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ==========================================================
# State Definition
# ==========================================================

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_docs: str
    solution: str
    review_comment: str
    iteration_count: int

# ==========================================================
# Node 1 - RAG Search
# ==========================================================

def rag_search_node(state: RAGState):

    latest_question = None

    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_question = msg.content
            break

    docs = retriever.invoke(latest_question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "retrieved_docs": context
    }

# ==========================================================
# Node 2 - Draft Solution
# ==========================================================

def draft_solution_node(state: RAGState):

    count = state.get("iteration_count", 0)

    messages = state["messages"]

    docs = state["retrieved_docs"]

    review = state.get("review_comment", "")

    conversation = "\n".join(
        [
            f"{type(m).__name__}: {m.content}"
            for m in messages[-10:]
        ]
    )

    if review:

        prompt = f"""
Conversation History:
{conversation}

Runbook Context:
{docs}

Security Feedback:
{review}

Provide an improved answer.
"""

    else:

        prompt = f"""
Conversation History:
{conversation}

Runbook Context:
{docs}

Answer the user's latest question using
conversation context and runbook data.
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "solution": response.content,
        "iteration_count": count + 1,
        "messages": [
            AIMessage(content=response.content)
        ]
    }

# ==========================================================
# Node 3 - Security Review
# ==========================================================

def security_review_node(state: RAGState):

    solution = state["solution"]

    review_prompt = f"""
Check this answer.

If safe and correct reply exactly:

APPROVED

Otherwise provide corrections.

Answer:

{solution}
"""

    response = llm.invoke(
        [HumanMessage(content=review_prompt)]
    )

    return {
        "review_comment": response.content
    }

# ==========================================================
# Decision Logic
# ==========================================================

def evaluate_review(state: RAGState):

    review = state["review_comment"]

    count = state["iteration_count"]

    if "APPROVED" in review.upper():
        return "finish"

    if count >= 2:
        return "finish"

    return "rewrite"

# ==========================================================
# Build Graph
# ==========================================================

workflow = StateGraph(RAGState)

workflow.add_node(
    "rag_search",
    rag_search_node
)

workflow.add_node(
    "draft_solution",
    draft_solution_node
)

workflow.add_node(
    "security_review",
    security_review_node
)

workflow.add_edge(
    START,
    "rag_search"
)

workflow.add_edge(
    "rag_search",
    "draft_solution"
)

workflow.add_edge(
    "draft_solution",
    "security_review"
)

workflow.add_conditional_edges(
    "security_review",
    evaluate_review,
    {
        "rewrite": "draft_solution",
        "finish": END
    }
)

memory = MemorySaver()

rag_app = workflow.compile(
    checkpointer=memory
)

# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(
    title="RAG Memory Chatbot"
)

class ChatRequest(BaseModel):
    query: str
    thread_id: str

class ChatResponse(BaseModel):
    query: str
    thread_id: str
    solution: str
    review_comment: str
    iteration_count: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:

        config = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }

        result = rag_app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=request.query
                    )
                ],
                "retrieved_docs": "",
                "solution": "",
                "review_comment": "",
                "iteration_count": 0
            },
            config=config
        )

        return {
            "query": request.query,
            "thread_id": request.thread_id,
            "solution": result["solution"],
            "review_comment": result["review_comment"],
            "iteration_count": result["iteration_count"]
        }

    except Exception as ex:

        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )
