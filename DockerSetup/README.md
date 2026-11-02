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
