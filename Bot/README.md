``
## Set up and deploy your multi-agent RAG chatbot in your Docker environment:
​
#### Step 1: Organize Your Project Files
​Ensure your project directory contains all the required files we created:
​main.py (FastAPI backend with LangGraph and session memory)
​app.py (Streamlit frontend)
​requirements.txt (containing langchain-huggingface, faiss-cpu, fastapi, streamlit, etc.)
​Dockerfile
​docker-compose.yml
​.env (containing your GROQ_API_KEY=your_groq_api_key_here)
​A folder named runbooks_pdf/ (with your PDF runbooks inside)
​A folder named faiss_index/ (or it will be generated)
​
#### Step 2: Add a .dockerignore File
​To prevent host artifacts (like local virtual environments) from bloating your build or causing read-only storage errors, create a .dockerignore file in your


#### Step 3: Clean up Existing Docker State (If Needed)
docker-compose down --volumes --remove-orphans

#### Step 4: Build and Launch the Containers
docker-compose up --build

#### Step 5: Access Your Application
​Once the services boot up successfully:
​Streamlit Frontend (User UI): Open your browser at http://localhost:8501
​FastAPI Backend (API Docs): Open your browser at ``
