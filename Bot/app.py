import streamlit as st
import requests
import uuid

# FastAPI backend URL (matches docker-compose service name)
API_URL = "http://rag-agent-api:8000/chat"

st.set_page_title("IT Support RAG Assistant", page_icon="🤖")
st.title("🤖 IT/DevOps Support RAG Assistant")
st.markdown("Ask questions about your enterprise infrastructure runbooks.")

# Initialize session state for chat history and unique session ID
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
if prompt := st.chat_input("Ask a troubleshooting question..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching runbooks & thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "session_id": st.session_state.session_id,
                        "question": prompt
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response received.")
                else:
                    answer = f"Error from server: {response.status_code} - {response.text}"
            except requests.exceptions.ConnectionError:
                answer = "Error: Could not connect to the backend API service. Ensure FastAPI is running."
            except Exception as e:
                answer = f"An unexpected error occurred: {str(e)}"
            
            st.markdown(answer)
            
    # Append assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar controls for clearing session/history
with st.sidebar:
    st.header("Session Management")
    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.text(f"Session ID:\n{st.session_state.session_id}")
