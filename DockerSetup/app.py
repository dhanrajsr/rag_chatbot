import streamlit as st
import requests

st.set_page_config(page_title="IT Support RAG Agent", layout="centered")

st.title("🛠️ IT Support Multi-Agent RAG System")
st.write("Ask your infrastructure and runbook questions below:")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_1"

# Display prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input text box
if prompt := st.chat_input("How do I fix a Kubernetes OOMKilled error?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Agents are thinking and searching runbooks...")
        
        try:
            # Call standard non-streaming /chat endpoint matching Swagger
            response = requests.post(
                "http://localhost:8000/chat",
                json={"query": prompt, "thread_id": st.session_state.thread_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                solution = data.get("solution", "No solution returned.")
                review = data.get("review_comment", "")
                
                full_response = f"**Solution:**\n{solution}\n\n*Review Status:* {review}"
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Error from backend: {response.text}")
                
        except Exception as e:
            st.error(f"Failed to connect to FastAPI backend: {e}")

# Admin Approval Sidebar Controls
with st.sidebar:
    st.header("⚙️ Admin Controls")
    thread_input = st.text_input("Thread ID", value=st.session_state.thread_id)
    if st.button("Approve Pending Solution"):
        res = requests.post("http://localhost:8000/approve", json={"thread_id": thread_input, "approved": True})
        st.success(res.json().get("message", "Approved!"))
        if "solution" in res.json():
            st.write(res.json().get("solution"))
