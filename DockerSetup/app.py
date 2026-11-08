import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="IT Support RAG",
    layout="wide"
)

st.title("🛠 IT Support Multi-Agent RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user1"

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question"):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    response = requests.post(
        BACKEND_URL,
        json={
            "query": prompt,
            "thread_id": st.session_state.thread_id
        }
    )

    if response.status_code == 200:

        answer = response.json()["solution"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
