import streamlit as st
import requests

st.title("Jinie HR Bot - Guardrails Test")

# API Configuration (Match your app.py settings)
API_URL = "http://127.0.0.1:5000/process_query"
# Replace with the API_KEY from your .env or Secrets Manager
HEADERS = {"x-api-key": "your_api_key_here"}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about HR Policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call your Flask API
    with st.chat_message("assistant"):
        try:
            payload = {"conv_id": "test_session", "query": prompt}
            response = requests.post(API_URL, json=payload, headers=HEADERS)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")