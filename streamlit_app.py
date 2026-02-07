import streamlit as st
import requests
import json

st.set_page_config(page_title="Jinie HR Bot", layout="centered")
st.title("🤖 Jinie HR Bot - Guardrails Test")

# API Configuration (Matches your app.py settings)
API_URL = "http://127.0.0.1:5000/process_query"
# Use the actual API key from your secrets_manager.py
API_KEY = "geminiS-03072023-secURE#Key123" 
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about HR Policies..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call your Flask API
    with st.chat_message("assistant"):
        try:
            # Matches app.py: data.get('query') and data.get('conversation_id')
            payload = {
                "query": prompt,
                "conversation_id": "streamlit_test_001" 
            }
            
            response = requests.post(API_URL, json=payload, headers=HEADERS)
            
            if response.status_code == 200:
                res_json = response.json()
                
                # Digging into the specific nested structure of your Flask return:
                # return jsonify({'statusCode': 200, 'body': {'response': parsed_response}})
                bot_response_data = res_json.get("body", {}).get("response", {})
                answer = bot_response_data.get("Answer", "I couldn't process that response.")
                urls = bot_response_data.get("urls", [])

                # Display the Answer
                st.markdown(answer)

                # Display Policy URLs if they exist
                if urls:
                    with st.expander("Reference Documents"):
                        for item in urls:
                            for title, url in item.items():
                                st.link_button(f"📄 {title}", url)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"Connection failed: {e}")