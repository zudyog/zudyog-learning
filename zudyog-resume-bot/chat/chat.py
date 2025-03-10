import streamlit as st
import requests
import uuid
import os

API_URL = os.getenv('API_URL', 'http://localhost:8101')

# Initialize session state variables
if 'conversation_id' not in st.session_state:
    st.session_state['conversation_id'] = str(uuid.uuid4())
if 'history' not in st.session_state:
    st.session_state['history'] = []


def get_bot_response(user_input):
    print(st.session_state['conversation_id'])
    payload = {
        "text": user_input,
        "temperature": 0.0,
        "threshold": 0.3,
        "conversation_id": st.session_state['conversation_id']
    }
    response = requests.post(
        f"{API_URL}/query_index", json=payload)
    if response.status_code == 200:
        data = response.json()
        bot_response = data['response']
        st.session_state['conversation_id'] = data.get(
            'conversation_id', st.session_state['conversation_id'])
        return bot_response
    else:
        return "Error: API request failed."


st.title("🤖 Chat with rapt-AI Bot")

# Display the chat messages
for chat in st.session_state['history']:
    with st.chat_message("user"):
        st.write(chat['user'])
    with st.chat_message("assistant"):
        st.write(chat['bot'])

# Accept user input
if prompt := st.chat_input("Type your message"):
    # Get bot response
    bot_response = get_bot_response(prompt)
    # Add to history
    st.session_state['history'].append({'user': prompt, 'bot': bot_response})
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    # Display bot message
    with st.chat_message("assistant"):
        st.write(bot_response)
