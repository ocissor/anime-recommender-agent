import streamlit as st
import uuid
import pickle
import requests
from dotenv import load_dotenv
load_dotenv()


API_URL = "http://127.0.0.1:8000/chat"

st.markdown(
    """
    <style>
    /* Target buttons in the sidebar */
    section[data-testid="stSidebar"] button {
        background-color: #262730;        /* Default background */
        color: white;                     /* Default text color */
        border: none;
        border-radius: 0.5rem;
    }

    section[data-testid="stSidebar"] button:hover {
        background-color: #4a4a4a !important;  /* Custom hover color */
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

def stream_graph(payload):
    response = requests.post(API_URL, json=payload, stream=True)

    content_type = response.headers.get("Content-Type", "").lower()

    if "application/json" in content_type:
        data = response.json()
        return data

    buffer = b""
    for chunk in response.iter_content(chunk_size=None):
        buffer += chunk
        try:
            step = pickle.loads(buffer)
            buffer = b""
            yield step
        except (pickle.UnpicklingError, EOFError):
            continue


# Set wide layout and title
st.set_page_config(page_title="Chatbot", layout="wide")
# --- Main Area ---
st.title("💬 Anime Recommender")
st.markdown("🚀 A Streamlit chatbot powered by GROQ")

if 'ui_messages' not in st.session_state:
    st.session_state.ui_messages = []

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


user_input = st.chat_input("Type your message here...", key="chat_input")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.ui_messages.append({"role": "user", "content": user_input})
    payload = {
        'user_input': user_input,
        'thread_id': st.session_state.thread_id
    }

    for step in stream_graph(payload):
        st.chat_message('assistant').markdown(step)
        st.session_state.ui_messages.append({'type':'assistant', 'content':step})




