import os
import streamlit as st
from dotenv import load_dotenv
from api.llm_provider import LLMProvider
from api.shield_net.decorators import shield_net
from api.simple_rag import SimpleRAG
from api.utils import prepare_messages

load_dotenv()


LLM_PROVIDER = os.environ.get("LLM_PROVIDER")
LLM_NAME = os.environ.get("LLM_NAME")

llm_provider = LLMProvider(provider=LLM_PROVIDER, model_name=LLM_NAME)

simple_rag_news_letter = SimpleRAG()

def get_response(system_prompt:str,prompt:str) -> str:
    messages = prepare_messages(system_prompt, prompt)
    return llm_provider.invoke_llm(messages)


# Initialize session state for chat histories and input
if "chat_original_model_history" not in st.session_state:
    st.session_state.chat_original_model_history = []
if "chat_with_shield_net_history" not in st.session_state:
    st.session_state.chat_with_shield_net_history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""  # To manage the input field state
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""  # To manage the system prompt state

st.set_page_config(layout="wide")
# Add title
st.title("Shield-Net Demo - Protected Poison RAG")

# Sidebar for system prompt
st.sidebar.header("System Prompt Configuration")
system_prompt_input = st.sidebar.text_area(
    "Enter a system prompt (optional):",
    value=st.session_state.system_prompt,
    placeholder="Add your system-level prompt here..."
)
if system_prompt_input:
    st.session_state.system_prompt = system_prompt_input

# Sidebar buttons for Clean and Restart
if st.sidebar.button("Clean Chat History"):
    st.session_state.chat_original_model_history = []
    st.session_state.chat_with_shield_net_history = []
    st.success("Chat history cleared!")

if st.sidebar.button("Restart Session"):
    st.session_state.chat_original_model_history = []
    st.session_state.chat_with_shield_net_history = []
    st.session_state.system_prompt = ""
    st.session_state.input_text = ""
    st.success("Session restarted!")

# Display chat interfaces in two columns
col1, col2 = st.columns(2)

# User input tied to session state
prompt = st.chat_input("Say something")

if prompt:  # When the user enters a message
    # Append user message to both chat histories
    st.session_state.chat_original_model_history.append({"role": "user", "content": prompt})
    st.session_state.chat_with_shield_net_history.append({"role": "user", "content": prompt})

    # If a system prompt exists, prepend it to the conversation
    if st.session_state.system_prompt:
        system_prompt = st.session_state.system_prompt.strip()
    else:
        system_prompt = None

    context = simple_rag_news_letter.get_context(prompt)
    print(context)
    prompt = f"{context} \n {prompt}"

    # Get responses from the LLM provider
    original_response = get_response(system_prompt=system_prompt,prompt=prompt)

    ## apply a decorator on the get response
    shield_net_response = shield_net(get_response)(system_prompt=system_prompt,prompt=prompt)

    # Append responses to respective chat histories
    st.session_state.chat_original_model_history.append({"role": "assistant", "content": original_response})
    st.session_state.chat_with_shield_net_history.append({"role": "assistant", "content": shield_net_response})

# Column 1: Chat A
with col1:
    st.header("Poison RAG")
    with st.container(height=500,border=True):
        chat_area = st.empty()
        chat_area.markdown('<div class="chat-column">', unsafe_allow_html=True)
        for chat in st.session_state.chat_original_model_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])
        chat_area.markdown('</div>', unsafe_allow_html=True)

# Column 2: Chat B
with col2:
    st.header("Poison RAG + Shield Net Model Protection")
    with st.container(height=500,border=True):
        chat_area = st.empty()
        chat_area.markdown('<div class="chat-column">', unsafe_allow_html=True)
        for chat in st.session_state.chat_with_shield_net_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])
        chat_area.markdown('</div>', unsafe_allow_html=True)