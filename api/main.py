import os
import streamlit as st
from dotenv import load_dotenv
from api.llm_provider import LLMProvider
from api.shield_net.decorators import shield_net
from api.simple_rag import SimpleRAG
from api.utils import prepare_messages

# Load environment variables
load_dotenv()

st.set_page_config(layout="wide")

# Cache the LLMProvider initialization
@st.cache_resource
def initialize_llm_provider():
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER")
    LLM_NAME = os.environ.get("LLM_NAME")
    return LLMProvider(provider=LLM_PROVIDER, model_name=LLM_NAME)

# Cache the SimpleRAG initialization
@st.cache_resource
def initialize_simple_rag():
    return SimpleRAG()

# Initialize cached resources
llm_provider = initialize_llm_provider()
simple_rag_news_letter = initialize_simple_rag()

# Function to get LLM response
def get_response(system_prompt: str, prompt: str) -> str:
    print(system_prompt)
    print(prompt)
    messages = prepare_messages(system_prompt, prompt)
    return llm_provider.invoke_llm(messages)

system_prompt_post_expert =  "you are expert to generate social media posts"
# Initialize session state variables
if "chat_original_model_history" not in st.session_state:
    st.session_state.chat_original_model_history = []
if "chat_with_shield_net_history" not in st.session_state:
    st.session_state.chat_with_shield_net_history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""  # To manage the input field state
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = system_prompt_post_expert  # To manage the system prompt state

# Set page configuration
st.title("Shield-Net Demo - Protected Poison RAG")

# Sidebar for system prompt configuration
st.sidebar.header("System Prompt Configuration")
system_prompt_input = st.sidebar.text_area(
    "Enter a system prompt (optional):",
    value=st.session_state.system_prompt or system_prompt_post_expert ,
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

    # Get context from SimpleRAG (cached)
    @st.cache_data
    def get_context_from_simple_rag(prompt):
        return simple_rag_news_letter.get_context(prompt)

    context = get_context_from_simple_rag(prompt)
    prompt = f"Context: {context} \n Question: {prompt}"

    # Get responses from the LLM provider
    original_response = get_response(system_prompt=system_prompt, prompt=prompt)

    # Apply Shield-Net decorator to get response
    shield_net_response = shield_net(get_response)(system_prompt=system_prompt, prompt=prompt)

    # Append responses to respective chat histories
    st.session_state.chat_original_model_history.append({"role": "assistant", "content": original_response})
    st.session_state.chat_with_shield_net_history.append({"role": "assistant", "content": shield_net_response})

# Column 1: Chat A
with col1:
    st.header("Poison RAG")
    with st.container():
        for chat in st.session_state.chat_original_model_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

# Column 2: Chat B
with col2:
    st.header("Poison RAG + Shield Net Model Protection")
    with st.container():
        for chat in st.session_state.chat_with_shield_net_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])
