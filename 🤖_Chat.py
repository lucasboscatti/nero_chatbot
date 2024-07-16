import streamlit as st

from libs.inference import *
from libs.prompt import *

# Set Streamlit page configuration
st.set_page_config(page_title="Chat With AuRoRa", layout="wide", page_icon="🤖")

st.title("🤖 AuRoRa Chat")
st.caption("💬 Nero Virtual Assistant")


def reset_conversation():
    st.session_state.messages = []


with st.sidebar:
    language = st.selectbox("Language", ("English", "Portuguese"))
    st.button("New Chat", on_click=reset_conversation, type="primary")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask your question:"):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        preamble = prompt_en if language == "English" else prompt_pt
        response = st.write_stream(chat_answer(question, preamble))
        with open("sources.txt", "r") as file:
            lines = file.readlines()
        if lines:
            st.markdown("Fontes:")
            for index, line in enumerate(lines, start=1):
                line = line.strip()
                formatted_source = f"\n[{index}] {line}"
                st.markdown(formatted_source)
                response += formatted_source
        st.session_state.messages.append({"role": "assistant", "content": response})
