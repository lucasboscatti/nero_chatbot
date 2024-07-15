import streamlit as st

from libs.inference import *

# Set Streamlit page configuration
st.set_page_config(page_title="Chat With AuRoRa", layout="wide", page_icon="🤖")
st.markdown("# 🤖 AuRoRa Chat")


def reset_conversation():
    st.session_state.messages = []


with st.sidebar:
    st.button("New Chat", on_click=reset_conversation, type="primary")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Faça sua pergunta"):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        response = st.write_stream(chat_answer(question))
        with open("sources.txt", "r") as file:
            index = 1
            st.markdown("Fontes: ")
            for line in file:
                line = line.strip()
                formmated_source = f"[{index}] {line}"
                st.markdown(formmated_source)
                index += 1
                response += "\n" + formmated_source
        st.session_state.messages.append({"role": "assistant", "content": response})
