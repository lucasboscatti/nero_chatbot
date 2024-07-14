import streamlit as st

from libs.model_inference import *

# Set Streamlit page configuration
st.set_page_config(page_title="Chat With AuRoRa", layout="wide", page_icon="🤖")


st.markdown("# 🤖 AuRoRa Chat")

def reset_conversation():
    st.session_state.messages = []


with st.sidebar:
    # Create a dropdown box
    language_options = ["English", "Portuguese"]
    article_area = st.selectbox(
        "Select an article area:",
        (
            "All",
            "CA - Controle Aéreo",
            "CT - Controle Terrestre",
            "FA - Formação Aérea",
            "FH - Formação Heterogêneo",
            "FT - Formação Terrestre",
            "HR - Interação Humano-Robô",
            "IA - Inteligência Artificial",
            "RC - Robótica Competições",
            "RE - Robótica Educacional",
        ),
    )

    selected_language = st.selectbox("Select language:", language_options)

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
        try:
            response = st.write_stream(to_sync_generator(process_stream(question)))
        except Exception as e:
            st.markdown("Internal error. Tente novamente mais tarde.")
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
