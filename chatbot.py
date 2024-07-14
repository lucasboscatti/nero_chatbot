import streamlit as st

from libs.model_inference import process_stream, to_sync_generator

# st.title("🤖 Converse com a AuRoRa")


def chatbot_page():
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

        st.session_state.messages.append({"role": "assistant", "content": response})
