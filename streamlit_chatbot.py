from typing import Dict, List

import streamlit as st

from libs.inference import chat_answer


def set_page_config():
    st.set_page_config(page_title="Chat With AuRoRa", layout="wide", page_icon="🤖")


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_conversation():
    st.session_state.messages = []


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_sources() -> List[str]:
    try:
        with open("sources.txt", "r") as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        return []


def display_sources(sources: List[str]):
    if sources:
        st.markdown("Sources:")
        for index, source in enumerate(sources, start=1):
            st.markdown(f"\n[{index}] {source}")


def handle_user_input(question: str):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        response = st.write_stream(chat_answer(question, st.session_state.message))

        sources = get_sources()
        display_sources(sources)

        full_response = (
            response + "\n" + "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources))
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def main():
    set_page_config()
    initialize_session_state()

    st.title("🤖 AuRoRa Chat")
    st.caption("💬 Nero Virtual Assistant")

    with st.sidebar:
        st.button("New Chat", on_click=reset_conversation, type="primary")
        st.markdown(
            '<p style="color: #666666; font-size: 12px; text-align: center;">© 2024 Nero. All rights reserved.</p>',
            unsafe_allow_html=True,
        )

    display_chat_history()

    if question := st.chat_input("Ask your question:"):
        with st.chat_message("user"):
            st.markdown(question)
        handle_user_input(question)


if __name__ == "__main__":
    main()
