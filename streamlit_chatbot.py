from typing import Dict, List

import streamlit as st

from libs.inference import chat_answer


def set_page_config():
    st.set_page_config(page_title="Chat With AuRoRa", layout="wide", page_icon="🤖")


def new_chat():
    st.session_state.messages = []


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_sources(sources: List[str]):
    if sources:
        st.markdown("Sources:")
        for index, source in enumerate(sources, start=1):
            st.markdown(f"\n[{index}] {source}")


def handle_user_input(question: str):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        response = st.write_stream(chat_answer(question, st.session_state.messages))

        with open("sources.txt", "r") as file:
            sources = [line.strip() for line in file.readlines()]

            source_response = ""
            if sources:
                source_response += "\nSources:\n"
                for index, source in enumerate(sources, start=1):
                    source_response += f"\n[{index}] {source}"

                st.markdown(source_response)
                response += "\n" + source_response

        st.session_state.messages.append({"role": "assistant", "content": response})


def set_sidebar_footer():
    # CSS for styling the sidebar footer
    css = """
    <style>
    .sidebar-footer {
        margin-top: 600px; /* Adjust the margin as needed */
        display: flex;
        justify-content: center;
        gap: 10px;
    }
    .sidebar-footer img {
        width: 24px;
        height: 24px;
    }
    .copyright {
        color: #666666;
        font-size: 12px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """

    social_links = """
    <div class="sidebar-footer">
        <a href="https://www.instagram.com/roboticaufv/" target="_blank">
            <img src="https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Instagram_colored_svg_1-512.png">
        </a>
        <a href="https://www.facebook.com/roboticaUFV" target="_blank">
            <img src="https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Facebook_colored_svg_copy-512.png">
        </a>
        <a href="https://www.linkedin.com/company/roboticaufv/" target="_blank">
            <img src="https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-512.png">
        </a>
        <a href="https://www.youtube.com/c/roboticaufv" target="_blank">
            <img src="https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Youtube_colored_svg-512.png">
        </a>
    </div>
    """

    copyright_text = """
    <p class="copyright">
        <a href="https://nero-chatbot-add-paper-area.streamlit.app/" target="_blank" style="color: #666666; text-decoration: none;">
            © 2024 Núcleo de Especialização em Robótica - UFV
        </a>
    </p>
    """

    st.sidebar.markdown(css + social_links, unsafe_allow_html=True)
    st.sidebar.markdown(copyright_text, unsafe_allow_html=True)


def main():
    set_page_config()
    initialize_session_state()

    st.title("🤖 AuRoRa Chat")
    st.caption("💬 Nero Virtual Assistant")

    with st.sidebar:
        st.button("New Chat", on_click=new_chat, type="primary")
        set_sidebar_footer()

    display_chat_history()

    if question := st.chat_input("Ask your question:"):
        with st.chat_message("user"):
            st.markdown(question)
        handle_user_input(question)


if __name__ == "__main__":
    main()
