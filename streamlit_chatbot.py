from typing import Dict, List

import streamlit as st

from libs.inference import chat_answer


def set_page_config():
    st.set_page_config(
        page_title="Chat With AuRoRa",
        layout="wide",
        page_icon="assets/nero_logo.png",
    )


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


def handle_user_input(question: str, research_area: str):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []

        for chunk in chat_answer(question, st.session_state.messages, research_area):
            if isinstance(chunk, list):  # This is the list of sources
                sources = chunk
            else:  # This is a text chunk
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        if sources:
            source_response = "\nSources:\n"
            for index, source in enumerate(sources, start=1):
                source_response += f"\n[{index}] {source}"
            st.markdown(source_response)
            full_response += "\n" + source_response

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def set_sidebar_text():
    # CSS for styling the sidebar footer
    css = """
    <style>
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 20px;
        font-family: Arial, sans-serif;
    }
    .sidebar h2 {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .sidebar p {
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .sidebar .stButton>button {
        width: 100%;
        margin-bottom: 20px;
    }
    .sidebar .stSelectbox {
        margin-bottom: 30px;
    }
    .follow-us {
        margin-top: 0px;
    }
    .follow-us p {
        font-size: 14px;
        margin-bottom: 10px;
    }
    .sidebar-footer {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    .sidebar-footer a {
        opacity: 0.7;
        transition: opacity 0.3s;
    }
    .sidebar-footer a:hover {
        opacity: 1;
    }
    .copyright {
        color: #AAAAAA;
        font-size: 12px;
        text-align: center;
    }
    .copyright a {
        color: #AAAAAA;
        text-decoration: none;
    }
    .copyright a:hover {
        text-decoration: underline;
    }
    </style>
    """

    about_us_text = """
    <h2>About Us</h2>
    <p>NERO, founded in October 2010, is a research laboratory at UFV focused on robotics, encompassing control, automation, electronics, and computer science. It addresses challenges in robot navigation, cooperation, and human-robot interaction using AI and computer vision. NERO provides extensive documentation and maintains the AuRoRA Platform for project facilitation, promoting an inclusive, collaborative environment to advance robotics.</p>
    """

    social_links = """
    <div class="follow-us">
        <p>Follow us on social media for the latest updates and news:</p>
    </div>
    <div class="sidebar-footer">
        <a href="https://www.instagram.com/roboticaufv/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/128/3955/3955024.png" width="24" height="24" alt="Instagram">
        </a>
        <a href="https://www.facebook.com/roboticaUFV" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/128/1384/1384053.png" width="24" height="24" alt="Facebook">
        </a>
        <a href="https://www.linkedin.com/company/roboticaufv/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/128/3536/3536505.png" width="24" height="24" alt="LinkedIn">
        </a>
        <a href="https://www.youtube.com/c/roboticaufv" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/128/3670/3670147.png" width="24" height="24" alt="YouTube">
        </a>
    </div>
    """

    copyright_text = """
    <p class="copyright">
        <a href="https://nero-chatbot-add-paper-area.streamlit.app/" target="_blank" style="color: #AAAAAA; text-decoration: none;">
            © 2024 Núcleo de Especialização em Robótica - UFV
        </a>
    </p>
    """

    st.markdown(css, unsafe_allow_html=True)
    st.sidebar.markdown(about_us_text, unsafe_allow_html=True)
    st.sidebar.markdown(social_links, unsafe_allow_html=True)
    st.sidebar.markdown(copyright_text, unsafe_allow_html=True)


def main():
    set_page_config()
    initialize_session_state()

    st.title("🤖 AuRoRa Chat")
    st.caption("💬 Nero Virtual Assistant")

    with st.sidebar:
        st.image("assets/nero_banner.png")
        research_area = st.selectbox(
            "Filter by research area",
            (
                "All",
                "Aerial Robot Control",
                "Ground Robot Control",
                "Robot Formation",
                "Robot Control",
                "Formation Control",
                "Human-Robot Interaction",
                "Artificial Intelligence",
                "Robotics Competition",
                "Educational Robotics",
            ),
        )
        st.button("New Chat", on_click=new_chat, type="primary")
        set_sidebar_text()

    display_chat_history()

    if question := st.chat_input("Ask your question:"):
        with st.chat_message("user"):
            st.markdown(question)
        handle_user_input(question, research_area)


if __name__ == "__main__":
    main()
