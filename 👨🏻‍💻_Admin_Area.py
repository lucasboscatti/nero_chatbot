import hmac
from datetime import datetime
from typing import Optional

import streamlit as st

from libs.indexing_articles import embedding_documents


def set_page_config():
    st.set_page_config(
        page_title="AuRoRa Chat Admin Area", layout="wide", page_icon="📝"
    )


def check_password() -> bool:
    """
    Checks if the user entered the correct password.

    Returns:
        bool: True if the password is correct, False otherwise.
    """

    def password_entered() -> None:
        """Validates the entered password and updates the session state."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True
    st.markdown("# Administrator Area")
    st.write(
        "This area is exclusively for administrators. If you are not an administrator, please return to AuRoRa Chat."
    )

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )

    if "password_correct" not in st.session_state:
        return False

    if not st.session_state["password_correct"]:
        st.error("😕 Incorrect password!")
        return False

    return True


def display_admin_page() -> None:
    """Displays the admin page if the password is correct."""
    if not check_password():
        st.stop()

    st.markdown("# 📝 Add new papers")

    article_doc = st.file_uploader(
        "Select an paper (pdf or docx)", type=["pdf", "docx"]
    )

    with st.form("pdf_form"):
        article_title = st.text_input("Paper title", placeholder="Title of the paper")
        research_area = st.selectbox(
            "Research area",
            (
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
            placeholder="Area of research",
        )
        publication_year = st.selectbox(
            "Year of publication",
            range(2000, datetime.now().year + 1),
            index=None,
            placeholder="Select the year of publication",
        )
        first_author = st.text_input(
            "First author", placeholder="Name of the first author"
        )
        gdrive_url = st.text_input(
            "Publication link on Google Drive",
            placeholder="URL of the publication on Google Drive",
        )

        submitted = st.form_submit_button("Submit")
        if submitted:
            handle_form_submission(
                article_doc,
                article_title,
                first_author,
                research_area,
                publication_year,
                gdrive_url,
            )


def handle_form_submission(
    article_doc: Optional[bytes],
    article_title: str,
    first_author: str,
    research_area: str,
    publication_year: int,
    gdrive_url: str,
) -> None:
    """
    Handles the form submission for adding a new article.

    Args:
        article_doc (Optional[bytes]): The uploaded article document.
        article_title (str): The title of the article.
        first_author (str): The name of the first author.
        article_area (str): The area/topic of the article.
        article_year (int): The year the article was published.
        gdrive_url (str): The Google Drive URL for the article.
    """
    if not all(
        [
            article_doc,
            article_title,
            first_author,
            research_area,
            publication_year,
            gdrive_url,
        ]
    ):
        st.error("Please fill in all fields and upload a file.")
        return

    article_metadata = {
        "article_title": article_title,
        "first_author": first_author,
        "research_area": research_area,
        "publication_year": publication_year,
        "source": gdrive_url,
    }

    try:
        success_embeddings = embedding_documents(article_doc, article_metadata)
        if success_embeddings:
            st.success("Article added successfully!")
        else:
            st.warning("Error inserting the article. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


set_page_config()
display_admin_page()
