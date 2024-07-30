import hmac
from datetime import datetime
from typing import Dict, List, Optional, Union

import streamlit as st

from libs.indexing_articles import embed_documents


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

    st.markdown("# 🖥️🔒 Administrator Area")
    st.markdown(
        "This area is exclusively for administrators. If you are not an administrator, "
        "please return to [AuRoRa Chat](https://nero-chatbot.streamlit.app/)."
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


def get_research_areas() -> List[str]:
    return [
        "Aerial Robot Control",
        "Ground Robot Control",
        "Robot Formation",
        "Robot Control",
        "Formation Control",
        "Human-Robot Interaction",
        "Artificial Intelligence",
        "Robotics Competition",
        "Educational Robotics",
    ]


def display_admin_page() -> None:
    """Displays the admin page if the password is correct."""
    if not check_password():
        st.stop()

    st.markdown("# 📝 Add new papers")

    article_doc = st.file_uploader("Select a paper (PDF or DOCX)", type=["pdf", "docx"])

    with st.form("pdf_form"):
        article_title = st.text_input("Paper title", placeholder="Title of the paper")
        research_area = st.selectbox(
            "Research area",
            options=get_research_areas(),
            index=None,
            placeholder="Area of research",
        )
        current_year = datetime.now().year
        publication_year = st.selectbox(
            "Year of publication",
            options=range(current_year, 1999, -1),
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


def validate_form_data(
    article_doc: bytes,
    article_title: str,
    first_author: str,
    research_area: str,
    publication_year: int,
    gdrive_url: str,
) -> str:
    """
    Validates the form data.

    Returns:
        str: An error message if validation fails, None otherwise.
    """
    if not article_doc:
        return "Please upload a file."
    if not all(
        [article_title, first_author, research_area, publication_year, gdrive_url]
    ):
        return "Please fill in all fields and upload a file."
    return None


def create_article_metadata(
    article_title: str,
    first_author: str,
    research_area: str,
    publication_year: int,
    gdrive_url: str,
) -> Dict[str, Union[str, int]]:
    """
    Creates a metadata dictionary for the article.

    Returns:
        Dict[str, Union[str, int]]: The article metadata.
    """
    return {
        "article_title": article_title,
        "first_author": first_author,
        "research_area": research_area,
        "publication_year": publication_year,
        "source": gdrive_url,
    }


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
        research_area (str): The area/topic of the article.
        publication_year (int): The year the article was published.
        gdrive_url (str): The Google Drive URL for the article.
    """
    error_message = validate_form_data(
        article_doc,
        article_title,
        first_author,
        research_area,
        publication_year,
        gdrive_url,
    )
    if error_message:
        st.error(error_message)
        return

    article_metadata = create_article_metadata(
        article_title, first_author, research_area, publication_year, gdrive_url
    )

    try:
        with st.spinner("Processing the paper... This may take a moment."):
            success_embeddings = embed_documents(article_doc, article_metadata)
        if success_embeddings:
            st.success("Paper added successfully!")
        else:
            st.warning("Error inserting the paper. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


set_page_config()
display_admin_page()
