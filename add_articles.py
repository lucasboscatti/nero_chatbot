import hmac
import os
from typing import Any

import streamlit as st
from database.database import (
    check_if_id_exists,
    connect_to_database,
    create_database,
    insert_article,
)
from libs.indexing_articles import embedding_documents


def check_password() -> bool:
    """Returns `True` if the user had the correct password."""

    def password_entered() -> None:
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True
    st.markdown("# Área do Administrador")
    st.write(
        "Esta área destina-se exclusivamente a administradores. Se você não é um administrador, por favor, retorne ao AuRoRa Chat."
    )

    # Show input for password.
    st.text_input("Senha", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Senha incorreta!")
    return False


def display_admin_page() -> None:
    """Display the admin page."""
    if not check_password():
        st.stop()
    st.markdown("# Adicione Novos Trabalhos")
    st.write("Para encontrar o ID do artigo no Google Drive, siga os passos abaixo:")
    st.markdown(
        """1. Procure pelo artigo no Google Drive.\n
2. Nos 3 pontos de menu, clique em `Compartilhar` e depois em `Copiar link`.\n
3. Após copiar o link, extraia apenas o ID do link e cole-o no campo abaixo. O link deve seguir o seguinte formato: `https://drive.google.com/file/d/<<ID>>/view?usp=sharing`

    > Por exemplo, para o link `https://drive.google.com/file/d/1234567abcdefg/view?usp=sharing`, o ID é `1234567abcdefg`."""
    )
    st.write(
        "**IMPORTANTE**: Verifique se o artigo está compartilhado com `Qualquer pessoa com o link`."
    )

    conn = connect_to_articles_db()

    with st.form("pdf_form"):
        gdrive_ID = st.text_input("ID do artigo no Google Drive")
        article_title = st.text_input("Nome do artigo")
        article_area = st.selectbox(
            "Selecionar a área do artigo",
            (
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

        submitted = st.form_submit_button("Enviar")
        if submitted:
            handle_form_submission(gdrive_ID, article_title, article_area, conn)


def handle_form_submission(
    gdrive_ID: str, article_title: str, article_area: str, conn: Any = None
) -> None:
    """Handle form submission."""
    if not gdrive_ID:
        st.warning("Por favor, insira um link de arquivo válido.")
        st.stop()

    if not article_title:
        st.warning("Por favor, insira o nome do artigo.")
        st.stop()

    if check_if_id_exists(conn, gdrive_ID):
        st.warning("Este artigo ja foi adicionado. Tente novamente.")
        st.stop()

    article_metadata = {
        "article_title": article_title,
        "article_area": article_area,
        "sources": f"https://drive.google.com/file/d/{gdrive_ID}",
    }

    success_embeddings = embedding_documents(article_metadata)

    if success_embeddings:
        success_insert = insert_article(
            conn,
            art_id=gdrive_ID,
            title=article_title,
            area=article_area,
            source=f"https://drive.google.com/file/d/{gdrive_ID}",
        )
        if success_insert:
            st.success("Artigo adicionado com sucesso!")

        else:
            st.warning("Erro ao inserir o artigo. Tente novamente.")


@st.cache_resource
def connect_to_articles_db() -> Any:
    """Connect to the articles database."""
    if not os.path.exists("database/articles.db"):
        create_database()
    return connect_to_database()
