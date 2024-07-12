import asyncio

import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="🤖 Converse com a AuRoRa", layout="wide")

from add_articles import display_admin_page
from chatbot import chatbot_page

# Add navigation to switch between pages
page_names_to_funcs = {
    "AuRoRa Chat": chatbot_page,
    "Cadastrar artigos": display_admin_page,
}

pages = st.sidebar.selectbox("Escolha uma página", page_names_to_funcs.keys())

# Run the selected page
page_names_to_funcs[pages]()
