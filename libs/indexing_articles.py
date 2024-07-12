import os
from typing import Dict, List, Tuple

import gdown
import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from llama_index.readers.smart_pdf_loader import SmartPDFLoader

os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

PINECONE_INDEX = st.secrets["PINECONE_INDEX"]


def add_metadata(documents: List, metadata: Dict) -> List:
    """
    Add metadata to a list of documents.

    Parameters:
        documents (List): List of documents to which metadata is added.
        metadata (Dict): Metadata to be added to the documents.

    Returns:
        List: Updated list of documents with added metadata.
    """
    new_documents = []
    for document in documents:
        new_document = Document(
            page_content=document.text,
            metadata=metadata,
        )
        new_documents.append(new_document)
    return new_documents


def create_pdf_documents(file_path: str, metadata: Dict) -> List:
    """
    Load PDF documents, add metadata, and split into chunks.

    Parameters:
        file_path (str): Path to the PDF file.
        metadata (Dict): Metadata to be added to the documents.

    Returns:
        List: List of document chunks.
    """
    documents = load_pdf(file_path)
    documents = add_metadata(documents, metadata)
    chunks = get_chunks(documents)
    return chunks


def load_pdf(path: str) -> List:
    """
    Load PDF files.

    Parameters:
        path (str): Path to the PDF file.

    Returns:
        List: List of loaded documents.
    """
    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
    documents = pdf_loader.load_data(path)
    # os.remove(path)
    return documents


def get_chunks(documents: List) -> List:
    """
    Split documents into chunks.

    Parameters:
        documents (List): List of documents to be split.

    Returns:
        List: List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_and_save_index(documents: List) -> Tuple[str, CohereEmbeddings]:
    """
    Create and save FAISS index from documents.

    Parameters:
        documents (List): List of documents to create index from.
        index_name (str): Name of the index.

    Returns:
        Tuple: A tuple containing index name and embedding model.
    """
    embedding_model = CohereEmbeddings(model="embed-multilingual-v3.0")
    PineconeVectorStore.from_documents(
        documents, index_name=PINECONE_INDEX, embedding=embedding_model
    )


def download_from_gdrive(file_ID: str) -> str:
    """
    Download file from Google Drive.

    Parameters:
        file_ID (str): ID of the file on Google Drive.

    Returns:
        str: Path to the downloaded file.
    """
    if not os.path.exists("./pdfs"):
        os.makedirs("./pdfs")

    file_path = f"./pdfs/{file_ID}.pdf"
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_ID}", file_path)
        return file_path
    except gdown.exceptions.FileURLRetrievalError:
        st.error(
            "Não foi possível recuperar o link público do arquivo. Talvez seja necessário alterar a permissão para 'Qualquer pessoa com o link' ou o ID esteja errado."
        )
        st.stop()


def embedding_documents(metadata: Dict) -> bool:
    """
    Embed documents.

    Parameters:
        metadata (Dict): Metadata of the document.

    Returns:
        bool: True if embedding is successful, False otherwise.
    """
    file_ID = metadata["sources"].split("/")[-1]
    file_path = download_from_gdrive(file_ID)
    if file_path:
        documents = create_pdf_documents(file_path, metadata)
        create_and_save_index(documents)
        return True
    return False
