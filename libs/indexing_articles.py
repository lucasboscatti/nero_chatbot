import os
import tempfile
from typing import Dict, List, Tuple

import nest_asyncio
import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from llama_parse import LlamaParse

nest_asyncio.apply()


os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]


PINECONE_INDEX = st.secrets["PINECONE_INDEX"]
LLAMA_CLOUD_API_KEY = st.secrets["LLAMA_CLOUD_API_KEY"]


def add_metadata(documents: List[Document], metadata: Dict) -> List[Document]:
    """
    Add metadata to a list of documents.

    Args:
        documents (List[Document]): List of documents to which metadata is added.
        metadata (Dict): Metadata to be added to the documents.

    Returns:
        List[Document]: Updated list of documents with added metadata.
    """
    return [Document(page_content=doc.text, metadata=metadata) for doc in documents]


def create_documents(file, metadata: Dict) -> List[Document]:
    """
    Load documents, add metadata, and split into chunks.

    Args:
        file (st.UploadedFile): Uploaded file object.
        metadata (Dict): Metadata to be added to the documents.

    Returns:
        List[Document]: List of document chunks.
    """
    documents = load_document(file)
    documents_with_metadata = add_metadata(documents, metadata)
    return get_chunks(documents_with_metadata)


def load_document(file) -> List[Document]:
    """
    Load documents using LlamaParse.

    Args:
        file (st.UploadedFile): Uploaded file object.

    Returns:
        List[Document]: List of loaded documents.
    """
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="text",
        num_workers=4,
        verbose=True,
    )

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{file.name.split('.')[-1]}"
    ) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        documents = parser.load_data(tmp_file_path)
    finally:
        os.remove(tmp_file_path)

    return documents


def get_chunks(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks.

    Args:
        documents (List[Document]): List of documents to be split.

    Returns:
        List[Document]: List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_and_save_index(
    documents: List[Document], research_area: str
) -> Tuple[str, CohereEmbeddings]:
    """
    Insert documents into Pinecone index.

    Args:
        documents (List[Document]): List of documents to create index from.

    Returns:
        None
    """
    embedding_model = CohereEmbeddings(model="embed-english-v3.0")
    PineconeVectorStore.from_documents(
        documents,
        index_name=PINECONE_INDEX,
        embedding=embedding_model,
        namespace=research_area,
    )
    return


def embedding_documents(file, metadata: Dict) -> bool:
    """
    Embed documents into the vector store.

    Args:
        file (st.UploadedFile): Uploaded file object.
        metadata (Dict): Metadata of the document.

    Returns:
        bool: True if embedding is successful, False otherwise.
    """
    try:
        documents = create_documents(file, metadata)
        create_and_save_index(documents, metadata["research_area"])
        return True
    except Exception as e:
        st.error(f"Error during document embedding: {str(e)}")
        return False
