import os
import tempfile
from typing import Any, Dict, List

import nest_asyncio
import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from llama_parse import LlamaParse
from streamlit.runtime.uploaded_file_manager import UploadedFile

from libs.config import Config

nest_asyncio.apply()

config = Config()


def add_metadata(documents: List[Document], metadata: Dict[str, Any]) -> List[Document]:
    """
    Add metadata to a list of documents.

    Args:
        documents (List[Document]): List of documents to which metadata is added.
        metadata (Dict[str, Any]): Metadata to be added to the documents.

    Returns:
        List[Document]: Updated list of documents with added metadata.
    """
    return [Document(page_content=doc.text, metadata=metadata) for doc in documents]


def create_documents(file: UploadedFile, metadata: Dict[str, Any]) -> List[Document]:
    """
    Load documents, add metadata, and split into chunks.

    Args:
        file (UploadedFile): Uploaded file object.
        metadata (Dict[str, Any]): Metadata to be added to the documents.

    Returns:
        List[Document]: List of document chunks.
    """
    documents = load_document(file)
    documents_with_metadata = add_metadata(documents, metadata)
    return split_documents(documents_with_metadata)


def load_document(file: UploadedFile) -> List[Document]:
    """
    Load documents using LlamaParse.

    Args:
        file (UploadedFile): Uploaded file object.

    Returns:
        List[Document]: List of loaded documents.

    Raises:
        ValueError: If the file cannot be processed.
    """
    parser = LlamaParse(
        api_key=config.LLAMA_CLOUD_API_KEY,
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
        if not documents:
            raise ValueError("No documents were extracted from the file.")
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}") from e
    finally:
        os.remove(tmp_file_path)
        return documents


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks.

    Args:
        documents (List[Document]): List of documents to be split.
        chunk_size (int, optional): The size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 200.

    Returns:
        List[Document]: List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_and_save_index(documents: List[Document]) -> None:
    """
    Insert documents into Pinecone index.

    Args:
        documents (List[Document]): List of documents to create index from.

    Raises:
        ValueError: If there's an error creating or saving the index.
    """
    try:
        embedding_model = CohereEmbeddings(model="embed-english-v3.0")
        PineconeVectorStore.from_documents(
            documents, index_name=config.PINECONE_INDEX, embedding=embedding_model
        )
    except Exception as e:
        raise ValueError(f"Error creating or saving index: {str(e)}") from e


def embed_documents(file: UploadedFile, metadata: Dict[str, Any]) -> None:
    """
    Embed documents into the vector store.

    Args:
        file (UploadedFile): Uploaded file object.
        metadata (Dict[str, Any]): Metadata of the document.

    Returns:
        bool: True if embedding is successful, False otherwise.
    """
    try:
        documents = create_documents(file, metadata)
        create_and_save_index(documents)
        return True
    except Exception as e:
        st.error(f"Error during document embedding: {str(e)}")
        return False
