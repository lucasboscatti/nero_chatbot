from typing import Dict, Generator, List

import cohere
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from libs.config import Config

config = Config()

cohere_client = cohere.Client(api_key=config.COHERE_API_KEY)
embeddings = CohereEmbedding(
    model="embed-english-v3.0", input_type="search_query", api_key=config.COHERE_API_KEY
)

Settings.embed_model = embeddings

pc = Pinecone(
    api_key=config.PINECONE_API_KEY,
)

pinecone_index = pc.Index(config.PINECONE_INDEX)

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    embedding=embeddings,
    api_key=config.PINECONE_API_KEY,
)


preamble = """

## Task & Context
You are AuRoRA, the Virtual Assistant of NERo (Núcleo de Especialização em Robótica) at the Universidade Federal de Viçosa. As an expert assistant, you help users answer questions about NERo's academic production. Use the provided documents to address inquiries regarding NERo's academic research. If a question is unrelated to NERo's research areas or if the information is not available in the provided articles, clearly state that you do not know the answer.

## Style Guide
Unless the user requests a different style of answer, respond in complete sentences using proper grammar and spelling.
"""


def rerank_documents(question: str, documents):
    docs = [doc.text for doc in documents]
    rerank = cohere_client.rerank(
        model="rerank-english-v3.0", query=question, documents=docs, top_n=3
    )

    reranked_documents = [documents[result.index] for result in rerank.results]

    return [
        {
            "title": f'[{doc.metadata.get("first_author")} ({int(doc.metadata.get("publication_year"))}). {doc.metadata.get("article_title")}]({doc.metadata.get("source")})',
            "snippet": doc.get_content(),
        }
        for doc in reranked_documents
    ]


def format_documents(query: str, research_area: str) -> List[Dict[str, str]]:
    filter = (
        MetadataFilters(
            filters=[
                MetadataFilter(
                    key="research_area", operator=FilterOperator.EQ, value=research_area
                ),
            ]
        )
        if research_area != "All"
        else None
    )

    retriever = VectorStoreIndex.from_vector_store(vector_store).as_retriever(
        similarity_top_k=10,
        filters=filter,
    )

    documents = retriever.retrieve(query)

    return documents


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    chat_history = []

    for message in messages[:-1]:
        if message["role"] == "user":
            chat_history.append({"role": "USER", "message": message["content"]})
        elif message["role"] == "assistant":
            chat_history.append({"role": "CHATBOT", "message": message["content"]})

    return chat_history if chat_history else None


def chat_answer(
    question: str, streamlit_chat_history: List, research_area: str
) -> Generator[str, None, None]:

    augmented_queries = cohere_client.chat(
        message=question,
        model="command-r-plus",
        temperature=0.3,
        chat_history=format_chat_history(streamlit_chat_history),
        search_queries_only=True,
    )

    related_documents = []
    if augmented_queries.search_queries:
        for augmented_query in augmented_queries.search_queries:
            documents = format_documents(augmented_query.text, research_area)
            related_documents.extend(documents)
        documents = rerank_documents(question, related_documents)
    else:
        documents = None

    citations = []
    for event in cohere_client.chat_stream(
        model="command-r-plus",
        message=question,
        preamble=preamble,
        chat_history=format_chat_history(streamlit_chat_history),
        documents=documents,
        temperature=0.0,
    ):
        if event.event_type == "text-generation":
            yield event.text
        elif event.event_type == "citation-generation":
            for cit in event.citations:
                citations.extend(cit.document_ids)
        elif event.event_type == "stream-end":
            sources = process_citations(event, citations)
            yield sources


def process_citations(event, citations):
    unique_citations = set(citations)
    sources = [
        doc["title"]
        for citation in unique_citations
        for doc in event.response.documents
        if doc["id"] == citation
    ]
    return list(set(sources))
