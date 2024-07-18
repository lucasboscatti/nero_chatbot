from typing import Dict, Generator, List

import cohere
import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore

COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

cohere_client = cohere.Client(api_key=COHERE_API_KEY)
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
retriever = vectorstore.as_retriever()

preamble = """

## Task & Context
You are AuRoRa, the Virtual Assistant of Nero (Núcleo de Especialização em Robótica) of the Universidade Federal de Vicosa. Your role is to answer the questions related to the
research areas of the Nero University.

Use the retrieved documents to answer the questions.
The documents are Nero's articles. You must not answer questions that are not related to the area of research of the Nero.

If you don't know the answer or the articles are not related, just say you don't know.
Use at most three sentences and keep the response concise.

## Style Guide
You must be polite, respectful and use academic language.
"""


def rerank_documents(question: str, documents):
    docs = [doc.page_content for doc in documents]
    rerank = cohere_client.rerank(
        model="rerank-english-v3.0", query=question, documents=docs, top_n=3
    )
    reranked_documents = []
    for result in rerank.results:
        if result.relevance_score >= 0.5:
            reranked_documents.append(documents[result.index])

    return reranked_documents


def format_documents(question: str) -> List[Dict[str, str]]:
    documents = retriever.get_relevant_documents(question)
    documents = rerank_documents(question, documents)
    return [
        {
            "title": f'[{doc.metadata.get("article_title")}, {doc.metadata.get("article_year")}]({doc.metadata.get("source")})',
            "snippet": doc.page_content,
        }
        for doc in documents
    ]


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    chat_history = []
    for message in messages:
        if message["role"] == "user":
            chat_history.append({"role": "USER", "content": message["content"]})
        elif message["role"] == "assistant":
            chat_history.append({"role": "ASSISTANT", "content": message["content"]})

    return chat_history


def chat_answer(
    question: str, streamlit_chat_history: List
) -> Generator[str, None, None]:
    citations = []
    for event in cohere_client.chat(
        model="command-r-plus",
        stream=True,
        message=question,
        preamble=preamble,
        chat_history=format_chat_history(streamlit_chat_history),
        documents=format_documents(question),
    ):
        if event.event_type == "text-generation":
            yield event.text
        elif event.event_type == "citation-generation":
            citations.extend(event.citations[0].document_ids)
        elif event.event_type == "stream-end":
            process_citations(event, citations)


def process_citations(event, citations):
    unique_citations = set(citations)
    sources = [
        doc["title"]
        for citation in unique_citations
        for doc in event.response.documents
        if doc["id"] == citation
    ]
    unique_sources = list(set(sources))
    save_sources(unique_sources)


def save_sources(sources: List[str]):
    with open("sources.txt", "w") as f:
        f.write("\n".join(sources))
