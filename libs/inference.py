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
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

preamble = """

## Task & Context
You are AuRoRa, the Virtual Assistant of Nero (Núcleo de Especialização em Robótica) at the Universidade Federal de Viçosa. Your primary role is to answer questions related to Nero's research areas, which include:

- Aerial Robot Control
- Ground Robot Control
- Robot Formation
- Robot Control
- Formation Control
- Human-Robot Interaction
- Artificial Intelligence
- Robotics Competition
- Educational Robotics

Use the provided documents, which consist of Nero's articles, to assist in answering these questions.

## Tone
Maintain a polite and respectful demeanor. Use academic language appropriate for a university setting

## Guidelines
1. Answer questions exclusively related to Nero's research areas
2. Base your responses on the retrieved documents (Nero's articles)
3. If the question is unrelated to Nero's research or if the information is not in the provided articles, state that you don't know the answer
4. Keep responses concise, using at most three sentences
5. Do not answer questions outside the scope of Nero's research areas
6. If the retrieved documents are not relevant to the question, inform the user that you don't have the necessary information
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
            "title": f'[{doc.metadata.get("first_author")} ({int(doc.metadata.get("publication_year"))}). {doc.metadata.get("article_title")}]({doc.metadata.get("source")})',
            "snippet": doc.page_content,
        }
        for doc in documents
    ]


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    chat_history = []

    for message in messages[:-1]:
        if message["role"] == "user":
            chat_history.append({"role": "USER", "message": message["content"]})
        elif message["role"] == "assistant":
            chat_history.append({"role": "CHATBOT", "message": message["content"]})

    print("CHAT HISTORY: ", chat_history)
    return chat_history if chat_history else None


def chat_answer(
    question: str, streamlit_chat_history: List
) -> Generator[str, None, None]:

    augmented_queries = cohere_client.chat(
        message=question,
        model="command-r-plus",
        temperature=0.2,
        search_queries_only=True,
    )

    if augmented_queries.search_queries:
        documents = format_documents(augmented_queries.search_queries[0].text)
    else:
        documents = None

    citations = []
    for event in cohere_client.chat_stream(
        model="command-r-plus",
        message=question,
        preamble=preamble,
        chat_history=format_chat_history(streamlit_chat_history),
        documents=documents,
        temperature=0.2,
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
