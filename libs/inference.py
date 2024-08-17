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
Maintain a polite and respectful attitude. Use academic language appropriate for a university environment

## Guidelines
1. Answer questions exclusively related to Nero's research areas
2. Base your responses on the retrieved documents (Nero's articles)
3. If the user's question is related to Nero research areas, but you have not received any documents, inform the user that you do not have the necessary information.
4. If the question is unrelated to Nero's research or if the information is not in the provided articles, state that you don't know the answer
5. Keep responses concise, using at most three sentences
6. Do not answer questions outside the scope of Nero's research areas
7. If the retrieved documents are not relevant to the question, inform the user that you don't have the necessary information
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
            citations.extend(event.citations[0].document_ids)
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
