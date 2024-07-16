import os

import cohere
import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore

os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

co = cohere.Client(api_key=st.secrets["COHERE_API_KEY"])

embeddings = CohereEmbeddings(
    cohere_api_key=st.secrets["COHERE_API_KEY"], model="embed-multilingual-v3.0"
)
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
retriever = vectorstore.as_retriever()


def format_documents(question):
    formatted_documents = []
    documents = retriever.get_relevant_documents(question)
    for document in documents:
        formatted_documents.append(
            {
                "title": f'[{document.metadata.get("article_title")}]({document.metadata.get("sources")})',
                "snippet": document.page_content,
            }
        )
    return formatted_documents


def chat_answer(question, preamble):
    citations = []
    for event in co.chat_stream(
        model="command-r-plus",
        message=question,
        preamble=preamble,
        documents=format_documents(question),
    ):
        if event.event_type == "text-generation":
            yield event.text

        if event.event_type == "citation-generation":
            citations.append(event.citations[0].document_ids)

        if event.event_type == "stream-end":
            citations = set(sum(citations, []))
            sources = []
            documents = event.response.documents
            for citation in citations:
                for document in documents:
                    if document["id"] == citation:
                        sources.append(document["title"])

            sources = list(set(sources))

            with open("sources.txt", "w") as f:
                f.write("\n".join(sources))
