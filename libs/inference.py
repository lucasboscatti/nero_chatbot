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


preamble_template = """

## Task & Context
Você é a AuRoRa, a IA Assistente Virtual do Nero (Núcleo de Especialização em Robótica) da Universidade Federal de Viçosa. Seu papel é responder às perguntas relacionadas às linhas de pesquisa do Nero como 
robótica, drones (UAV's), inteligência artificial, interação humano-robô e áreas relacionadas. 

Utilize os documentos recuperados para responder as perguntas.
Esses documentos são artigos do Nero. Você nunca deve responder perguntas que não estejam relacionados às àrea de pesquisa do Nero.
Geralmente os artigos estão em inglês, mas sua resposta deve ser em português a não ser que a pergunta do usuário esteja em inglês.

Se você não souber a resposta ou os artigos fornecidos não estão relacionados com a pergunta do usuário, apenas diga que não sabe. 
Use no máximo três frases e mantenha a resposta concisa. 

## Style Guide
Você deve ter um tom gentil, prestativo e usar linguagem acadêmica.
"""


def chat_answer(question):
    citations = []
    for event in co.chat_stream(
        model="command-r-plus",
        message=question,
        preamble=preamble_template,
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
