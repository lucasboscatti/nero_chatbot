import asyncio
import logging
import os
from typing import Annotated, AsyncGenerator, Literal, TypedDict

import streamlit as st
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_cohere import CohereEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

MAX_RETRIES = 1
VERBOSE = True

tavily_search_tool = TavilySearchResults(max_results=3)
embeddings = CohereEmbeddings(
    cohere_api_key=st.secrets["COHERE_API_KEY"], model="embed-multilingual-v3.0"
)
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_nero_articles",
    "Busca e retorne informações dos artigos do Nero sobre robótica, inteligência artificial, interação humano-robô, dronos, reinforcement learning e outras áreas correlatas",
)

tools = [retriever_tool]

llm = ChatGroq(model="llama3-70b-8192", temperature=0)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """
    logger.info("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    llm_with_tool = llm.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""Você é um avaliador analisando a relevância de um documento recuperado em relação a uma pergunta do usuário. \n 
        Aqui está o documento recuperado: \n\n {context} \n\n
        Aqui está a pergunta do usuário: {question} \n
        Se o documento contiver palavra(s)-chave ou significado semântico relacionado à pergunta do usuário, avalie-o como relevante. \n
        Dê uma pontuação binária 'yes' ou 'no' para indicar se o documento é relevante para a pergunta.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        logger.info("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        logger.info("---DECISION: DOCS NOT RELEVANT---")
        logger.info(score)
        return "rewrite"


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    logger.info("---CALL AGENT---")
    messages = state["messages"]
    model = llm.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    logger.info("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Olhe para a entrada e tente raciocinar sobre a intenção/ significado semântico subjacente. \n
    Aqui está a pergunta inicial:
    \n ------- \n
    {question} 
    \n ------- \n
    Formule uma pergunta aprimorada: """,
        )
    ]

    # Grader
    response = llm.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    logger.info("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    # Prompt
    prompt = PromptTemplate(
        template="""Você é um assistente para tarefas de perguntas e respostas. Use os seguintes trechos de contexto recuperados para responder à pergunta. Se não souber a resposta, diga apenas que não sabe. Use no máximo três frases e mantenha a resposta concisa.
        Pergunta: {question}
        Contexto: {context}
        Resposta:""",
        input_variables=["question", "context"],
    )

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()


async def process_stream(message):
    inputs = {"messages": [("human", message)]}
    async for event in graph.astream_events(inputs, version="v2"):
        logger.info(event)
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"]
            yield content.content


def to_sync_generator(async_gen: AsyncGenerator):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                yield loop.run_until_complete(anext(async_gen))
            except StopAsyncIteration:
                break
    finally:
        loop.close()
