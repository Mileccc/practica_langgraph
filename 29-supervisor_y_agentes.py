import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated, List, Tuple, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
import torch
from langchain_experimental.tools import PythonREPLTool

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
os.environ['TAVILY_API_KEY'] = os.environ.get("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


tavily_tool = TavilySearchResults(max_results=5)

python_repl_tool = PythonREPLTool()

loader = DirectoryLoader('./source', glob="./*.txt", loader_cls=TextLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len
)
new_docs = text_splitter.split_documents(documents=docs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': device}
# set True to compute cosine similarity
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

db = Chroma.from_documents(new_docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 4})


@tool
def RAG(state):
    """Utilícelo para ejecutar el RAG. Si la pregunta está relacionada con Japón o Deportes, utilice esta herramienta para obtener los resultados."""

    print('-> Calling RAG ->')
    question = state
    print('Question:', question)

    template = """Responda a la pregunta basándose únicamente en el siguiente contexto:
    {context}

    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = retrieval_chain.invoke(question)
    return result


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


members = ["RAG", "Researcher", "Coder"]
options = ["FINISH"] + members


system_prompt = (
    """Eres un supervisor encargado de gestionar una conversación entre los siguientes trabajadores:  {members}. Dada la siguiente petición del usuario, responda con el trabajador que debe actuar a continuación. Utilice la herramienta RAG cuando las preguntas estén relacionadas con Japón o de categoría Deportes. Cada trabajador realizará una tarea y responderá con sus resultados y estado. Cuando termine, responde con FINISH."""
)

# El uso de llamadas a funciones openai puede facilitarnos el análisis sintáctico de la salida
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Dada la conversación anterior, ¿quién debe actuar a continuación? ¿O debemos TERMINAR? Seleccione una de: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)


class AgentState(TypedDict):
    # La anotación indica al gráfico que siempre se añadirán nuevos mensajes a los estados actuales
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # El campo «next» indica el siguiente destino
    next: str


research_agent = create_agent(
    llm,
    [tavily_tool],
    "Usted es investigador web."
)
research_node = functools.partial(
    agent_node, agent=research_agent, name="Researcher")

# NOTE: ESTO REALIZA LA EJECUCIÓN DE CÓDIGO ARBITRARIO. PROCEDA CON PRECAUCIÓN
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "Puedes generar código python seguro para analizar datos y generar gráficos utilizando matplotlib.",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

RAG_agent = create_agent(
    llm,
    [RAG],
    "Utilice estas herramientas cuando las preguntas estén relacionadas con Japón o de la categoría Deportes..",
)
rag_node = functools.partial(agent_node, agent=RAG_agent, name="RAG")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("RAG", rag_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # Queremos que nuestros trabajadores SIEMPRE «informen» al supervisor cuando hayan terminado.
    workflow.add_edge(member, "supervisor")
# El supervisor rellena el campo «siguiente» en el estado del grafo que dirige a un nodo o termina
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges(
    "supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")

graph = workflow.compile()


for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="""Obtén el PIB de Japón en los últimos 4 años de RAG,»
                « y luego dibuja un gráfico lineal del mismo.»
                « Una vez que lo codifiques, termina.""")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")
