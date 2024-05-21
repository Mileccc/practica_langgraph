from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated, List, Union
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
import operator
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
import time
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def nodo_tools_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    # current_state = event.get("dialog_state")
    # if current_state:
    #     print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + \
                    [("user", "Desarrolla una respuesta acorde a la pregunta.")]
                state = {**state, "messages": messages}
            else:
                print(
                    f"***************************{result}************************************")
                break
        return {"messages": result}


tools = [TavilySearchResults(max_results=1)]

llm = ChatGroq(temperature=0, model="llama3-70b-8192", streaming=False)
llm_herramienta = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            # Rol
            Eres un asistente amable y flexible experto en localizar informacion en la web.\n
            # Tarea
            Cuando se te requiera información usarás la herramienta llamada TavilySearchResults para consultar su información y tener mejor conocimiento antes de responderle al usuario. Primero tomaras 5 palabras clave de la pregunta y enviaras esa lista con las palabras claves a la herramienta .
            # Idioma de respuesta: Español
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)

assistant_runnable = prompt | llm_herramienta

builder = StateGraph(State)
builder.add_node("asistante", Assistant(assistant_runnable))
builder.add_node("tools", nodo_tools_fallback(tools))
builder.set_entry_point("asistante")
builder.add_conditional_edges("asistante", tools_condition, {
                              "tools": "tools", END: END})
builder.add_edge("tools", "asistante")
ejecutor = builder.compile()

preguntas = [
    "Explicame como funciona LangGraph.",
    "¿que son los agentes en ese contexto?",
    "Dame un ejemplo de uso.",
]

_printed = set()

for pregunta in preguntas:
    events = ejecutor.stream(
        {"messages": ("user", pregunta)}, stream_mode="values")
    for event in events:
        _print_event(event, _printed)
    time.sleep(2)
