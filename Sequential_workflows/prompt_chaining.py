from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template1 = ChatPromptTemplate(
    [("human", "give me an outline to write an essay on the topic: {topic}")]
)

prompt_template2 = ChatPromptTemplate(
    [
        (
            "human",
            "write a short essay on the given topic according to the given outline.\ntopic: {topic}\noutline:{outline}",
        )
    ]
)


class essay_state(TypedDict):
    topic: str
    outline: str
    essay: str


graph = StateGraph(essay_state)

chain1 = prompt_template1 | llm_model
chain2 = prompt_template2 | llm_model


def get_outline(state: essay_state) -> essay_state:
    topic = state["topic"]
    response = chain1.invoke({"topic": topic})
    state["outline"] = response.content
    return state


def get_essay(state: essay_state) -> essay_state:
    topic = state["topic"]
    outline = state["outline"]
    response = chain2.invoke({"topic": topic, "outline": outline})
    state["essay"] = response.content
    return state


graph.add_node("get_outline", get_outline)
graph.add_node("get_essay", get_essay)

graph.add_edge(START, "get_outline")
graph.add_edge("get_outline", "get_essay")
graph.add_edge("get_essay", END)

workflow = graph.compile()

final_state = workflow.invoke({"topic": "Pawan Kalyan"})

print(final_state)
