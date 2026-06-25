from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class LLM_state(TypedDict):
    query: str
    response: str


graph = StateGraph(LLM_state)


def llm_call(state: LLM_state) -> LLM_state:
    query = state["query"]
    response = llm_model.invoke(query)
    state["response"] = response.content
    return state


graph.add_node("llm_call", llm_call)

graph.add_edge(START, "llm_call")
graph.add_edge("llm_call", END)

workflow = graph.compile()

final_state = workflow.invoke({"query": "what is the capital of Andhra Pradesh"})

print(final_state["response"])
