from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()


class eval_model(BaseModel):
    feedback: str
    score: int


class essay_state(TypedDict):
    topic: str
    language_score: int
    language_feedback: str
    clarity_score: int
    clarity_feedback: str
    essay: str


llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

structured_llm = llm_model.with_structured_output(eval_model)

graph = StateGraph(essay_state)


def get_essay(state: essay_state):
    topic = state["topic"]
    response = llm_model.invoke(
        f"generate a short essay on the given topic.\ntopic:{topic}"
    )
    return {"essay": response.content}


def get_language_score(state: essay_state):
    essay = state["essay"]
    response = structured_llm.invoke(
        f"give the feedback and score out of 10 on the given essay on the basis of language.\nessay:{essay}"
    )
    return {"language_feedback": response.feedback, "language_score": response.score}


def get_clarity_score(state: essay_state):
    essay = state["essay"]
    response = structured_llm.invoke(
        f"give the feedback and score out of 10 on the given essay on the basis of clarity of thought.\nessay:{essay}"
    )
    return {"clarity_feedback": response.feedback, "clarity_score": response.score}


graph.add_node("get_essay", get_essay)
graph.add_node("get_language", get_language_score)
graph.add_node("get_clarity", get_clarity_score)

graph.add_edge(START, "get_essay")
graph.add_edge("get_essay", "get_language")
graph.add_edge("get_essay", "get_clarity")
graph.add_edge("get_language", END)
graph.add_edge("get_clarity", END)

workflow = graph.compile()

final_state = workflow.invoke({"topic": "cricket"})

print(final_state["language_feedback"])

workflow = graph.compile()
