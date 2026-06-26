from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()


class evaluate_schema(BaseModel):
    feedback: str
    score: int


class EssayState(TypedDict):
    topic: str
    essay: str
    feedback: str
    score: int


graph = StateGraph(EssayState)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def generate_essay(state: EssayState) -> EssayState:
    topic = state["topic"]
    response = model.invoke(f"generate a small essay on the topic {topic}")
    return {"essay": response.content}


structured_model = model.with_structured_output(evaluate_schema)


def evaluate_essay(state: EssayState) -> EssayState:
    essay = state["essay"]
    response = structured_model.invoke(
        f"Evaluate the following essay based on language, analysis and give the score between 1-10.\nEssay->{essay}"
    )
    return {"feedback": response.feedback, "score": response.score}


def improve_essay(state: EssayState) -> EssayState:
    essay = state["essay"]
    feedback = state["feedback"]
    response = model.invoke(
        f"improve the essay based on the feedback.\n Essay->{essay}.\nFeedback->{feedback}"
    )
    return {"essay": response.content}


def check_score(state: EssayState) -> Literal["end", "improve"]:
    score = state["score"]

    return "end" if score > 6 else "improve"


graph.add_node("generate_essay", generate_essay)
graph.add_node("evaluate_essay", evaluate_essay)
graph.add_node("improve_essay", improve_essay)

graph.add_edge(START, "generate_essay")
graph.add_edge("generate_essay", "evaluate_essay")
graph.add_conditional_edges(
    "evaluate_essay", check_score, {"improve": "improve_essay", "end": END}
)

graph.add_edge("improve_essay", "evaluate_essay")

workflow = graph.compile()

final_state = workflow.invoke({"topic": "cricket"})

print(final_state["score"])
