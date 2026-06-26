from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class Feedback_State(TypedDict):
    feedback: str
    sentiment: str
    diagnosis: str
    response: str


class Sentiment_Model(BaseModel):
    sentiment: Annotated[
        Literal["positive", "negative"],
        Field(..., description="sentiment of the feedback"),
    ]


structure_model = llm_model.with_structured_output(Sentiment_Model)


graph = StateGraph(Feedback_State)


def get_sentiment(state: Feedback_State):
    feedback = state["feedback"]

    response = Sentiment_Model.invoke(
        f"give the sentiment of the feedback weather it is positive or negative in a single word.\nfeedback: {feedback}"
    )

    return {"sentiment": response.content}


def check_sentiment(state: Feedback_State) -> Literal["positive", "negative"]:
    sentiment = state["sentiment"]

    if sentiment == "positive":
        return "positive"
    else:
        return "negative"


def positive_response(state: Feedback_State):
    feedback = state["feedback"]
    response = llm_model.invoke(
        f"give the positive response for the feedback.\nfeedback: {feedback}"
    )

    return {"response": response.content}


def negative_response(state: Feedback_State):
    feedback = state["feedback"]
    response = llm_model.invoke(
        f"you are the automatic responding agent, you will craft the response based on the user feedback. give the negative response for the feedback.\nfeedback: {feedback}"
    )

    return {"response": response.content}


def get_diagnosis(state: Feedback_State):
    feedback = state["feedback"]
    response = llm_model.invoke(
        f"give the diagnosis for the feedback.\nfeedback: {feedback}"
    )

    return {"diagnosis": response.content}


graph.add_node("get_sentiment", get_sentiment)
graph.add_node("get_diagnosis", get_diagnosis)
graph.add_node("positive_response", positive_response)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "get_sentiment")
graph.add_conditional_edges(
    "get_sentiment",
    check_sentiment,
    {"positive": "positive_response", "negative": "negative_response"},
)
graph.add_edge("negative_response", "get_diagnosis")
graph.add_edge("get_diagnosis", END)
graph.add_edge("positive_response", END)

workflow = graph.compile()

final_state = workflow.invoke(
    {
        "feedback": "I’m really disappointed with this product. It stopped working after just two weeks of use. The build quality feels cheap, and the performance is much slower than advertised. I contacted customer support, but they were unhelpful and took days to respond. Definitely not worth the money."
    }
)

print(final_state["response"])
