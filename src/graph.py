from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import sys
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from src.state import AnimeGraphState
from src.recommendation_agent import anime_recommendation_agent, anime_recommendation_agent_multi_query


graph = StateGraph(AnimeGraphState)

graph.add_node("anime_recommender",anime_recommendation_agent_multi_query)

graph.add_edge(START, "anime_recommender")

graph.add_edge("anime_recommender", END)

checkpointer = InMemorySaver()
recommend_anime = graph.compile(checkpointer=checkpointer)

# Save Mermaid PNG to a file
png_bytes = recommend_anime.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)


if __name__ == "__main__":
    # Example usage

    config = {"configurable": {"thread_id": "1"}}
    input = {"messages": [], "user_input": "Recommend me anime with less than 100 episodes."}
    out = recommend_anime.invoke(input, config)



