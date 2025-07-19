from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from src.prompt import anime_prompt
from src.state import AnimeGraphState
from config.database import index
from src.generate_embeddings import AnimeEmbeddings
import os 

load_dotenv()

def anime_recommendation_agent(state : AnimeGraphState) -> AnimeGraphState:
    llm = ChatGroq(
        model_name='llama-3.1-8b-instant',
        temperature=0.1,
        max_tokens=512
    )

    model = AnimeEmbeddings()

    user_input = state['user_input']
    messages = state['messages']

    messages = [msg for msg in messages if msg.content and msg.content.strip() != '']

    query_embedding = model.embeddings.embed_documents([user_input])[0]

    context = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
    )

    context_to_list = [doc['metadata']['description'] for doc in context['matches']]  

    formatted_prompt = anime_prompt.invoke(
        {
            "context": context_to_list,
            "history": messages,
            "question": user_input
        }
    )

    response = llm.invoke(formatted_prompt)

    state['messages'] = list(messages) + [HumanMessage(content = user_input), response]
    
    return state


