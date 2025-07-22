from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from src.state import AnimeGraphState
from config.database import index
from src.generate_embeddings import AnimeEmbeddings
from src.prompt import multi_query_prompt, anime_prompt, prompt_rag_fusion, summary_prompt, prompt_hyde
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from utils.helpers import get_unique_union
from langchain_google_genai import ChatGoogleGenerativeAI


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


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def anime_recommendation_agent_multi_query(state : AnimeGraphState) -> AnimeGraphState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=os.getenv("GEMINI_API_KEY"))

    model = AnimeEmbeddings()

    user_input = state['user_input']

    generate_queries = (
        multi_query_prompt 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    multi_queries = generate_queries.invoke({"question":user_input})

    multi_query_context = []

    for query in multi_queries:
        query_embedding = model.embeddings.embed_documents([query])[0]

        context = index.query(
        vector=query_embedding,
        top_k=2,
        include_metadata=True
        )

        context_to_list = [doc['metadata']['description'] for doc in context['matches']]

        multi_query_context.append(context_to_list)
    
    unique_context_docs = get_unique_union(multi_query_context)

    updated_summary_prompt = summary_prompt.invoke({"context":unique_context_docs})

    context_summary = llm.invoke(updated_summary_prompt).content

    messages = state['messages']

    formatted_prompt = anime_prompt.invoke(
        {
            "context": context_summary,
            "history": messages,
            "question": user_input
        }
    )

    response = llm.invoke(formatted_prompt)

    print(response.content)

    state['messages'] = list(messages) + [HumanMessage(content = user_input), response]

    return state


def reciprocal_rank_fusion(docs : list[list], k = 60):
    rank_values = {}
    for sublist in docs:
        for i, doc in enumerate(sublist):
            doc_str = dumps(doc)

            if doc_str not in rank_values.keys():
                rank_values[doc_str] = 0
            
            rank_values[doc_str] += 1/(k + i)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(rank_values.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def anime_recommendation_agent_rag_fusion(state : AnimeGraphState) -> AnimeGraphState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=os.getenv("GEMINI_API_KEY"))

    model = AnimeEmbeddings()

    user_input = state['user_input']

    generate_queries = (
        prompt_rag_fusion 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    multi_queries = generate_queries.invoke({"question":user_input})

    multi_query_context = []

    for query in multi_queries:
        query_embedding = model.embeddings.embed_documents([query])[0]

        context = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
        )

        context_to_list = [doc['metadata']['description'] for doc in context['matches']]

        multi_query_context.append(context_to_list)


    reranked_context_docs = reciprocal_rank_fusion(multi_query_context)

    messages = state['messages']

    formatted_prompt = anime_prompt.invoke(
        {
            "context": reranked_context_docs,
            "history": messages,
            "question": user_input
        }
    )

    response = llm.invoke(formatted_prompt)

    state['messages'] = list(messages) + [HumanMessage(content = user_input), response]


    return state


def anime_recommendation_agent_HyDE(state : AnimeGraphState) -> AnimeGraphState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=os.getenv("GEMINI_API_KEY"))

    model = AnimeEmbeddings()

    user_input = state['user_input']

    generate_docs_for_retrieval = (
        prompt_hyde 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    detailed_input = generate_docs_for_retrieval.invoke({"question":user_input})
    
    query_embedding = model.embeddings.embed_documents([detailed_input])[0]

    context = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
    )

    context_to_list = [doc['metadata']['description'] for doc in context['matches']]

    messages = state['messages']

    formatted_prompt = anime_prompt.invoke(
        {
            "context": context_to_list,
            "history": messages,
            "question": detailed_input
        }
    )

    response = llm.invoke(formatted_prompt)

    state['messages'] = list(messages) + [HumanMessage(content = user_input), response]


    return state



