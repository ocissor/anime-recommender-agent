from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import HumanMessage
import uuid 
import pickle
from fastapi.responses import StreamingResponse
from src.graph import recommend_anime

from dotenv import load_dotenv
load_dotenv()

conversation_histories = {}

app = FastAPI()

class AnimeRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None
    

@app.get("/")
async def chat_endpoint():

    return {"response": "Welcome to the Anime Recommender AI! How can I help you today?"}

@app.post("/chat")
async def chat_endpoint(request: AnimeRequest):
    # Append user message to state
    if request.thread_id is None:
        request.thread_id = str(uuid.uuid4())
    
    if request.thread_id not in conversation_histories.keys():
        conversation_histories[request.thread_id] = []

    previous_messages = conversation_histories.get(request.thread_id)
    # Invoke the graph asynchronously with current state
    if request.user_input.strip() == "":
        return {'response':"please do not send an empty message"}
        # input = {"messages": previous_messages}
    
    input = {"messages": previous_messages, 'user_input': request.user_input.strip()}

    config = {"configurable": {"thread_id": request.thread_id}}

    async def stream_graph():
        async for step in recommend_anime.astream(input,config=config):
            # Serialize Python object to bytes
            if '__interrupt__' not in step:
                key = list(step.keys())[0]
                conversation_histories[request.thread_id] = list(step[key].get('messages',[]))
            
            yield pickle.dumps(step)
    

    return StreamingResponse(stream_graph(), media_type="application/octet-stream")


