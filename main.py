import os
import uuid
from typing import Dict

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Gemini Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Session:
    def __init__(self):
        self.chat = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
        ).start_chat(history=[])

sessions: Dict[str, Session] = {}

class StartResponse(BaseModel):
    session_id: str
    welcome_message: str

class MessageRequest(BaseModel):
    session_id: str
    text: str

class MessageResponse(BaseModel):
    text: str

@app.post("/chat/start", response_model=StartResponse)
async def start_chat():
    session_id = str(uuid.uuid4())
    sessions[session_id] = Session()
    welcome_message = (
        "Привет! "
        "Напишите сообщение для начала диалога."
    )
    return StartResponse(session_id=session_id, welcome_message=welcome_message)

@app.post("/chat/message", response_model=MessageResponse)
async def chat_message(request: MessageRequest):
    session = sessions.get(request.session_id)
    response = session.chat.send_message(request.text)
    return MessageResponse(text=response.text)
