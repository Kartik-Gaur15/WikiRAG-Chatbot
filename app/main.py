from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

from app.rag import WikipediaRAG

load_dotenv()

app = FastAPI(title="WikiRAG-Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = WikipediaRAG()

class ChatRequest(BaseModel):
    query: str
    history: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = rag_system.query(request.query, chat_history=request.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        workers=1,
        reload=True
    )
