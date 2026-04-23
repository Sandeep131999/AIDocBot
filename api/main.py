from fastapi import FastAPI
#This is from Fast API to store the models
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

app = FastAPI()

# Env variables
gemini_url = os.getenv("GEMINI_BASE_URL")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# System prompt (strict anti-hallucination)
system_prompt = """
You are SCII company chatbot.

Rules:
- Answer based on general understanding if exact company data is not available.
- Do NOT make up specific products, tools, or claims about SCII.
- If a question asks for specific internal details, say "I don't know".
- Keep answers realistic and general, not fabricated.
"""

# Request body model
class ChatRequest(BaseModel):
    mode : str
    user: str


#Dynamic Prompt Creation 
def get_prompt(mode: str, user: str):

    if mode == "basic":
        return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user}
            ]
    elif mode == "fewshot":
        return [
            {"role": "system", "content":system_prompt},
            {"role": "user", "content": "What ai products does scii have ? "},
            {"role": "assistant", "content": "I don't know ?"},
            {"role": "user", "content": user} 
        ]
    elif mode == "react":
        return [
            {"role": "system", "content":
             system_prompt +
             "\n\nUse this format:\nThought: think carefully\nFinal Answer: give factual answer only."},
            {"role": "user", "content": user}
        ]

    return [
        {"role": "system", "content": "Default mode"},
        {"role": "user", "content": user},
    ]
        

# Async client (global)
gem = AsyncOpenAI(base_url=gemini_url, api_key=gemini_api_key)

@app.post("/chat")
async def chat(request: ChatRequest):
    try :
        response = await gem.chat.completions.create(
            model="gemini-2.5-flash",
            messages= get_prompt(request.mode.lower(),request.user)
        )
        return {
            "response": response.choices[0].message.content
        }
    except :
        return {
            "response":"Sorry for the incovience we will let you back ?"
        }


#Today Learned 
#Docker need to 

#Still needs to implemet rag 
#Still needs to learn chroma db 
#Still need to use langchain
#Still need to use langgraph
