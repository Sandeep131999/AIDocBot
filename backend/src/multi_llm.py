from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os

def build_llm():
    """
    Enterprise Multi-LLM with local Ollama as final fallback.
    Order: Gemini (quality) -> Groq (speed) -> OpenRouter (free) -> Ollama (local/offline)
    """
    llms = []
    order = [x.strip() for x in os.getenv("LLM_PROVIDER_ORDER","gemini,groq,openrouter,ollama").split(",")]

    # 1. Gemini - Quality
    if "gemini" in order and os.getenv("GEMINI_API_KEY"):
        try:
            llms.append(ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL","gemini-2.0-flash"),
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE",0.1)),
                max_output_tokens=int(os.getenv("LLM_MAX_TOKENS",1024))
            ))
            print("[LLM] Gemini added")
        except Exception as e: print(f"[LLM] Gemini skip: {e}")

    # 2. Groq - Speed
    if "groq" in order and os.getenv("GROQ_API_KEY"):
        try:
            llms.append(ChatGroq(
                model=os.getenv("GROQ_MODEL","llama-3.1-8b-instant"),
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE",0.1)),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS",1024))
            ))
            print("[LLM] Groq added")
        except Exception as e: print(f"[LLM] Groq skip: {e}")

    # 3. OpenRouter - Free tier rotation
    if "openrouter" in order and os.getenv("OPENROUTER_API_KEY"):
        try:
            llms.append(ChatOpenAI(
                model=os.getenv("OPENROUTER_MODEL","meta-llama/llama-3.1-8b-instruct:free"),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=float(os.getenv("LLM_TEMPERATURE",0.1)),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS",1024)),
                default_headers={
                    "HTTP-Referer": os.getenv("APP_URL","http://localhost:8000"),
                    "X-Title": os.getenv("APP_NAME","Enterprise-RAG")
                }
            ))
            print("[LLM] OpenRouter added")
        except Exception as e: print(f"[LLM] OpenRouter skip: {e}")

    # 4. Ollama - LOCAL OPEN SOURCE (NO API KEY NEEDED) - FINAL FALLBACK
    if "ollama" in order:
        try:
            ollama_model = os.getenv("OLLAMA_MODEL","llama3.1:8b")
            ollama_base = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
            if ollama_model: # Only if set
                llms.append(ChatOllama(
                    model=ollama_model,
                    base_url=ollama_base,
                    temperature=float(os.getenv("LLM_TEMPERATURE",0.1)),
                    num_predict=int(os.getenv("LLM_MAX_TOKENS",1024)),
                ))
                print(f"[LLM] Ollama LOCAL added: {ollama_model} @ {ollama_base}")
        except Exception as e: print(f"[LLM] Ollama skip (is ollama serve running?): {e}")

    if not llms:
        raise ValueError("No LLMs configured. Check.env or start ollama serve")

    # LangChain native fallback chain
    primary = llms[0]
    fallbacks = llms[1:]
    print(f"[LLM] Chain: {' -> '.join([l.__class__.__name__ for l in llms])}")
    return primary.with_fallbacks(fallbacks) if fallbacks else primary

# Singleton for FastAPI
_llm_instance = None
def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = build_llm()
    return _llm_instance