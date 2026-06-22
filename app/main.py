from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.agent import build_agent, format_history
from app.config import settings
import logging
import traceback
import time

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Cache del agente (se construye una sola vez al arrancar)
agent_executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    logger.info("Inicializando agente...")
    agent_executor = build_agent()
    logger.info("Agente listo.")
    yield
    logger.info("Apagando agente.")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modelos de entrada/salida ────────────────────────────────
class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

class ChatResponse(BaseModel):
    response: str
    elapsed_seconds: float

# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": settings.app_version}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint síncrono — FastAPI lo ejecuta automáticamente en un thread pool,
    lo que evita bloquear el event loop con las llamadas bloqueantes de LangChain.
    """
    if not agent_executor:
        raise HTTPException(status_code=503, detail="Agente no inicializado")

    start = time.time()
    try:
        history = format_history([m.model_dump() for m in request.history])

        result = agent_executor.invoke({
            "input": request.message,
            "chat_history": history,
        })

        # Normalizar output (Gemini puede devolver str, list o dict)
        raw_output = result.get("output", "")
        if isinstance(raw_output, list):
            response = " ".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in raw_output
            )
        elif isinstance(raw_output, dict):
            response = raw_output.get("text", str(raw_output))
        else:
            response = str(raw_output) if raw_output else "No se pudo generar una respuesta."

    except Exception as e:
        logger.error(f"Error en el agente:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        response=response,
        elapsed_seconds=round(time.time() - start, 2),
    )

@app.get("/tables")
def list_tables():
    """Lista las tablas disponibles en el lakehouse."""
    from app.tools import TABLA_DESCRIPCION
    return {"tables": TABLA_DESCRIPCION}
