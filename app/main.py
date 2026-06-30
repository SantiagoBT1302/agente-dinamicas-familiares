from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.agent import build_agent, format_history
from app.config import settings
from app.db_history import init_chat_table, load_history, save_messages, get_sessions, get_session_messages
import logging
import traceback
import time
import uuid

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
    logger.info("Inicializando tabla de sesiones en Databricks...")
    init_chat_table()
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
    role: str      # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None   # None = nueva sesión

class ChatResponse(BaseModel):
    response: str
    session_id: str                 # Siempre se devuelve para que el cliente lo reutilice
    elapsed_seconds: float

# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": settings.app_version}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint principal del agente con persistencia de historial en Databricks.
    - Si no se envía session_id se crea una nueva sesión.
    - El historial se carga desde Databricks automáticamente.
    """
    if not agent_executor:
        raise HTTPException(status_code=503, detail="Agente no inicializado")

    # Gestionar sesión
    session_id = request.session_id or str(uuid.uuid4())

    # Cargar historial desde Databricks
    raw_history = load_history(session_id)
    history = format_history(raw_history)

    start = time.time()
    try:
        result = agent_executor.invoke({
            "input": request.message,
            "chat_history": history,
        })

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

    elapsed = round(time.time() - start, 2)

    # Guardar par usuario/asistente en Databricks
    save_messages(session_id, request.message, response, elapsed)

    return ChatResponse(
        response=response,
        session_id=session_id,
        elapsed_seconds=elapsed,
    )

@app.get("/sessions")
def list_sessions(limit: int = 50):
    """Lista las sesiones de chat más recientes."""
    return {"sessions": get_sessions(limit)}

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Devuelve el historial completo de una sesión para mostrar el chat."""
    messages = get_session_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return {"session_id": session_id, "messages": messages}

@app.get("/tables")
def list_tables():
    """Lista las tablas disponibles en el lakehouse."""
    from app.tools import TABLA_DESCRIPCION
    return {"tables": TABLA_DESCRIPCION}
