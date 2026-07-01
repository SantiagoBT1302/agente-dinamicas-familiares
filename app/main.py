from fastapi import FastAPI, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.agent import build_agent, format_history
from app.config import settings
from app.db_history import load_history, save_messages, get_sessions, get_session_messages, delete_session
import logging
import traceback
import time
import uuid
import threading

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# ── Inicialización lazy del agente ───────────────────────────
# El agente se construye en la primera solicitud, no al arrancar.
# Esto garantiza que Cloud Run pueda hacer el health check inmediatamente.
_agent_executor = None
_agent_lock = threading.Lock()

def get_agent():
    """Devuelve el agente, construyéndolo la primera vez (thread-safe)."""
    global _agent_executor
    if _agent_executor is None:
        with _agent_lock:
            if _agent_executor is None:   # double-check tras el lock
                logger.info("Construyendo agente por primera vez...")
                _agent_executor = build_agent()
                logger.info("Agente listo.")
    return _agent_executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup mínimo: solo registra que la app arrancó.
    # Todo lo costoso (LangSmith, Databricks) se inicializa en la primera solicitud.
    logger.info("Aplicación iniciada. Agente y DB se inicializan en primera solicitud.")
    yield
    logger.info("Apagando aplicación.")


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
    session_id: str | None = None   # None = nueva sesión (o se usa la cookie)

class ChatResponse(BaseModel):
    response: str
    session_id: str
    elapsed_seconds: float

# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — responde inmediatamente sin esperar al agente."""
    return {"status": "ok", "version": settings.app_version}

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    resp: Response,
    session_cookie: str | None = Cookie(default=None, alias="session_id"),
):
    """
    Endpoint principal del agente con persistencia de historial en Databricks.
    - Prioridad de session_id: body > cookie > nueva sesión
    - El session_id se guarda en cookie automáticamente
    """
    agent_executor = get_agent()

    # Prioridad: session_id en body > cookie > nueva sesión
    session_id = request.session_id or session_cookie or str(uuid.uuid4())

    # Guardar session_id en cookie (30 días)
    resp.set_cookie(
        key="session_id",
        value=session_id,
        max_age=30 * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
    )

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
            answer = " ".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in raw_output
            )
        elif isinstance(raw_output, dict):
            answer = raw_output.get("text", str(raw_output))
        else:
            answer = str(raw_output) if raw_output else "No se pudo generar una respuesta."

    except Exception as e:
        logger.error(f"Error en el agente:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - start, 2)

    # Guardar par usuario/asistente en Databricks
    save_messages(session_id, request.message, answer, elapsed)

    return ChatResponse(
        response=answer,
        session_id=session_id,
        elapsed_seconds=elapsed,
    )


@app.get("/sessions")
def list_sessions(limit: int = 50):
    """Lista las sesiones de chat más recientes."""
    return {"sessions": get_sessions(limit)}

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Devuelve el historial completo de una sesión."""
    messages = get_session_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return {"session_id": session_id, "messages": messages}

@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    """Elimina todos los mensajes de una sesión."""
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Error eliminando la sesión")
    return {"session_id": session_id, "deleted": True}

@app.post("/sessions/new")
def new_session(resp: Response):
    """Inicia una nueva sesión limpiando la cookie actual."""
    new_id = str(uuid.uuid4())
    resp.set_cookie(
        key="session_id",
        value=new_id,
        max_age=30 * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
    )
    return {"session_id": new_id}

@app.get("/tables")
def list_tables():
    """Lista las tablas disponibles en el lakehouse."""
    from app.tools import TABLA_DESCRIPCION
    return {"tables": TABLA_DESCRIPCION}
