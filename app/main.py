import sys
import logging
import traceback
import time
import uuid
import threading

# ── Imports defensivos: si algo falla, el servidor arranca igual
# y el error aparece en los logs de Cloud Run ──────────────────
try:
    from fastapi import FastAPI, HTTPException, Cookie, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from contextlib import asynccontextmanager
    print("✓ FastAPI importado", flush=True)
except Exception as e:
    print(f"✗ ERROR importando FastAPI: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)   # sin FastAPI no hay nada que hacer

try:
    from app.config import settings
    print("✓ Config importado", flush=True)
except Exception as e:
    print(f"✗ ERROR importando config: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    from app.agent import build_agent, format_history
    AGENT_OK = True
    print("✓ Agent importado", flush=True)
except Exception as e:
    AGENT_OK = False
    print(f"✗ ERROR importando agent: {e}", flush=True)
    traceback.print_exc()

DB_ERROR = None
try:
    from app.db_history import load_history, save_messages, get_sessions, get_session_messages, delete_session
    DB_OK = True
    print("✓ DB history importado", flush=True)
except Exception as e:
    DB_OK = False
    DB_ERROR = f"{type(e).__name__}: {e}"
    print(f"✗ ERROR importando db_history: {DB_ERROR}", flush=True)
    traceback.print_exc()
    # Stubs para que el servidor funcione aunque falle la BD
    def load_history(session_id): return []
    def save_messages(session_id, user_msg, assistant_msg, elapsed): pass
    def get_sessions(limit=50): return []
    def get_session_messages(session_id): return []
    def delete_session(session_id): return True

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# ── Inicialización lazy del agente ───────────────────────────
_agent_executor = None
_agent_lock = threading.Lock()

def get_agent():
    global _agent_executor
    if _agent_executor is None:
        with _agent_lock:
            if _agent_executor is None:
                logger.info("Construyendo agente...")
                _agent_executor = build_agent()
                logger.info("Agente listo.")
    return _agent_executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("▶ Lifespan startup", flush=True)
    yield
    print("■ Lifespan shutdown", flush=True)


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

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    elapsed_seconds: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": settings.app_version,
        "agent_ok": AGENT_OK,
        "db_ok": DB_OK,
        "db_error": DB_ERROR,   # muestra el error exacto si db_history falla
    }

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    resp: Response,
    session_cookie: str | None = Cookie(default=None, alias="session_id"),
):
    if not AGENT_OK:
        raise HTTPException(status_code=503, detail="Agente no disponible (error de importación)")

    agent_executor = get_agent()
    session_id = request.session_id or session_cookie or str(uuid.uuid4())

    resp.set_cookie(
        key="session_id",
        value=session_id,
        max_age=30 * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
    )

    raw_history = load_history(session_id)
    history = format_history(raw_history) if AGENT_OK else []

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
    save_messages(session_id, request.message, answer, elapsed)

    return ChatResponse(response=answer, session_id=session_id, elapsed_seconds=elapsed)


@app.get("/sessions")
def list_sessions(limit: int = 50):
    return {"sessions": get_sessions(limit)}

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    messages = get_session_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return {"session_id": session_id, "messages": messages}

@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Error eliminando la sesión")
    return {"session_id": session_id, "deleted": True}

@app.post("/sessions/new")
def new_session(resp: Response):
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
    from app.tools import TABLA_DESCRIPCION
    return {"tables": TABLA_DESCRIPCION}
