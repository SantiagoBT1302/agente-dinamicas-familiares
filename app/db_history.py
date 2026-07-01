from databricks import sql
from app.config import settings
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger(__name__)

CATALOG = "workspace"
TABLE   = f"{CATALOG}.gold.chat_sessions"


def _get_connection():
    return sql.connect(
        server_hostname=settings.databricks_host.replace("https://", ""),
        http_path=settings.databricks_http_path,
        access_token=settings.databricks_token,
    )


def init_chat_table():
    """Crea la tabla de sesiones si no existe."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {TABLE} (
                        session_id      STRING        NOT NULL,
                        message_id      STRING        NOT NULL,
                        timestamp       TIMESTAMP     NOT NULL,
                        role            STRING        NOT NULL,
                        content         STRING        NOT NULL,
                        elapsed_seconds DOUBLE
                    ) USING DELTA
                    PARTITIONED BY (session_id)
                """)
        logger.info(f"✓ Tabla {TABLE} lista.")
    except Exception as e:
        logger.error(f"Error creando tabla de sesiones: {e}")


def load_history(session_id: str) -> list[dict]:
    """Carga el historial de una sesión ordenado cronológicamente."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT role, content
                    FROM {TABLE}
                    WHERE session_id = '{session_id}'
                    ORDER BY timestamp ASC
                """)
                rows = cursor.fetchall()
                return [{"role": row[0], "content": row[1]} for row in rows]
    except Exception as e:
        logger.error(f"Error cargando historial de sesión {session_id}: {e}")
        return []


def save_messages(session_id: str, user_msg: str, assistant_msg: str, elapsed: float):
    """Guarda el par usuario/asistente en la tabla de sesiones."""
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        user_id      = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())

        def esc(text: str) -> str:
            return text.replace("'", "''")

        with _get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {TABLE}
                        (session_id, message_id, timestamp, role, content, elapsed_seconds)
                    VALUES
                        ('{session_id}', '{user_id}',      '{now}', 'user',      '{esc(user_msg)}',      0.0),
                        ('{session_id}', '{assistant_id}', '{now}', 'assistant', '{esc(assistant_msg)}', {elapsed})
                """)
        logger.info(f"✓ Mensajes guardados en sesión {session_id}")
    except Exception as e:
        logger.error(f"Error guardando mensajes en sesión {session_id}: {e}")


def get_sessions(limit: int = 50) -> list[dict]:
    """Lista las sesiones recientes con el primer mensaje del usuario."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT session_id,
                           MIN(timestamp)  AS inicio,
                           MAX(timestamp)  AS ultimo,
                           COUNT(*)        AS total_mensajes,
                           FIRST(content)  AS primer_mensaje
                    FROM {TABLE}
                    WHERE role = 'user'
                    GROUP BY session_id
                    ORDER BY ultimo DESC
                    LIMIT {limit}
                """)
                rows = cursor.fetchall()
                cols = ["session_id", "inicio", "ultimo", "total_mensajes", "primer_mensaje"]
                return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        logger.error(f"Error listando sesiones: {e}")
        return []


def delete_session(session_id: str) -> bool:
    """Elimina todos los mensajes de una sesión."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    DELETE FROM {TABLE}
                    WHERE session_id = '{session_id}'
                """)
        logger.info(f"✓ Sesión {session_id} eliminada.")
        return True
    except Exception as e:
        logger.error(f"Error eliminando sesión {session_id}: {e}")
        return False


def get_session_messages(session_id: str) -> list[dict]:
    """Devuelve todos los mensajes de una sesión para mostrar el chat completo."""
    try:
        with _get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT role, content, timestamp, elapsed_seconds
                    FROM {TABLE}
                    WHERE session_id = '{session_id}'
                    ORDER BY timestamp ASC
                """)
                rows = cursor.fetchall()
                cols = ["role", "content", "timestamp", "elapsed_seconds"]
                return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        logger.error(f"Error obteniendo mensajes de sesión {session_id}: {e}")
        return []
