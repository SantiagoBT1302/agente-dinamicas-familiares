"""
Entry point para Vercel — importa la app FastAPI desde app/main.py
Vercel busca automáticamente este archivo en api/index.py
"""
import sys
import os

# Agrega la raíz del proyecto al path para que app/ sea importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: F401 — Vercel lo descubre automáticamente
