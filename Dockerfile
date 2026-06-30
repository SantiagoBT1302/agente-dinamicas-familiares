FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código fuente
COPY app/ ./app/

# Puerto (Cloud Run inyecta $PORT, por defecto 8080)
EXPOSE 8080

# Arranque — usa la variable $PORT que Cloud Run inyecta automáticamente
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
