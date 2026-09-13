FROM python:3.9-slim

WORKDIR /app

# Copia só o requirements primeiro para aproveitar o cache de camadas do Docker
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY src/ src/
COPY config/ config/
COPY app/ app/

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
