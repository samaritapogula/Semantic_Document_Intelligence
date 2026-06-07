
FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


COPY requirements.txt .


RUN apt-get update && \
    apt-get install -y build-essential poppler-utils libgl1-mesa-glx && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    rm -rf /root/.cache/pip && \
    apt-get clean


COPY . /app

ENV MODEL_LOCAL_ONLY=1


CMD ["python", "src/main.py"]
