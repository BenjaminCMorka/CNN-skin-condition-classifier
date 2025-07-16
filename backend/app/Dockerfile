FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY app/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model.py .
COPY utils.py .
COPY download_model.py .

EXPOSE 8000

CMD ["sh", "-c", "python download_model.py && python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
