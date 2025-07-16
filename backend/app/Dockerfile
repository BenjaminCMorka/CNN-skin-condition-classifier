FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY main.py .
COPY model.py .
COPY utils.py .

# Expose port
EXPOSE 8000

# Command to download model from S3, then run uvicorn
CMD ["sh", "-c", "python download_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT"]
