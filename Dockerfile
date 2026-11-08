# Backend API image.  Build: docker build -t econocausal .   Run: docker run -p 8000:8000 econocausal
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
EXPOSE 8000
CMD ["uvicorn", "econocausal.api:app", "--host", "0.0.0.0", "--port", "8000"]