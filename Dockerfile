# Purpose: Production inference image used for local testing, Vertex AI, and SageMaker.
# Build:
#   docker build -t xai-inference:local .
# Run locally:
#   docker run --rm -p 8080:8080 xai-inference:local

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serving.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-serving.txt

COPY app.py train_model.py breast-cancer.csv ./

# Train model artifacts during image build so the container is self-contained.
RUN python3 train_model.py

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
