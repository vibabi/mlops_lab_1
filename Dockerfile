FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.10-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY config/ ./config/
COPY dvc.yaml ./
COPY dvc.lock ./

RUN mkdir -p data/processed data/raw models

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "src/train.py"]
CMD ["--n_estimators", "15", "--max_depth", "6"]