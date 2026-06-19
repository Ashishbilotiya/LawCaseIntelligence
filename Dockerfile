FROM python:3.10-slim

# Install system dependencies for PyMuPDF, ChromaDB, and Git (for HF model download)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install sentence-transformers so the Dockerfile can download the embedding model
RUN python -c "import sentence_transformers; print('sentence-transformers OK')"

# Copy project files
COPY . .

# Pre-download embedding model into image (faster cold-start, no HF rate limits on runtime)
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN mkdir -p /app/.cache/huggingface && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', cache_folder='/app/.cache/huggingface')"

# Set environment variables
ENV PYTHONPATH="/app/LawCaseIntelligence"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

# Expose the port Render uses
EXPOSE 10000

# Use gunicorn with the gevent worker for SocketIO support
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--worker-class", "gevent", "--workers", "1", "--timeout", "600", "--graceful-timeout", "120", "LawCaseIntelligence.app:app"]
