FROM python:3.10-slim

# Install system dependencies for PyMuPDF and ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH="/app/LawCaseIntelligence"
ENV PYTHONUNBUFFERED=1

# Expose the port Render uses
EXPOSE 10000

# Use gunicorn with the gevent worker for SocketIO support
# We point to LawCaseIntelligence/app.py and use the 'app' instance
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--worker-class", "gevent", "--workers", "1", "--timeout", "300", "LawCaseIntelligence.app:app"]
