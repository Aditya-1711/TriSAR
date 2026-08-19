FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for headless rendering and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and project layout
COPY . .

# Set Python path to app root
ENV PYTHONPATH=/app
ENV BATCH_MODE=1

# Default command: Run single demonstration episode
CMD ["python", "experiments/run_episode.py"]
