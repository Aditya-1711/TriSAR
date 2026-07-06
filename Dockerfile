FROM python:3.10-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY trishield_core/ trishield_core/

# Create an output directory for artifacts
RUN mkdir -p artifacts

# The default command will run the simulation
# and generate the trishield_sim.gif inside the mapped /app/artifacts volume
CMD ["python", "trishield_core/simulation.py"]
