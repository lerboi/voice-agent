FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download Silero VAD model at build time
# Can't use `agent.py download-files` because config.py requires env vars
RUN python -c "from livekit.plugins.silero import VAD; VAD.load()"

CMD ["python", "agent.py", "start"]
