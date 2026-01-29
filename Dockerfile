FROM python:3.11-slim

# Install system dependencies for manimgl (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    build-essential \
    ffmpeg \
    libcairo2-dev \
    libpango1.0-dev \
    pkg-config \
    python3-dev \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    dvipng \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglfw3-dev \
    libglfw3 \
    xvfb \
    xauth \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install heavy dependencies first (cached unless versions change)
# These are the slowest: manimgl (~180s), torch, chromadb, sentence-transformers
RUN pip install --no-cache-dir \
    manimgl \
    chromadb \
    sentence-transformers \
    google-genai \
    mcp \
    pydantic \
    pydantic-settings \
    boto3 \
    aiosqlite \
    pydub \
    datasets

# Copy project files and install (fast - deps already cached)
COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[rag]"

ENV MANIM_MCP_SERVER_HOST=0.0.0.0
ENV PATH="/usr/local/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "manim_mcp.cli", "serve"]
