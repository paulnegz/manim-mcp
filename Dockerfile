FROM python:3.11-slim

# Install system dependencies for manimgl (cached layer)
# LaTeX packages required by manimgl tex_templates.yml:
# - Default template: babel, inputenc, fontenc, amsmath, amssymb, dsfont,
#   setspace, tipa, relsize, textcomp, mathrsfs, calligra, wasysym,
#   ragged2e, physics, xcolor, microtype, pifont
# - Additional templates may need: fourier, txfonts, pxfonts, mathastext, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    git \
    gcc \
    g++ \
    build-essential \
    # FFmpeg for video encoding
    ffmpeg \
    # Cairo/Pango for graphics
    libcairo2-dev \
    libpango1.0-dev \
    pkg-config \
    python3-dev \
    # === LaTeX packages (comprehensive for manimgl) ===
    texlive-latex-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-plain-generic \
    texlive-pstricks \
    # Specific packages manimgl needs
    tipa \
    cm-super \
    lmodern \
    # DVI to SVG/PNG conversion (required by manimgl)
    dvisvgm \
    dvipng \
    # === OpenGL dependencies ===
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglfw3-dev \
    libglfw3 \
    # Headless rendering
    xvfb \
    xauth \
    # Other
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
    anthropic \
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
