# manim-mcp

Text-to-video animation powered by [manimgl](https://github.com/3b1b/manim) (3Blue1Brown's library) and a multi-agent LLM pipeline. Describe what you want to see, and get a rendered animation back.

Works as a **CLI tool**, an **LLM-powered agent**, or an **MCP server** for integration with AI assistants like Claude.

## Examples

| Circle to Square Transform | 3D Rotating Cube |
|---------------------------|------------------|
| `manim-mcp gen "Transform a blue circle into a red square"` | `manim-mcp gen "A 3D cube rotating. Use ThreeDScene."` |
| ![Circle to Square](assets/circle_to_square.gif) | ![Rotating Cube](assets/rotating_cube_preview.gif) |

## Features

- **RAG-powered code generation** - Uses 3,100+ indexed 3Blue1Brown scenes for high-quality examples
- **Multi-agent pipeline** - Concept analysis, scene planning, code generation, and code review
- **Self-learning** - Stores error patterns and fixes for continuous improvement
- **Multi-provider LLM** - Supports Google Gemini and Anthropic Claude
- **Audio narration** - Optional TTS with Gemini voices
- **ChromaDB integration** - Vector similarity search for relevant code examples

## Quick Start

```bash
pip install -e ".[rag]"
```

### Prerequisites

- Python 3.11+
- [manimgl](https://github.com/3b1b/manim) installed: `pip install manimgl`
- A [Google Gemini API key](https://ai.google.dev/) set as `MANIM_MCP_GEMINI_API_KEY`
- Optional: ChromaDB (for RAG), ffmpeg (for audio mixing), LaTeX (for math text), S3/MinIO (for cloud storage)

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# LLM Provider
MANIM_MCP_GEMINI_API_KEY=your-gemini-api-key
MANIM_MCP_GEMINI_MODEL=gemini-2.5-flash-preview-05-20  # default

# Alternative: Claude
# MANIM_MCP_LLM_PROVIDER=claude
# MANIM_MCP_CLAUDE_API_KEY=your-claude-api-key
# MANIM_MCP_CLAUDE_MODEL=claude-sonnet-4-20250514

# RAG (ChromaDB)
MANIM_MCP_RAG_ENABLED=true
MANIM_MCP_CHROMADB_HOST=localhost
MANIM_MCP_CHROMADB_PORT=8000

# S3 Storage (optional)
MANIM_MCP_S3_ENDPOINT=localhost:9000
MANIM_MCP_S3_ACCESS_KEY=minioadmin
MANIM_MCP_S3_SECRET_KEY=minioadmin
MANIM_MCP_S3_BUCKET=manim-renders
```

## Usage

### Generate an animation

```bash
manim-mcp generate "Animate a matrix transformation showing rotation and scaling"
manim-mcp gen "Visualize the central limit theorem" --quality high --format mp4
```

### Edit an existing animation

```bash
manim-mcp edit <render_id> "Make the vectors red and add axis labels"
```

### List, inspect, delete renders

```bash
manim-mcp list --status completed --limit 10
manim-mcp get <render_id>
manim-mcp delete <render_id> --yes
```

### Agent mode

Let the LLM interpret multi-step requests:

```bash
manim-mcp prompt "Create a video on eigenvectors, then edit it with better colors"
```

### MCP server

Start the Model Context Protocol server for integration with Claude, Cursor, or other MCP clients:

```bash
manim-mcp serve
manim-mcp serve --transport stdio
manim-mcp serve --transport streamable-http
```

### RAG Indexing

Index 3Blue1Brown scenes and manimgl library documentation:

```bash
# Index from 3b1b videos repository
manim-mcp index-3b1b /path/to/3b1b/videos

# Index manimgl library source
manim-mcp index-lib

# Check collection stats
manim-mcp rag-stats
```

## Docker

Run with all dependencies (ChromaDB, MinIO):

```bash
export MANIM_MCP_GEMINI_API_KEY=your-api-key
docker compose up
```

This starts:
- MCP server on port 8000
- ChromaDB on port 8001
- MinIO on ports 9000/9001

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-AGENT PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   prompt ──► ConceptAnalyzer ──► ScenePlanner ──► CodeGenerator ──► CodeReviewer
│                    │                   │                │                │   │
│                    │                   │                │                │   │
│                    ▼                   ▼                ▼                ▼   │
│              ┌─────────────────────────────────────────────────────────┐     │
│              │                    ChromaDB RAG                         │     │
│              │  ┌──────────────┬──────────────┬──────────────────┐    │     │
│              │  │ manim_scenes │  manim_docs  │  error_patterns  │    │     │
│              │  │   (3,138)    │    (470)     │  (self-learning) │    │     │
│              │  └──────────────┴──────────────┴──────────────────┘    │     │
│              └─────────────────────────────────────────────────────────┘     │
│                                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RENDER PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   validated code ──► CodeSandbox ──► manimgl (xvfb) ──► S3 upload ──► URL   │
│         │                                    │                               │
│         │                                    │                               │
│         ▼                                    ▼                               │
│   ┌───────────┐                       ┌───────────────┐                     │
│   │  SQLite   │                       │    MinIO/S3   │                     │
│   │ (tracker) │                       │   (storage)   │                     │
│   └───────────┘                       └───────────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **ConceptAnalyzer** | Extracts domain, complexity, and key concepts from prompts |
| **ScenePlanner** | Designs animation structure, timing, and transitions |
| **CodeGenerator** | Generates manimgl code with RAG few-shot examples |
| **CodeReviewer** | Validates code quality and applies fixes |
| **ChromaDBService** | Vector similarity search across 3,600+ indexed documents |
| **CodeSandbox** | AST-based security validation (blocks dangerous code) |
| **ManimRenderer** | Executes manimgl with xvfb for headless rendering |
| **S3Storage** | Uploads to MinIO/S3 with presigned URLs |
| **RenderTracker** | Persists job metadata in SQLite |

### RAG Collections

| Collection | Documents | Description |
|------------|-----------|-------------|
| `manim_scenes` | 3,138 | Production 3Blue1Brown scene code |
| `manim_docs` | 470 | manimgl library API documentation |
| `error_patterns` | dynamic | Self-learning error/fix patterns |

### Self-Learning

The system learns from every error:
1. **Validation failures** - Stored with fixes when LLM corrects them
2. **Render failures** - Stored for future pattern matching
3. **Successful fixes** - Stored as error→fix pairs for RAG retrieval

This creates a feedback loop where the system improves over time.

## MCP Tools

When running as an MCP server, these tools are available:

| Tool | Description |
|------|-------------|
| `generate_animation` | Create an animation from a text prompt |
| `edit_animation` | Edit an existing animation with instructions |
| `list_renders` | List past renders with pagination and filtering |
| `get_render` | Get full details and a fresh download URL |
| `delete_render` | Permanently delete a render and its files |
| `rag_search` | Search the RAG database for similar scenes |
| `rag_stats` | Get collection statistics |

## Recommended Prompts

The system performs best with mathematical and educational topics that have high RAG coverage:

| Topic | Indexed Scenes | Example Prompts |
|-------|----------------|-----------------|
| Linear Algebra | 810+ | "Animate a matrix transformation", "Show eigenvectors during transformation" |
| Geometry | 568+ | "Visual proof of Pythagorean theorem", "Inscribed angle theorem" |
| Probability | 290+ | "Central limit theorem", "Bayes theorem with updating priors" |
| Calculus | 178+ | "Derivative as tangent slope", "Riemann sums converging to integral" |

## Development

```bash
pip install -e ".[dev,rag]"
pytest
```

## License

MIT
