"""Pydantic models for manim-mcp inputs, outputs, and internal state."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class RenderQuality(str, Enum):
    low = "low"            # 480p15
    medium = "medium"      # 720p30
    high = "high"          # 1080p60
    production = "production"  # 1440p60
    fourk = "fourk"        # 2160p60

    @property
    def flag(self) -> str:
        return {"low": "l", "medium": "m", "high": "h", "production": "p", "fourk": "k"}[self.value]


class OutputFormat(str, Enum):
    mp4 = "mp4"
    gif = "gif"
    webm = "webm"
    mov = "mov"
    png = "png"


class RenderStatus(str, Enum):
    pending = "pending"
    generating = "generating"
    rendering = "rendering"
    uploading = "uploading"
    completed = "completed"
    failed = "failed"


# ── Tool Input Models ──────────────────────────────────────────────────

class GenerateAnimationInput(BaseModel):
    prompt: str = Field(..., description="Text description of the animation to create")
    quality: RenderQuality = Field(RenderQuality.medium, description="Render quality")
    format: OutputFormat = Field(OutputFormat.mp4, description="Output format")
    resolution: Optional[str] = Field(None, description="Resolution as WxH (e.g. '1920x1080')")
    fps: Optional[int] = Field(None, description="Frames per second override")
    background_color: Optional[str] = Field(None, description="Background color (e.g. '#000000')")
    transparent: bool = Field(False, description="Transparent background")
    save_last_frame: bool = Field(False, description="Save last frame as image instead of video")


class EditAnimationInput(BaseModel):
    render_id: str = Field(..., description="ID of the animation to edit")
    instructions: str = Field(..., description="What to change (e.g. 'make the circle red')")
    quality: RenderQuality = Field(RenderQuality.medium, description="Render quality")
    format: OutputFormat = Field(OutputFormat.mp4, description="Output format")
    resolution: Optional[str] = Field(None, description="Resolution as WxH (e.g. '1920x1080')")
    fps: Optional[int] = Field(None, description="Frames per second override")
    background_color: Optional[str] = Field(None, description="Background color (e.g. '#000000')")
    transparent: bool = Field(False, description="Transparent background")
    save_last_frame: bool = Field(False, description="Save last frame as image instead of video")


class GetRenderInput(BaseModel):
    render_id: str = Field(..., description="Render job ID")


class ListRendersInput(BaseModel):
    limit: int = Field(20, ge=1, le=100, description="Max results")
    offset: int = Field(0, ge=0, description="Pagination offset")
    status: Optional[RenderStatus] = Field(None, description="Filter by status")


class DeleteRenderInput(BaseModel):
    render_id: str = Field(..., description="Render job ID")


# ── Internal Models ────────────────────────────────────────────────────

class RenderSceneInput(BaseModel):
    """Internal model passed to the renderer — not exposed to users."""
    code: str
    scene_name: Optional[str] = None
    quality: RenderQuality = RenderQuality.medium
    format: OutputFormat = OutputFormat.mp4
    background_color: Optional[str] = None
    transparent: bool = False
    fps: Optional[int] = None
    resolution: Optional[str] = None
    save_last_frame: bool = False


# ── Output Models ──────────────────────────────────────────────────────

class SceneInfo(BaseModel):
    name: str
    has_construct: bool
    base_classes: list[str]
    line_number: int


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    scenes_found: list[str] = Field(default_factory=list)


class RenderMetadata(BaseModel):
    render_id: str
    parent_render_id: Optional[str] = None
    status: RenderStatus
    scene_name: Optional[str] = None
    quality: Optional[str] = None
    format: Optional[str] = None
    original_prompt: Optional[str] = None
    source_code: Optional[str] = None
    edit_instructions: Optional[str] = None
    s3_url: Optional[str] = None
    s3_object_key: Optional[str] = None
    presigned_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    duration_seconds: Optional[float] = None
    render_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    local_path: Optional[str] = None
    code_hash: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class AnimationResult(BaseModel):
    render_id: str
    status: RenderStatus
    url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    format: Optional[str] = None
    quality: Optional[str] = None
    file_size_bytes: Optional[int] = None
    resolution: Optional[str] = None
    render_time_seconds: Optional[float] = None
    prompt: Optional[str] = None
    source_code: Optional[str] = None
    message: str = ""


# ── RAG Models ────────────────────────────────────────────────────────

class RAGResult(BaseModel):
    """A single result from RAG similarity search."""
    document_id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    similarity_score: float = 0.0


class RAGIndexStatus(BaseModel):
    """Status of RAG collections."""
    available: bool = False
    scenes_count: int = 0
    docs_count: int = 0
    errors_count: int = 0


# ── Agent Models ──────────────────────────────────────────────────────

class MathDomain(str, Enum):
    algebra = "algebra"
    calculus = "calculus"
    geometry = "geometry"
    linear_algebra = "linear_algebra"
    probability = "probability"
    physics = "physics"
    general = "general"


class Complexity(str, Enum):
    simple = "simple"
    moderate = "moderate"
    complex = "complex"


class ConceptAnalysis(BaseModel):
    """Output of the ConceptAnalyzerAgent."""
    domain: MathDomain = MathDomain.general
    complexity: Complexity = Complexity.moderate
    key_concepts: list[str] = Field(default_factory=list)
    visual_elements: list[str] = Field(default_factory=list)
    suggested_duration: int = 15  # seconds


class SceneSegment(BaseModel):
    """A segment of the animation scene plan."""
    name: str
    description: str
    duration: float  # seconds
    mobjects: list[str] = Field(default_factory=list)
    animations: list[str] = Field(default_factory=list)


class ScenePlan(BaseModel):
    """Output of the ScenePlannerAgent."""
    title: str
    segments: list[SceneSegment] = Field(default_factory=list)
    total_duration: float = 0.0
    rag_examples: list[str] = Field(default_factory=list)  # IDs of relevant examples


class CodeReviewResult(BaseModel):
    """Output of the CodeReviewerAgent."""
    approved: bool = False
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    fixed_code: Optional[str] = None


class AgentPipelineResult(BaseModel):
    """Complete result of the multi-agent pipeline."""
    concept_analysis: Optional[ConceptAnalysis] = None
    scene_plan: Optional[ScenePlan] = None
    generated_code: str = ""
    review_result: Optional[CodeReviewResult] = None
    rag_context_used: bool = False
    original_template_code: Optional[str] = None  # For quality comparison


# ── Audio / TTS Models ─────────────────────────────────────────────────

class AudioSegment(BaseModel):
    """A single audio segment from TTS generation."""
    index: int
    text: str
    audio_data: Optional[bytes] = None
    duration_ms: int = 0


class NarrationScript(BaseModel):
    """A narration script split into sentences."""
    sentences: list[str] = Field(default_factory=list)
    total_duration_estimate: float = 0.0
