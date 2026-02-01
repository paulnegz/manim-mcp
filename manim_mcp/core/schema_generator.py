"""Schema-based scene generation: Generate JSON schema, then compile to Python.

This module implements a structured approach to Manim code generation:
1. Generate a JSON schema describing the scene using Gemini's structured output mode
2. Validate the schema against Manim API (mobject types, animation functions, etc.)
3. Compile the validated schema to clean, commented Python code

Benefits over raw code generation:
- Structure is FORCED by schema (no dead code, no variable shadowing)
- Comments are AUTOMATIC (from narration steps)
- Each section is atomic and validated
- Easy to validate before rendering
- Gemini 2.0+ supports structured output natively

Based on IMPROVEMENT_PLAN.md "Structured JSON Output (Codex-style)" approach.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from manim_mcp.prompts import get_schema_generator_system, get_schema_generator_narration

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig
    from manim_mcp.core.rag import ChromaDBService

logger = logging.getLogger(__name__)


# === Data Classes for Schema Structure ===


@dataclass
class ObjectDef:
    """Definition of a Manim mobject to be created.

    Attributes:
        name: Variable name for the object (e.g., "square", "label_a")
        mobject_type: Manim class name (e.g., "Square", "Polygon", "Tex")
        params: Constructor parameters as key-value pairs
        methods: Optional post-construction method calls (e.g., [("set_color", {"color": "BLUE"})])
    """

    name: str
    mobject_type: str
    params: dict[str, Any] = field(default_factory=dict)
    methods: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "mobject_type": self.mobject_type,
            "params": self.params,
            "methods": [{"method": m, "args": a} for m, a in self.methods],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectDef":
        """Create from dictionary (JSON deserialization)."""
        methods = []
        for m in data.get("methods", []):
            if isinstance(m, dict):
                methods.append((m.get("method", ""), m.get("args", {})))
            elif isinstance(m, (list, tuple)) and len(m) >= 2:
                methods.append((m[0], m[1]))
        return cls(
            name=data["name"],
            mobject_type=data["mobject_type"],
            params=data.get("params", {}),
            methods=methods,
        )


@dataclass
class StepDef:
    """Definition of an animation step (corresponds to one narration sentence).

    Attributes:
        narration: The narration text for this step
        new_objects: Objects to create at this step
        animations: Animation calls (e.g., ["ShowCreation(square)", "Write(label)"])
        wait_time: Seconds to wait after animations complete
        transforms: Object transformations (e.g., {"square": "new_square"})
    """

    narration: str
    new_objects: list[ObjectDef] = field(default_factory=list)
    animations: list[str] = field(default_factory=list)
    wait_time: float = 2.0
    transforms: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "narration": self.narration,
            "new_objects": [obj.to_dict() for obj in self.new_objects],
            "animations": self.animations,
            "wait_time": self.wait_time,
            "transforms": self.transforms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepDef":
        """Create from dictionary (JSON deserialization)."""
        new_objects = [ObjectDef.from_dict(obj) for obj in data.get("new_objects", [])]
        return cls(
            narration=data.get("narration", ""),
            new_objects=new_objects,
            animations=data.get("animations", []),
            wait_time=data.get("wait_time", 2.0),
            transforms=data.get("transforms", {}),
        )


@dataclass
class SceneSchema:
    """Complete schema for a Manim scene.

    Attributes:
        class_name: Name of the Scene class
        colors: Color aliases (e.g., {"a_side": "BLUE_A", "b_side": "GREEN_A"})
        constants: Numeric constants (e.g., {"a_len": 2.0, "b_len": 1.5})
        setup_objects: Objects created at the start (before any animations)
        steps: Animation steps (each corresponds to a narration sentence)
        imports: Additional imports needed (beyond manimlib)
    """

    class_name: str
    colors: dict[str, str] = field(default_factory=dict)
    constants: dict[str, Any] = field(default_factory=dict)
    setup_objects: list[ObjectDef] = field(default_factory=list)
    steps: list[StepDef] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "class_name": self.class_name,
            "colors": self.colors,
            "constants": self.constants,
            "setup_objects": [obj.to_dict() for obj in self.setup_objects],
            "steps": [step.to_dict() for step in self.steps],
            "imports": self.imports,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneSchema":
        """Create from dictionary (JSON deserialization)."""
        setup_objects = [ObjectDef.from_dict(obj) for obj in data.get("setup_objects", [])]
        steps = [StepDef.from_dict(step) for step in data.get("steps", [])]
        return cls(
            class_name=data.get("class_name", "GeneratedScene"),
            colors=data.get("colors", {}),
            constants=data.get("constants", {}),
            setup_objects=setup_objects,
            steps=steps,
            imports=data.get("imports", []),
        )


# === JSON Schema for Gemini Structured Output ===


# Pydantic-compatible JSON Schema for Gemini's structured output mode
SCENE_JSON_SCHEMA = {
    "type": "object",
    "required": ["class_name", "steps"],
    "properties": {
        "class_name": {
            "type": "string",
            "description": "PascalCase name for the Scene class (e.g., 'PythagoreanProof')",
        },
        "colors": {
            "type": "object",
            "description": "Color aliases mapping variable names to Manim color constants. Use descriptive names like 'a_side_color' not 'BLUE_A'. Example: {'a_side_color': 'BLUE_A', 'b_side_color': 'GREEN_A'}",
            "additionalProperties": {"type": "string"},
        },
        "constants": {
            "type": "object",
            "description": "Numeric constants for the scene. Example: {'a_length': 2.0, 'b_length': 1.5, 'scale_factor': 0.8}",
            "additionalProperties": {},
        },
        "imports": {
            "type": "array",
            "description": "Additional imports beyond 'from manimlib import *'. Example: ['import numpy as np', 'from math import sqrt']",
            "items": {"type": "string"},
        },
        "setup_objects": {
            "type": "array",
            "description": "Objects created at the start, before any animations. These appear immediately without animation.",
            "items": {
                "type": "object",
                "required": ["name", "mobject_type"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Variable name (snake_case, e.g., 'main_square')",
                    },
                    "mobject_type": {
                        "type": "string",
                        "description": "Manim class name (e.g., 'Square', 'Polygon', 'Tex', 'NumberPlane')",
                    },
                    "params": {
                        "type": "object",
                        "description": "Constructor parameters. Use constant/color variable names, not raw values. Example: {'side_length': 'a_length', 'color': 'a_side_color'}",
                        "additionalProperties": {},
                    },
                    "methods": {
                        "type": "array",
                        "description": "Post-construction method calls. Example: [{'method': 'shift', 'args': {'direction': 'LEFT * 2'}}]",
                        "items": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string"},
                                "args": {"type": "object", "additionalProperties": {}},
                            },
                        },
                    },
                },
            },
        },
        "steps": {
            "type": "array",
            "description": "Animation steps, each corresponding to one narration sentence. Steps execute sequentially.",
            "items": {
                "type": "object",
                "required": ["narration"],
                "properties": {
                    "narration": {
                        "type": "string",
                        "description": "The narration text for this step (from the script)",
                    },
                    "new_objects": {
                        "type": "array",
                        "description": "New objects to create at this step",
                        "items": {
                            "type": "object",
                            "required": ["name", "mobject_type"],
                            "properties": {
                                "name": {"type": "string"},
                                "mobject_type": {"type": "string"},
                                "params": {
                                    "type": "object",
                                    "additionalProperties": {},
                                },
                                "methods": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "method": {"type": "string"},
                                            "args": {
                                                "type": "object",
                                                "additionalProperties": {},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "animations": {
                        "type": "array",
                        "description": "Animation calls as strings. Example: ['ShowCreation(square)', 'Write(label)', 'FadeIn(area, run_time=2)']",
                        "items": {"type": "string"},
                    },
                    "wait_time": {
                        "type": "number",
                        "description": "Seconds to wait after animations (default: 2.0)",
                        "default": 2.0,
                    },
                    "transforms": {
                        "type": "object",
                        "description": "Transform animations: maps source object to target. Example: {'old_square': 'new_square'} generates Transform(old_square, new_square)",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
        },
    },
}


# === Known Manim Types for Validation ===


# Common Manim mobject types (manimgl)
KNOWN_MOBJECT_TYPES = {
    # Basic shapes
    "Circle", "Square", "Rectangle", "Triangle", "Polygon", "RegularPolygon",
    "Line", "Arrow", "DoubleArrow", "Vector", "DashedLine", "Arc", "ArcBetweenPoints",
    "Dot", "SmallDot", "Ellipse", "Annulus", "Sector", "AnnularSector",
    # 3D shapes
    "Sphere", "Cube", "Cylinder", "Cone", "Prism", "Torus",
    # Text and math
    "Text", "Tex", "TexText", "MathTex", "Title", "BulletedList",
    "Paragraph", "Code",
    # Groups
    "VGroup", "Group", "VMobject", "Mobject",
    # Graphs and axes
    "Axes", "NumberPlane", "CoordinateSystem", "ThreeDAxes",
    "NumberLine", "BarChart", "Graph",
    # Images and SVG
    "ImageMobject", "SVGMobject",
    # Special
    "SurroundingRectangle", "BackgroundRectangle", "Cross", "Checkmark",
    "Brace", "BraceLabel", "BraceText",
    "DecimalNumber", "Integer", "Variable",
    # Function graphs
    "ParametricCurve", "FunctionGraph",
}

# Common Manim animation types (manimgl)
KNOWN_ANIMATION_TYPES = {
    # Creation
    "ShowCreation", "Create", "DrawBorderThenFill", "Write", "AddTextLetterByLetter",
    "ShowIncreasingSubsets", "ShowSubmobjectsOneByOne",
    # Fading
    "FadeIn", "FadeOut", "FadeInFromPoint", "FadeOutToPoint",
    "FadeInFrom", "FadeOutAndShift", "FadeInFromDown", "FadeInFromLarge",
    # Transform
    "Transform", "ReplacementTransform", "TransformFromCopy", "ClockwiseTransform",
    "CounterclockwiseTransform", "MoveToTarget", "ApplyMethod",
    "TransformMatchingShapes", "TransformMatchingTex",
    # Movement
    "MoveAlongPath", "Rotating", "Rotate", "SpinInFromNothing",
    "GrowFromCenter", "GrowFromPoint", "GrowFromEdge", "GrowArrow",
    "SpinInFromNothing",
    # Indication
    "Indicate", "FocusOn", "Flash", "CircleIndicate", "ShowPassingFlash",
    "ShowCreationThenDestruction", "ShowCreationThenFadeOut",
    "AnimationOnSurroundingRectangle", "ShowPassingFlashAround",
    "Wiggle", "TurnInsideOut", "FlashAround",
    # Updates
    "UpdateFromFunc", "UpdateFromAlphaFunc", "MaintainPositionRelativeTo",
    # Uncreation
    "Uncreate", "ShrinkToCenter", "FadeOutAndShiftDown",
    # Special
    "Succession", "AnimationGroup", "LaggedStart", "LaggedStartMap",
    "Wait", "Homotopy", "ComplexHomotopy", "PhaseFlow", "MoveAlongPath",
}

# Manim color constants
KNOWN_COLORS = {
    # Basic colors
    "WHITE", "BLACK", "GREY", "GRAY", "DARK_GREY", "DARK_GRAY",
    "LIGHT_GREY", "LIGHT_GRAY", "GREY_BROWN", "GRAY_BROWN",
    # Primary
    "RED", "GREEN", "BLUE", "YELLOW", "ORANGE", "PINK", "PURPLE",
    "TEAL", "MAROON", "GOLD",
    # Shades (A, B, C, D, E variants)
    "RED_A", "RED_B", "RED_C", "RED_D", "RED_E",
    "GREEN_A", "GREEN_B", "GREEN_C", "GREEN_D", "GREEN_E",
    "BLUE_A", "BLUE_B", "BLUE_C", "BLUE_D", "BLUE_E",
    "YELLOW_A", "YELLOW_B", "YELLOW_C", "YELLOW_D", "YELLOW_E",
    "TEAL_A", "TEAL_B", "TEAL_C", "TEAL_D", "TEAL_E",
    "PURPLE_A", "PURPLE_B", "PURPLE_C", "PURPLE_D", "PURPLE_E",
    "GREY_A", "GREY_B", "GREY_C", "GREY_D", "GREY_E",
    "GRAY_A", "GRAY_B", "GRAY_C", "GRAY_D", "GRAY_E",
    # Special 3b1b colors
    "BLUE_E", "BLUE_D", "BLUE_C", "BLUE_B", "BLUE_A",
    "TEAL_E", "TEAL_D", "TEAL_C", "TEAL_B", "TEAL_A",
    "GREEN_E", "GREEN_D", "GREEN_C", "GREEN_B", "GREEN_A",
    "YELLOW_E", "YELLOW_D", "YELLOW_C", "YELLOW_B", "YELLOW_A",
    "GOLD_E", "GOLD_D", "GOLD_C", "GOLD_B", "GOLD_A",
    "RED_E", "RED_D", "RED_C", "RED_B", "RED_A",
    "MAROON_E", "MAROON_D", "MAROON_C", "MAROON_B", "MAROON_A",
    "PURPLE_E", "PURPLE_D", "PURPLE_C", "PURPLE_B", "PURPLE_A",
    "PINK",
    # Hex colors (allow these patterns)
}


# === Validation Errors ===


class SchemaValidationError(Enum):
    """Types of schema validation errors."""

    INVALID_MOBJECT_TYPE = "invalid_mobject_type"
    INVALID_ANIMATION_TYPE = "invalid_animation_type"
    INVALID_COLOR = "invalid_color"
    UNDEFINED_VARIABLE = "undefined_variable"
    DUPLICATE_VARIABLE = "duplicate_variable"
    INVALID_CLASS_NAME = "invalid_class_name"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_PARAM_TYPE = "invalid_param_type"


@dataclass
class ValidationIssue:
    """A validation issue found in the schema."""

    error_type: SchemaValidationError
    message: str
    location: str  # e.g., "steps[2].new_objects[0].mobject_type"
    suggestion: str | None = None


# === Schema Generator Class ===


class SchemaGenerator:
    """Generates structured scene schemas using Gemini's JSON mode.

    This class provides a structured approach to Manim code generation:
    1. generate_schema() - Use Gemini with JSON mode to create a scene schema
    2. validate_schema() - Validate the schema against Manim API
    3. compile_schema_to_python() - Convert validated schema to Python code
    """

    def __init__(
        self,
        config: "ManimMCPConfig",
        rag: "ChromaDBService | None" = None,
    ) -> None:
        """Initialize the schema generator.

        Args:
            config: Application configuration (contains Gemini API key, model, etc.)
            rag: Optional RAG service for enhanced validation and examples
        """
        self.config = config
        self.rag = rag

        # Initialize Gemini client
        from google import genai

        self._genai = genai
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model_name = config.gemini_model

        logger.info(
            "[SCHEMA-GEN] Initialized with model=%s, rag=%s",
            self.model_name,
            "enabled" if rag and rag.available else "disabled",
        )

    async def generate_schema(
        self,
        prompt: str,
        narration: list[str] | None = None,
    ) -> dict:
        """Generate a structured scene schema using Gemini's JSON mode.

        Args:
            prompt: Text description of the animation to create
            narration: Optional list of narration sentences (for step synchronization)

        Returns:
            Dictionary representing the scene schema (SceneSchema.to_dict() format)
        """
        # Build the system prompt
        system_prompt = self._build_system_prompt()

        # Build the user prompt with narration if provided
        user_prompt = self._build_user_prompt(prompt, narration)

        # Call Gemini with structured output
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=self._genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=SCENE_JSON_SCHEMA,
                    temperature=0.2,  # Lower temperature for more consistent schema output
                ),
            )

            # Parse the JSON response
            schema_text = response.text.strip()
            logger.debug("[SCHEMA-GEN] Raw response: %s", schema_text[:500])

            schema_dict = json.loads(schema_text)
            logger.info(
                "[SCHEMA-GEN] Generated schema with %d steps, %d colors, %d constants",
                len(schema_dict.get("steps", [])),
                len(schema_dict.get("colors", {})),
                len(schema_dict.get("constants", {})),
            )

            return schema_dict

        except json.JSONDecodeError as e:
            logger.error("[SCHEMA-GEN] Failed to parse JSON response: %s", e)
            raise ValueError(f"Gemini returned invalid JSON: {e}")
        except Exception as e:
            logger.error("[SCHEMA-GEN] Generation failed: %s", e)
            raise

    def validate_schema(self, schema: dict) -> list[ValidationIssue]:
        """Validate a schema against Manim API.

        Checks:
        - Mobject types are valid Manim classes
        - Animation types are valid Manim animations
        - Color references are valid Manim colors
        - Variable names are properly defined before use
        - Class name is valid PascalCase

        Args:
            schema: Scene schema dictionary

        Returns:
            List of validation issues (empty if valid)
        """
        issues: list[ValidationIssue] = []

        # Track defined variables for reference checking
        defined_vars: set[str] = set()

        # 1. Validate class name
        class_name = schema.get("class_name", "")
        if not class_name or not class_name[0].isupper() or not class_name.isidentifier():
            issues.append(
                ValidationIssue(
                    error_type=SchemaValidationError.INVALID_CLASS_NAME,
                    message=f"Invalid class name: '{class_name}'",
                    location="class_name",
                    suggestion="Use PascalCase like 'PythagoreanProof'",
                )
            )

        # 2. Track color and constant definitions
        colors = schema.get("colors", {})
        constants = schema.get("constants", {})
        defined_vars.update(colors.keys())
        defined_vars.update(constants.keys())

        # 3. Validate color values
        for color_name, color_value in colors.items():
            if not self._is_valid_color(color_value):
                issues.append(
                    ValidationIssue(
                        error_type=SchemaValidationError.INVALID_COLOR,
                        message=f"Unknown color: '{color_value}'",
                        location=f"colors.{color_name}",
                        suggestion=f"Use a known Manim color like BLUE_A, RED, etc.",
                    )
                )

        # 4. Validate setup_objects
        for i, obj in enumerate(schema.get("setup_objects", [])):
            obj_issues = self._validate_object_def(obj, f"setup_objects[{i}]", defined_vars)
            issues.extend(obj_issues)
            # Add object name to defined vars
            if obj.get("name"):
                defined_vars.add(obj["name"])

        # 5. Validate steps
        for step_idx, step in enumerate(schema.get("steps", [])):
            step_location = f"steps[{step_idx}]"

            # Validate new_objects in this step
            for obj_idx, obj in enumerate(step.get("new_objects", [])):
                obj_issues = self._validate_object_def(
                    obj, f"{step_location}.new_objects[{obj_idx}]", defined_vars
                )
                issues.extend(obj_issues)
                if obj.get("name"):
                    defined_vars.add(obj["name"])

            # Validate animations
            for anim_idx, anim in enumerate(step.get("animations", [])):
                anim_issues = self._validate_animation(
                    anim, f"{step_location}.animations[{anim_idx}]", defined_vars
                )
                issues.extend(anim_issues)

            # Validate transforms
            for source, target in step.get("transforms", {}).items():
                if source not in defined_vars:
                    issues.append(
                        ValidationIssue(
                            error_type=SchemaValidationError.UNDEFINED_VARIABLE,
                            message=f"Transform source '{source}' is not defined",
                            location=f"{step_location}.transforms.{source}",
                        )
                    )
                # Target should be a newly defined object in this step's new_objects
                step_new_names = {obj.get("name") for obj in step.get("new_objects", [])}
                if target not in step_new_names and target not in defined_vars:
                    issues.append(
                        ValidationIssue(
                            error_type=SchemaValidationError.UNDEFINED_VARIABLE,
                            message=f"Transform target '{target}' is not defined",
                            location=f"{step_location}.transforms.{source}",
                        )
                    )

        logger.info(
            "[SCHEMA-GEN] Validation complete: %d issues found",
            len(issues),
        )
        return issues

    def _validate_object_def(
        self,
        obj: dict,
        location: str,
        defined_vars: set[str],
    ) -> list[ValidationIssue]:
        """Validate an object definition."""
        issues: list[ValidationIssue] = []

        # Check mobject type
        mobject_type = obj.get("mobject_type", "")
        if mobject_type and mobject_type not in KNOWN_MOBJECT_TYPES:
            # Check if it might be a typo
            suggestion = self._find_similar(mobject_type, KNOWN_MOBJECT_TYPES)
            issues.append(
                ValidationIssue(
                    error_type=SchemaValidationError.INVALID_MOBJECT_TYPE,
                    message=f"Unknown mobject type: '{mobject_type}'",
                    location=f"{location}.mobject_type",
                    suggestion=f"Did you mean '{suggestion}'?" if suggestion else None,
                )
            )

        # Check for duplicate variable names
        obj_name = obj.get("name", "")
        if obj_name in defined_vars:
            issues.append(
                ValidationIssue(
                    error_type=SchemaValidationError.DUPLICATE_VARIABLE,
                    message=f"Duplicate variable name: '{obj_name}'",
                    location=f"{location}.name",
                    suggestion="Use unique variable names for each object",
                )
            )

        # Check param references (if they reference defined vars/colors)
        for param_name, param_value in obj.get("params", {}).items():
            if isinstance(param_value, str):
                # Check if it's a variable reference (not a literal)
                if self._looks_like_variable(param_value):
                    if param_value not in defined_vars and param_value not in KNOWN_COLORS:
                        # Could be a positional expression like "LEFT * 2" - skip these
                        if not any(op in param_value for op in ["*", "+", "-", "/"]):
                            issues.append(
                                ValidationIssue(
                                    error_type=SchemaValidationError.UNDEFINED_VARIABLE,
                                    message=f"Undefined variable in params: '{param_value}'",
                                    location=f"{location}.params.{param_name}",
                                )
                            )

        return issues

    def _validate_animation(
        self,
        anim_str: str,
        location: str,
        defined_vars: set[str],
    ) -> list[ValidationIssue]:
        """Validate an animation string."""
        issues: list[ValidationIssue] = []

        # Extract animation type from string like "ShowCreation(square)"
        match = re.match(r"(\w+)\s*\(", anim_str)
        if not match:
            return issues  # Can't parse, skip validation

        anim_type = match.group(1)
        if anim_type not in KNOWN_ANIMATION_TYPES:
            suggestion = self._find_similar(anim_type, KNOWN_ANIMATION_TYPES)
            issues.append(
                ValidationIssue(
                    error_type=SchemaValidationError.INVALID_ANIMATION_TYPE,
                    message=f"Unknown animation type: '{anim_type}'",
                    location=location,
                    suggestion=f"Did you mean '{suggestion}'?" if suggestion else None,
                )
            )

        # Extract object references from the animation
        # Simple extraction of first argument (the target object)
        inner_match = re.match(r"\w+\s*\(\s*(\w+)", anim_str)
        if inner_match:
            target = inner_match.group(1)
            # Skip if it looks like a class name (PascalCase) - might be a constructor
            if not target[0].isupper() and target not in defined_vars:
                issues.append(
                    ValidationIssue(
                        error_type=SchemaValidationError.UNDEFINED_VARIABLE,
                        message=f"Animation references undefined object: '{target}'",
                        location=location,
                    )
                )

        return issues

    def _is_valid_color(self, color: str) -> bool:
        """Check if a color string is a valid Manim color."""
        # Known colors
        if color in KNOWN_COLORS:
            return True
        # Hex color pattern
        if re.match(r"^#[0-9A-Fa-f]{6}$", color):
            return True
        # RGB function
        if re.match(r"^rgb\s*\(", color, re.IGNORECASE):
            return True
        return False

    def _looks_like_variable(self, s: str) -> bool:
        """Check if a string looks like a variable reference."""
        # All caps = likely a constant (BLUE_A, LEFT, etc.)
        if s.isupper() or "_" in s and s.replace("_", "").isupper():
            return True
        # snake_case = likely a variable
        if re.match(r"^[a-z][a-z0-9_]*$", s):
            return True
        return False

    def _find_similar(self, name: str, known: set[str]) -> str | None:
        """Find a similar name in the known set (simple Levenshtein)."""
        name_lower = name.lower()
        for known_name in known:
            # Exact case-insensitive match
            if known_name.lower() == name_lower:
                return known_name
            # Prefix match
            if known_name.lower().startswith(name_lower[:3]):
                return known_name
        return None

    def compile_schema_to_python(self, schema: dict) -> str:
        """Compile a JSON schema to valid Manim Python code.

        Generates clean, commented Python code following 3blue1brown conventions.

        Args:
            schema: Scene schema dictionary

        Returns:
            Complete Python code string
        """
        lines: list[str] = []

        # 1. Imports
        lines.append("from manimlib import *")
        for imp in schema.get("imports", []):
            lines.append(imp)
        lines.append("")
        lines.append("")

        # 2. Class definition
        class_name = schema.get("class_name", "GeneratedScene")
        lines.append(f"class {class_name}(Scene):")
        lines.append("    def construct(self):")

        # 3. Colors section
        colors = schema.get("colors", {})
        if colors:
            lines.append("        # === Colors ===")
            for name, value in colors.items():
                lines.append(f"        {name} = {value}")
            lines.append("")

        # 4. Constants section
        constants = schema.get("constants", {})
        if constants:
            lines.append("        # === Constants ===")
            for name, value in constants.items():
                lines.append(f"        {name} = {self._format_value(value)}")
            lines.append("")

        # 5. Setup objects
        setup_objects = schema.get("setup_objects", [])
        if setup_objects:
            lines.append("        # === Setup ===")
            for obj in setup_objects:
                lines.extend(self._compile_object(obj, indent=8))
            lines.append("")

        # 6. Steps
        for i, step in enumerate(schema.get("steps", []), 1):
            narration = step.get("narration", "")
            lines.append(f"        # === Step {i}: {narration} ===")

            # Create new objects
            for obj in step.get("new_objects", []):
                lines.extend(self._compile_object(obj, indent=8))

            # Transforms
            for source, target in step.get("transforms", {}).items():
                lines.append(f"        self.play(Transform({source}, {target}))")

            # Animations
            animations = step.get("animations", [])
            if animations:
                if len(animations) == 1:
                    lines.append(f"        self.play({animations[0]})")
                else:
                    # Multiple animations - group them
                    anim_str = ", ".join(animations)
                    if len(anim_str) > 60:
                        # Multi-line format
                        lines.append("        self.play(")
                        for anim in animations:
                            lines.append(f"            {anim},")
                        lines.append("        )")
                    else:
                        lines.append(f"        self.play({anim_str})")

            # Wait
            wait_time = step.get("wait_time", 2.0)
            if wait_time > 0:
                lines.append(f"        self.wait({wait_time})")

            lines.append("")

        # Remove trailing empty lines and ensure file ends with newline
        while lines and lines[-1] == "":
            lines.pop()
        lines.append("")

        return "\n".join(lines)

    def _compile_object(self, obj: dict, indent: int = 8) -> list[str]:
        """Compile an object definition to Python lines."""
        lines: list[str] = []
        pad = " " * indent

        name = obj.get("name", "obj")
        mobject_type = obj.get("mobject_type", "VMobject")
        params = obj.get("params", {})
        methods = obj.get("methods", [])

        # Build constructor call
        if params:
            param_strs = [
                f"{k}={self._format_value(v)}" for k, v in params.items()
            ]
            param_str = ", ".join(param_strs)
            if len(param_str) > 50:
                # Multi-line format
                lines.append(f"{pad}{name} = {mobject_type}(")
                for k, v in params.items():
                    lines.append(f"{pad}    {k}={self._format_value(v)},")
                lines.append(f"{pad})")
            else:
                lines.append(f"{pad}{name} = {mobject_type}({param_str})")
        else:
            lines.append(f"{pad}{name} = {mobject_type}()")

        # Method calls
        for method_item in methods:
            if isinstance(method_item, dict):
                method = method_item.get("method", "")
                args = method_item.get("args", {})
            elif isinstance(method_item, (list, tuple)) and len(method_item) >= 2:
                method, args = method_item[0], method_item[1]
            else:
                continue

            if args:
                if isinstance(args, dict):
                    arg_strs = [f"{k}={self._format_value(v)}" for k, v in args.items()]
                    arg_str = ", ".join(arg_strs)
                else:
                    arg_str = self._format_value(args)
                lines.append(f"{pad}{name}.{method}({arg_str})")
            else:
                lines.append(f"{pad}{name}.{method}()")

        return lines

    def _format_value(self, value: Any) -> str:
        """Format a value for Python code output."""
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, str):
            # Check if it's a variable/constant reference (not a string literal)
            # Variables: snake_case, SCREAMING_CASE, or expressions
            if re.match(r"^[A-Z][A-Z0-9_]*$", value):  # CONSTANT
                return value
            if re.match(r"^[a-z_][a-z0-9_]*$", value):  # variable
                return value
            if any(op in value for op in ["*", "+", "-", "/", "(", ")"]):  # expression
                return value
            if value in KNOWN_COLORS:
                return value
            # Otherwise, it's a string literal
            return repr(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            formatted = [self._format_value(v) for v in value]
            return f"[{', '.join(formatted)}]"
        if isinstance(value, dict):
            items = [f"{repr(k)}: {self._format_value(v)}" for k, v in value.items()]
            return "{" + ", ".join(items) + "}"
        return repr(value)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for schema generation."""
        return get_schema_generator_system()

    def _build_user_prompt(
        self,
        prompt: str,
        narration: list[str] | None = None,
    ) -> str:
        """Build the user prompt with optional narration."""
        user_prompt = f"Create a Manim animation for:\n{prompt}"

        if narration:
            script_text = "\n".join(f"{i+1}. {sentence}" for i, sentence in enumerate(narration))
            user_prompt += get_schema_generator_narration(script_text, len(narration))

        return user_prompt


# === Convenience Functions ===


async def generate_scene_from_prompt(
    prompt: str,
    config: "ManimMCPConfig",
    narration: list[str] | None = None,
    rag: "ChromaDBService | None" = None,
    validate: bool = True,
) -> tuple[str, dict, list[ValidationIssue]]:
    """Generate Manim code from a prompt using the schema-based approach.

    This is a convenience function that combines schema generation, validation,
    and compilation in one call.

    Args:
        prompt: Text description of the animation
        config: Application configuration
        narration: Optional narration script (list of sentences)
        rag: Optional RAG service for enhanced context
        validate: Whether to validate the schema (default True)

    Returns:
        Tuple of (python_code, schema_dict, validation_issues)
    """
    generator = SchemaGenerator(config, rag)

    # Generate schema
    schema = await generator.generate_schema(prompt, narration)

    # Validate
    issues = []
    if validate:
        issues = generator.validate_schema(schema)

    # Compile to Python
    code = generator.compile_schema_to_python(schema)

    return code, schema, issues


def compile_schema(schema: dict) -> str:
    """Compile a schema dictionary to Python code.

    Standalone function for use without async context.

    Args:
        schema: Scene schema dictionary

    Returns:
        Python code string
    """
    # Create a minimal generator just for compilation
    # (doesn't need config/API for compilation only)

    class MinimalGenerator:
        def compile_schema_to_python(self, schema: dict) -> str:
            return SchemaGenerator.__dict__["compile_schema_to_python"](self, schema)

        def _compile_object(self, obj: dict, indent: int = 8) -> list[str]:
            return SchemaGenerator.__dict__["_compile_object"](self, obj, indent)

        def _format_value(self, value: Any) -> str:
            return SchemaGenerator.__dict__["_format_value"](self, value)

    return MinimalGenerator().compile_schema_to_python(schema)


def validate_scene_schema(schema: dict) -> list[ValidationIssue]:
    """Validate a schema dictionary.

    Standalone function for use without async context.

    Args:
        schema: Scene schema dictionary

    Returns:
        List of validation issues (empty if valid)
    """

    class MinimalGenerator:
        rag = None

        def validate_schema(self, schema: dict) -> list[ValidationIssue]:
            return SchemaGenerator.__dict__["validate_schema"](self, schema)

        def _validate_object_def(self, obj: dict, location: str, defined_vars: set[str]) -> list[ValidationIssue]:
            return SchemaGenerator.__dict__["_validate_object_def"](self, obj, location, defined_vars)

        def _validate_animation(self, anim_str: str, location: str, defined_vars: set[str]) -> list[ValidationIssue]:
            return SchemaGenerator.__dict__["_validate_animation"](self, anim_str, location, defined_vars)

        def _is_valid_color(self, color: str) -> bool:
            return SchemaGenerator.__dict__["_is_valid_color"](self, color)

        def _looks_like_variable(self, s: str) -> bool:
            return SchemaGenerator.__dict__["_looks_like_variable"](self, s)

        def _find_similar(self, name: str, known: set[str]) -> str | None:
            return SchemaGenerator.__dict__["_find_similar"](self, name, known)

    return MinimalGenerator().validate_schema(schema)
