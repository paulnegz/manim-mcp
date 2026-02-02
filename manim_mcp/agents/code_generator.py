"""CodeGeneratorAgent: Generates Manim code from scene plans with RAG context."""

from __future__ import annotations

import ast
import asyncio
import logging
import re
from typing import TYPE_CHECKING

from manim_mcp.agents.base import BaseAgent
from manim_mcp.models import ConceptAnalysis, ScenePlan
from manim_mcp.prompts import get_code_generator_system

if TYPE_CHECKING:
    from manim_mcp.core.api_graph import ManimAPIGraph

logger = logging.getLogger(__name__)

# Common Manim method/class keywords to extract from prompts
# Maps common user-facing terms to their actual manimgl API names
MANIM_KEYWORD_MAP = {
    # Animations
    "transform": ["Transform", "ReplacementTransform", "TransformMatchingTex"],
    "fade": ["FadeIn", "FadeOut", "FadeTransform"],
    "rotate": ["Rotate", "rotate", "Rotating"],
    "move": ["shift", "move_to", "next_to"],
    "write": ["Write", "ShowCreation"],
    "create": ["ShowCreation", "Create"],
    "draw": ["ShowCreation", "DrawBorderThenFill"],
    "animate": ["animate", "Animation"],
    "morph": ["Transform", "TransformMatchingTex", "ReplacementTransform"],
    "grow": ["GrowFromCenter", "GrowArrow", "GrowFromPoint"],
    "indicate": ["Indicate", "ShowPassingFlash", "FlashAround"],
    "highlight": ["Indicate", "FlashAround", "SurroundingRectangle"],
    # Mobjects
    "circle": ["Circle", "Dot", "Annulus"],
    "square": ["Square", "Rectangle"],
    "line": ["Line", "Arrow", "DashedLine", "Vector"],
    "arrow": ["Arrow", "Vector", "GrowArrow"],
    "dot": ["Dot", "SmallDot"],
    "text": ["Tex", "TexText", "Text"],
    "equation": ["Tex", "MathTex"],
    "formula": ["Tex", "MathTex"],
    "graph": ["Axes", "get_graph", "NumberPlane"],
    "plot": ["Axes", "get_graph", "plot"],
    "axes": ["Axes", "NumberPlane", "CoordinateSystem"],
    "function": ["get_graph", "FunctionGraph", "ParametricCurve"],
    "curve": ["ParametricCurve", "get_graph", "CubicBezier"],
    "vector": ["Vector", "Arrow", "NumberPlane"],
    "matrix": ["Matrix", "IntegerMatrix", "DecimalMatrix"],
    "polygon": ["Polygon", "RegularPolygon", "Triangle"],
    "triangle": ["Triangle", "Polygon"],
    "group": ["VGroup", "Group"],
    # Math concepts
    "integral": ["get_area", "get_riemann_rectangles", "Integral"],
    "riemann": ["get_riemann_rectangles", "Axes"],
    "derivative": ["TangentLine", "get_secant_slope_group"],
    "tangent": ["TangentLine", "get_tangent_line"],
    "secant": ["get_secant_slope_group", "Line"],
    "area": ["get_area", "get_area_under_graph", "Polygon"],
    "limit": ["ValueTracker", "always_redraw"],
    "series": ["VGroup", "LaggedStartMap"],
    "sum": ["VGroup", "Tex"],
    # Advanced
    "tracker": ["ValueTracker", "always_redraw"],
    "updater": ["add_updater", "always_redraw", "become"],
    "trace": ["TracedPath", "add_updater"],
    "path": ["TracedPath", "ParametricCurve", "VMobject"],
    "3d": ["ThreeDScene", "Surface", "ThreeDAxes"],
    "surface": ["Surface", "ParametricSurface"],
    "camera": ["camera", "set_camera_orientation", "move_camera"],
}

# Threshold for "very high quality" match - use code directly
# Lower threshold = more direct use of verified 3b1b code (fewer LLM generation errors)
DIRECT_USE_THRESHOLD = 0.08  # Keep low - 3b1b code is correct, LLM generation has errors

# Minimum code length for direct use (short snippets need LLM enhancement)
DIRECT_USE_MIN_CHARS = 800

# Known 3blue1brown color naming patterns
KNOWN_3B1B_COLORS = {
    # Standard color aliases
    "A_COLOR", "B_COLOR", "C_COLOR", "HYPOTENUSE_COLOR",
    "SIDE_COLORS", "LINE_COLOR", "AREA_COLOR", "GRAPH_COLOR",
    "FUNCTION_COLOR", "DERIVATIVE_COLOR", "INTEGRAL_COLOR",
    "X_COLOR", "Y_COLOR", "Z_COLOR", "T_COLOR",
    # Common semantic colors
    "HIGHLIGHT_COLOR", "FOCUS_COLOR", "LABEL_COLOR",
}

# Manim positioning methods to extract
POSITIONING_METHODS = {
    "next_to", "align_to", "move_to", "shift", "to_edge", "to_corner",
    "center", "arrange", "arrange_in_grid", "set_x", "set_y", "set_z",
}

# Animation methods to track
ANIMATION_METHODS = {
    "play", "wait", "add", "remove", "FadeIn", "FadeOut", "Write",
    "ShowCreation", "Create", "Transform", "ReplacementTransform",
    "TransformMatchingTex", "Indicate", "GrowFromCenter", "DrawBorderThenFill",
    "LaggedStartMap", "Succession", "AnimationGroup",
}

# Common mobject classes
MOBJECT_CLASSES = {
    "Circle", "Square", "Rectangle", "Triangle", "Polygon", "Line", "Arrow",
    "Dot", "Vector", "Axes", "NumberPlane", "Graph", "ParametricCurve",
    "Tex", "MathTex", "Text", "TexText", "VGroup", "Group", "VMobject",
    "Surface", "ThreeDAxes", "ValueTracker", "DecimalNumber",
}


class CodeGeneratorAgent(BaseAgent):
    """Generates Manim code from scene plans, using RAG for few-shot examples."""

    name = "code_generator"
    _api_graph: "ManimAPIGraph | None" = None

    async def _get_api_graph(self) -> "ManimAPIGraph | None":
        """Get or initialize the API relationship graph.

        Returns:
            ManimAPIGraph instance or None if unavailable
        """
        if self._api_graph is not None:
            return self._api_graph

        try:
            from manim_mcp.core.api_graph import get_api_graph
            self._api_graph = await get_api_graph(self.rag if self.rag_available else None)
            if self._api_graph and self._api_graph.available:
                stats = self._api_graph.get_graph_stats()
                logger.info(
                    "[API-GRAPH] Initialized with %d nodes, %d edges",
                    stats.get("nodes", 0),
                    stats.get("edges", 0),
                )
            return self._api_graph
        except ImportError:
            logger.debug("[API-GRAPH] networkx not installed - graph unavailable")
            return None
        except Exception as e:
            logger.warning("[API-GRAPH] Failed to initialize: %s", e)
            return None

    async def _get_method_sequence_suggestions(
        self,
        analysis: ConceptAnalysis,
        plan: ScenePlan,
    ) -> list[dict]:
        """Get method sequence suggestions from the API graph.

        Args:
            analysis: Concept analysis with visual elements
            plan: Scene plan with segments

        Returns:
            List of suggested method sequences
        """
        api_graph = await self._get_api_graph()
        if not api_graph or not api_graph.available:
            return []

        suggestions = []

        # Get mobjects from analysis
        visual_elements = analysis.visual_elements or []

        # For each visual element, suggest animation sequences
        for element in visual_elements[:5]:  # Limit to first 5 elements
            # Normalize element name to match graph nodes
            element_normalized = element.strip().replace(" ", "")

            # Check if element exists as a node
            node_info = api_graph.get_method_info(element_normalized)
            if not node_info:
                # Try common variations
                for variant in [element_normalized, element_normalized.title(), element_normalized.upper()]:
                    node_info = api_graph.get_method_info(variant)
                    if node_info:
                        element_normalized = variant
                        break

            if node_info:
                # Get what commonly follows this mobject/method
                next_methods = api_graph.get_valid_next_methods(
                    element_normalized,
                    limit=5,
                )
                if next_methods:
                    suggestions.append({
                        "element": element_normalized,
                        "category": node_info.category,
                        "next_methods": [
                            {"name": m[0], "weight": m[1], "relation": m[2]}
                            for m in next_methods
                        ],
                    })

                # Get commonly paired methods
                paired = api_graph.get_commonly_paired(element_normalized, limit=3)
                if paired:
                    suggestions.append({
                        "element": element_normalized,
                        "paired_with": [{"name": p[0], "weight": p[1]} for p in paired],
                    })

        # For each segment, suggest animation chains
        for seg in plan.segments[:3]:  # First 3 segments
            if seg.animations:
                for anim in seg.animations[:2]:
                    anim_normalized = anim.strip().replace(" ", "")
                    next_methods = api_graph.get_valid_next_methods(anim_normalized, limit=3)
                    if next_methods:
                        suggestions.append({
                            "segment": seg.name,
                            "animation": anim_normalized,
                            "can_follow_with": [m[0] for m in next_methods],
                        })

        if suggestions:
            logger.info("[API-GRAPH] Generated %d method sequence suggestions", len(suggestions))

        return suggestions

    def _extract_patterns_from_examples(self, examples: list[dict]) -> dict:
        """Extract reusable patterns from RAG examples, not just raw code.

        Uses AST parsing to extract:
        1. Color naming patterns - How colors are aliased (e.g., A_COLOR = BLUE_A)
        2. Object creation patterns - Common mobject creation idioms
        3. Animation sequences - Typical animation chains
        4. Positioning patterns - Layout conventions (next_to, align_to, etc.)
        5. Anti-patterns - Things to avoid (from error examples)

        Args:
            examples: List of RAG example dicts with 'content' field containing code

        Returns:
            Dict with extracted patterns organized by category
        """
        patterns = {
            "color_naming": [],
            "object_creation": [],
            "animation_sequences": [],
            "positioning": [],
            "anti_patterns": [],
        }

        for example in examples:
            content = example.get("content", "")
            metadata = example.get("metadata", {})
            is_error = metadata.get("category") == "error" or "ERROR:" in content

            if is_error:
                # Extract anti-patterns from error examples
                anti_patterns = self._extract_anti_patterns(content)
                patterns["anti_patterns"].extend(anti_patterns)
            else:
                # Parse code and extract patterns
                try:
                    tree = ast.parse(content)
                    patterns["color_naming"].extend(
                        self._extract_color_patterns(tree, content)
                    )
                    patterns["object_creation"].extend(
                        self._extract_object_creation_patterns(tree, content)
                    )
                    patterns["animation_sequences"].extend(
                        self._extract_animation_sequences(tree, content)
                    )
                    patterns["positioning"].extend(
                        self._extract_positioning_patterns(tree, content)
                    )
                except SyntaxError as e:
                    logger.debug("[PATTERN] Failed to parse example: %s", e)
                    # Try regex-based extraction as fallback
                    patterns["color_naming"].extend(
                        self._extract_color_patterns_regex(content)
                    )
                    patterns["positioning"].extend(
                        self._extract_positioning_patterns_regex(content)
                    )

        # Deduplicate and limit each category
        for key in patterns:
            # Remove duplicates while preserving order
            seen = set()
            unique = []
            for item in patterns[key]:
                item_str = str(item)
                if item_str not in seen:
                    seen.add(item_str)
                    unique.append(item)
            patterns[key] = unique[:10]  # Limit to 10 per category

        # Log extraction stats
        total = sum(len(v) for v in patterns.values())
        logger.info(
            "[PATTERN] Extracted %d patterns: colors=%d, objects=%d, animations=%d, positioning=%d, anti=%d",
            total,
            len(patterns["color_naming"]),
            len(patterns["object_creation"]),
            len(patterns["animation_sequences"]),
            len(patterns["positioning"]),
            len(patterns["anti_patterns"]),
        )

        return patterns

    def _extract_color_patterns(self, tree: ast.AST, code: str) -> list[dict]:
        """Extract color naming conventions from AST.

        Looks for patterns like:
        - A_COLOR = BLUE_A
        - SIDE_COLORS = [RED, GREEN, BLUE]
        - color=BLUE, fill_color=GREEN
        """
        color_patterns = []

        for node in ast.walk(tree):
            # Assignment: COLOR_NAME = COLOR_VALUE
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # Check if it looks like a color constant
                        if name.upper() == name and ("COLOR" in name or name in KNOWN_3B1B_COLORS):
                            # Get the source for the value
                            try:
                                value_src = ast.get_source_segment(code, node.value)
                                if value_src:
                                    color_patterns.append({
                                        "type": "alias",
                                        "name": name,
                                        "value": value_src,
                                        "example": f"{name} = {value_src}",
                                    })
                            except Exception:
                                pass

            # Keyword arguments: color=BLUE, fill_color=RED
            if isinstance(node, ast.keyword):
                if node.arg and "color" in node.arg.lower():
                    try:
                        value_src = ast.get_source_segment(code, node.value)
                        if value_src and len(value_src) < 50:  # Skip complex expressions
                            color_patterns.append({
                                "type": "usage",
                                "param": node.arg,
                                "value": value_src,
                                "example": f"{node.arg}={value_src}",
                            })
                    except Exception:
                        pass

        return color_patterns

    def _extract_color_patterns_regex(self, code: str) -> list[dict]:
        """Fallback regex-based color pattern extraction."""
        color_patterns = []

        # Match: COLOR_NAME = COLOR_VALUE
        color_assign_re = re.compile(
            r'^([A-Z_]+COLOR[A-Z_]*)\s*=\s*([A-Z_]+(?:\s*\[.*?\])?)',
            re.MULTILINE
        )
        for match in color_assign_re.finditer(code):
            color_patterns.append({
                "type": "alias",
                "name": match.group(1),
                "value": match.group(2),
                "example": match.group(0).strip(),
            })

        # Match known 3b1b color names
        for color_name in KNOWN_3B1B_COLORS:
            if color_name in code:
                # Find the assignment
                pattern = re.compile(rf'{color_name}\s*=\s*([A-Z_\[\],\s]+)')
                match = pattern.search(code)
                if match:
                    color_patterns.append({
                        "type": "alias",
                        "name": color_name,
                        "value": match.group(1).strip(),
                        "example": f"{color_name} = {match.group(1).strip()}",
                    })

        return color_patterns

    def _extract_object_creation_patterns(self, tree: ast.AST, code: str) -> list[dict]:
        """Extract common mobject creation idioms.

        Looks for patterns like:
        - axes = Axes(x_range=[...], y_range=[...])
        - graph = axes.get_graph(lambda x: ...)
        - label = Tex(r"...").next_to(...)
        """
        creation_patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    var_name = target.id
                    value = node.value

                    # Direct class instantiation
                    if isinstance(value, ast.Call):
                        func_name = self._get_call_name(value)
                        if func_name and func_name in MOBJECT_CLASSES:
                            try:
                                call_src = ast.get_source_segment(code, value)
                                if call_src and len(call_src) < 200:
                                    creation_patterns.append({
                                        "type": "instantiation",
                                        "class": func_name,
                                        "variable": var_name,
                                        "example": f"{var_name} = {call_src}",
                                    })
                            except Exception:
                                pass

                    # Method chain (e.g., obj.get_graph().set_color(...))
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                        method_name = value.func.attr
                        if method_name.startswith("get_"):
                            try:
                                call_src = ast.get_source_segment(code, value)
                                if call_src and len(call_src) < 200:
                                    creation_patterns.append({
                                        "type": "factory_method",
                                        "method": method_name,
                                        "variable": var_name,
                                        "example": f"{var_name} = {call_src}",
                                    })
                            except Exception:
                                pass

        return creation_patterns

    def _extract_animation_sequences(self, tree: ast.AST, code: str) -> list[dict]:
        """Extract typical animation chains.

        Looks for patterns like:
        - self.play(Write(title), run_time=2)
        - self.play(Transform(a, b))
        - self.wait(1)
        - Sequences of self.play() calls
        """
        animation_sequences = []
        play_calls = []

        for node in ast.walk(tree):
            # Find self.play() and self.wait() calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and
                        node.func.value.id == "self"):
                        method = node.func.attr
                        if method in ("play", "wait", "add", "remove"):
                            try:
                                call_src = ast.get_source_segment(code, node)
                                if call_src and len(call_src) < 150:
                                    play_calls.append({
                                        "method": method,
                                        "line": getattr(node, 'lineno', 0),
                                        "example": call_src,
                                    })
                            except Exception:
                                pass

        # Group consecutive plays into sequences
        if play_calls:
            # Sort by line number
            play_calls.sort(key=lambda x: x["line"])

            current_seq = []
            for call in play_calls:
                if not current_seq:
                    current_seq.append(call)
                elif call["line"] - current_seq[-1]["line"] <= 3:
                    # Within 3 lines, part of same sequence
                    current_seq.append(call)
                else:
                    # Start new sequence
                    if len(current_seq) >= 2:
                        animation_sequences.append({
                            "type": "sequence",
                            "length": len(current_seq),
                            "methods": [c["method"] for c in current_seq],
                            "examples": [c["example"] for c in current_seq[:3]],
                        })
                    current_seq = [call]

            # Don't forget last sequence
            if len(current_seq) >= 2:
                animation_sequences.append({
                    "type": "sequence",
                    "length": len(current_seq),
                    "methods": [c["method"] for c in current_seq],
                    "examples": [c["example"] for c in current_seq[:3]],
                })

        return animation_sequences

    def _extract_positioning_patterns(self, tree: ast.AST, code: str) -> list[dict]:
        """Extract layout conventions.

        Looks for patterns like:
        - .next_to(obj, DOWN, buff=0.5)
        - .align_to(obj, LEFT)
        - .to_corner(UL)
        - VGroup(...).arrange(DOWN, buff=0.3)
        """
        positioning_patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    method = node.func.attr
                    if method in POSITIONING_METHODS:
                        try:
                            call_src = ast.get_source_segment(code, node)
                            # Extract just the method call part
                            if call_src:
                                # Find the .method_name(...) part
                                method_match = re.search(
                                    rf'\.{method}\([^)]*\)',
                                    call_src
                                )
                                if method_match and len(method_match.group(0)) < 100:
                                    positioning_patterns.append({
                                        "type": "method",
                                        "method": method,
                                        "example": method_match.group(0),
                                    })
                        except Exception:
                            pass

        return positioning_patterns

    def _extract_positioning_patterns_regex(self, code: str) -> list[dict]:
        """Fallback regex-based positioning pattern extraction."""
        positioning_patterns = []

        for method in POSITIONING_METHODS:
            pattern = re.compile(rf'\.{method}\([^)]*\)')
            for match in pattern.finditer(code):
                if len(match.group(0)) < 100:
                    positioning_patterns.append({
                        "type": "method",
                        "method": method,
                        "example": match.group(0),
                    })

        return positioning_patterns

    def _extract_anti_patterns(self, content: str) -> list[dict]:
        """Extract anti-patterns from error examples.

        Parses error documents with ERROR:/FIX: format.
        """
        anti_patterns = []

        # Parse ERROR:/FIX: format
        if "ERROR:" in content and "FIX:" in content:
            parts = content.split("FIX:")
            error_part = parts[0].replace("ERROR:", "").strip()
            fix_part = parts[-1].strip() if len(parts) > 1 else ""

            # Try to extract the specific wrong code
            wrong_code_match = re.search(r'```python?\n?(.*?)```', error_part, re.DOTALL)
            right_code_match = re.search(r'```python?\n?(.*?)```', fix_part, re.DOTALL)

            anti_patterns.append({
                "type": "error_fix",
                "error": error_part[:200] if len(error_part) > 200 else error_part,
                "fix": fix_part[:200] if len(fix_part) > 200 else fix_part,
                "wrong_code": wrong_code_match.group(1).strip() if wrong_code_match else None,
                "right_code": right_code_match.group(1).strip() if right_code_match else None,
            })

        # Also look for common mistake patterns
        mistake_patterns = [
            (r'tips\s*=\s*True', "Axes does not have tips parameter"),
            (r'x_length\s*=', "Axes uses width/height, not x_length/y_length"),
            (r'MathTex\s*\(', "Use Tex() not MathTex() in manimgl"),
            (r'Create\s*\(', "Use ShowCreation() not Create() in manimgl"),
            (r'\.plot\s*\(', "Use .get_graph() not .plot() in manimgl"),
        ]

        for pattern, description in mistake_patterns:
            if re.search(pattern, content):
                anti_patterns.append({
                    "type": "common_mistake",
                    "pattern": pattern,
                    "description": description,
                })

        return anti_patterns

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of a function/class being called."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _format_patterns_for_prompt(self, patterns: dict) -> str:
        """Format extracted patterns for inclusion in the prompt."""
        parts = []

        # Color naming conventions
        if patterns["color_naming"]:
            parts.append("COLOR NAMING CONVENTIONS (from 3blue1brown):")
            aliases = [p for p in patterns["color_naming"] if p.get("type") == "alias"]
            usages = [p for p in patterns["color_naming"] if p.get("type") == "usage"]

            if aliases:
                parts.append("  Semantic color aliases:")
                for p in aliases[:5]:
                    parts.append(f"    {p['example']}")

            if usages:
                parts.append("  Common color parameters:")
                unique_params = set()
                for p in usages[:5]:
                    param_val = f"{p['param']}={p['value']}"
                    if param_val not in unique_params:
                        unique_params.add(param_val)
                        parts.append(f"    {p['example']}")
            parts.append("")

        # Object creation patterns
        if patterns["object_creation"]:
            parts.append("OBJECT CREATION PATTERNS:")
            instantiations = [p for p in patterns["object_creation"] if p.get("type") == "instantiation"]
            factories = [p for p in patterns["object_creation"] if p.get("type") == "factory_method"]

            if instantiations:
                parts.append("  Direct instantiation:")
                for p in instantiations[:4]:
                    parts.append(f"    {p['example']}")

            if factories:
                parts.append("  Factory methods:")
                for p in factories[:4]:
                    parts.append(f"    {p['example']}")
            parts.append("")

        # Animation sequences
        if patterns["animation_sequences"]:
            parts.append("ANIMATION SEQUENCE PATTERNS:")
            for seq in patterns["animation_sequences"][:3]:
                methods_str = " -> ".join(seq["methods"][:5])
                parts.append(f"  Pattern: {methods_str}")
                for ex in seq["examples"][:2]:
                    parts.append(f"    {ex}")
            parts.append("")

        # Positioning patterns
        if patterns["positioning"]:
            parts.append("POSITIONING PATTERNS:")
            # Group by method
            by_method = {}
            for p in patterns["positioning"]:
                method = p.get("method", "other")
                if method not in by_method:
                    by_method[method] = []
                by_method[method].append(p["example"])

            for method, examples in list(by_method.items())[:5]:
                unique_examples = list(set(examples))[:2]
                parts.append(f"  {method}:")
                for ex in unique_examples:
                    parts.append(f"    {ex}")
            parts.append("")

        # Anti-patterns
        if patterns["anti_patterns"]:
            parts.append("ANTI-PATTERNS TO AVOID:")
            for anti in patterns["anti_patterns"][:5]:
                if anti.get("type") == "error_fix":
                    parts.append(f"  DON'T: {anti['error'][:100]}")
                    if anti.get("wrong_code"):
                        parts.append(f"    Wrong: {anti['wrong_code'][:80]}")
                    if anti.get("fix"):
                        parts.append(f"  DO: {anti['fix'][:100]}")
                    if anti.get("right_code"):
                        parts.append(f"    Right: {anti['right_code'][:80]}")
                elif anti.get("type") == "common_mistake":
                    parts.append(f"  AVOID: {anti['description']}")
            parts.append("")

        return "\n".join(parts)

    async def process(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
        plan: ScenePlan,
        narration_script: list[str] | None = None,
    ) -> tuple[str, str | None]:
        """Generate Manim code based on the scene plan.

        Args:
            prompt: Original user prompt
            analysis: Concept analysis from first agent
            plan: Scene plan from second agent
            narration_script: Pre-generated narration script (for code-audio sync)

        Returns:
            Tuple of (generated_code, original_template_code_or_none)
        """
        logger.debug("Generating code for: %s (%d segments)",
                     plan.title, len(plan.segments))
        if narration_script:
            logger.info("Code will follow %d-sentence narration script", len(narration_script))

        # Get RAG examples for few-shot context - TIERED PARALLEL APPROACH
        # Tier 1: Critical (API sigs + scenes + graph) - always run in parallel
        # Tier 2: Supplementary (patterns + errors) - only if tier 1 insufficient
        rag_context = ""
        high_quality_template = None
        error_patterns = []
        animation_patterns = []
        api_signatures = []
        method_sequences = []

        if self.rag_available and plan.rag_examples:
            logger.info("[RAG] Querying sources for code generation (tiered parallel)")

            # TIER 1: Critical queries in parallel
            # - RAG examples (similar scenes for few-shot)
            # - API signatures (parameter correctness)
            # - Method sequences (in-memory graph, very fast)
            tier1_results = await asyncio.gather(
                self._get_rag_examples(prompt, analysis),
                self._get_api_signatures(prompt, analysis),
                self._get_method_sequence_suggestions(analysis, plan),
                return_exceptions=True,
            )

            rag_result, api_result, sequences_result = tier1_results

            if isinstance(rag_result, tuple):
                rag_context, high_quality_template = rag_result
            elif isinstance(rag_result, Exception):
                logger.warning("[RAG] Error fetching examples: %s", rag_result)

            if isinstance(api_result, list):
                api_signatures = api_result
            elif isinstance(api_result, Exception):
                logger.warning("[RAG] Error fetching API signatures: %s", api_result)

            if isinstance(sequences_result, list):
                method_sequences = sequences_result
            elif isinstance(sequences_result, Exception):
                logger.warning("[API-GRAPH] Error fetching method sequences: %s", sequences_result)

            # TIER 2: Supplementary queries (parallel, but conditional)
            # Only fetch if we need more context
            need_patterns = not high_quality_template or len(rag_context) < 500
            need_errors = bool(api_signatures)  # Errors relate to API misuse

            if need_patterns or need_errors:
                tier2_tasks = []
                if need_patterns:
                    tier2_tasks.append(self._get_animation_patterns(prompt, analysis))
                else:
                    tier2_tasks.append(asyncio.sleep(0))  # Placeholder

                if need_errors:
                    tier2_tasks.append(self._get_error_patterns(prompt, analysis))
                else:
                    tier2_tasks.append(asyncio.sleep(0))  # Placeholder

                tier2_results = await asyncio.gather(*tier2_tasks, return_exceptions=True)

                if need_patterns and isinstance(tier2_results[0], list):
                    animation_patterns = tier2_results[0]
                elif need_patterns and isinstance(tier2_results[0], Exception):
                    logger.debug("[RAG] Patterns query failed: %s", tier2_results[0])

                if need_errors and isinstance(tier2_results[1], list):
                    error_patterns = tier2_results[1]
                elif need_errors and isinstance(tier2_results[1], Exception):
                    logger.debug("[RAG] Errors query failed: %s", tier2_results[1])
        else:
            # Still get method sequences from API graph (doesn't need RAG)
            method_sequences = await self._get_method_sequence_suggestions(analysis, plan)

        # Store template code for reference
        template_code = high_quality_template.get("content") if high_quality_template else None

        # Extract patterns from RAG examples using AST parsing
        # This gives us structured patterns instead of raw code dumps
        extracted_patterns = {}
        if rag_context or error_patterns or animation_patterns:
            # Collect all examples for pattern extraction
            all_examples = []
            if high_quality_template:
                all_examples.append(high_quality_template)
            # Add animation patterns (they have code content)
            for pattern in animation_patterns:
                all_examples.append(pattern)
            # Add error patterns as anti-pattern sources
            for err in error_patterns:
                all_examples.append({
                    "content": err.get("content", ""),
                    "metadata": {"category": "error"},
                })
            # Extract patterns from collected examples
            if all_examples:
                extracted_patterns = self._extract_patterns_from_examples(all_examples)

        # For VERY HIGH quality matches, use 3b1b code DIRECTLY (no conversion!)
        # manimgl will run it natively
        # Requirements: high similarity score AND sufficient code length
        if high_quality_template:
            score = high_quality_template.get("similarity_score", 0)
            template_len = len(high_quality_template.get("content", ""))
            if score >= DIRECT_USE_THRESHOLD and template_len >= DIRECT_USE_MIN_CHARS:
                code = self._use_directly(high_quality_template, plan)
                logger.info(
                    "[DIRECT-USE] Using 3b1b code directly for score=%.3f (%d chars)",
                    score, len(code)
                )
                return code, template_code
            elif score >= DIRECT_USE_THRESHOLD:
                logger.info(
                    "[DIRECT-USE] Skipping - code too short (%d chars < %d min)",
                    template_len, DIRECT_USE_MIN_CHARS
                )

        # Build generation prompt with extracted patterns
        gen_prompt = self._build_prompt(
            prompt, analysis, plan, rag_context, high_quality_template,
            error_patterns, animation_patterns, api_signatures, narration_script,
            method_sequences, extracted_patterns
        )

        # Generate code
        system = get_code_generator_system(self.config.latex_available)
        code = await self._llm_call(gen_prompt, system)

        return self._strip_fences(code), template_code

    def _use_directly(self, template: dict, plan: ScenePlan) -> str:
        """Use high-quality 3b1b code directly with import fixes.

        Fixes imports to work with manimlib and updates class name.
        """
        import re

        code = template.get("content", "")

        # Fix imports - replace 3b1b custom imports with standard manimlib
        import_replacements = [
            ("from manim_imports_ext import *", "from manimlib import *"),
            ("from big_ol_pile_of_manim_imports import *", "from manimlib import *"),
            ("from manimlib.imports import *", "from manimlib import *"),
            ("from manim import *", "from manimlib import *"),  # CE -> manimgl
        ]
        for old, new in import_replacements:
            code = code.replace(old, new)

        # Ensure manimlib import exists if missing entirely
        if "from manimlib import" not in code and "import manimlib" not in code:
            code = "from manimlib import *\n\n" + code

        # Update scene class name to match the plan title
        words = re.sub(r"[^a-zA-Z0-9\s]", "", plan.title).split()
        class_name = "".join(word.capitalize() for word in words) or "GeneratedScene"

        # Replace the existing class name (handle various Scene types)
        code = re.sub(
            r"class\s+(\w+)\s*\(\s*(\w*Scene)\s*\)",
            f"class {class_name}(\\2)",
            code,
            count=1,
        )

        return code

    async def _get_rag_examples(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> tuple[str, dict | None]:
        """Retrieve relevant code examples AND library docs from RAG.

        Returns:
            Tuple of (rag_context_string, high_quality_match_or_none)
            If a high-quality 3b1b match is found (score > 0.02), returns it separately
            for direct adaptation rather than just inspiration.
        """
        if not self.rag_available:
            logger.debug("[RAG] RAG not available for code examples")
            return "", None

        # Search for similar scenes (3b1b prioritized)
        search_query = f"{analysis.domain.value}: {prompt}"
        logger.info("[RAG] Code generator searching: %s", search_query[:100])

        # ALSO search library documentation for correct API usage
        doc_results = await self.rag.search_documentation(
            query=f"manimgl {' '.join(analysis.visual_elements or [])} Mobject Scene animation",
            n_results=3,
        )
        if doc_results:
            logger.info("[RAG] Found %d library docs for API reference", len(doc_results))

        results = await self.rag.search_similar_scenes(
            query=search_query,
            n_results=3,
            prioritize_3b1b=True,
        )

        if not results:
            logger.info("[RAG] No code examples found")
            return "", None

        logger.info(
            "[RAG] Found %d code examples (scores: %s)",
            len(results),
            [f"{r.get('similarity_score', 0):.3f}" for r in results],
        )

        # Check for high-quality 3b1b match
        high_quality_match = None
        for result in results:
            score = result.get("similarity_score", 0)
            content = result.get("content", "")
            is_3b1b = "manim_imports_ext" in content or "OldTex" in content

            # High match threshold: similarity > 0.02 (distance < 0.98) for 3b1b code
            if is_3b1b and score > 0.02:
                logger.info(
                    "[RAG] HIGH QUALITY 3b1b match found! score=%.3f - will use as template",
                    score,
                )
                high_quality_match = result
                break

        examples = []
        for i, result in enumerate(results[:2], 1):
            code = result.get("content", "")
            meta = result.get("metadata", {})
            score = result.get("similarity_score", 0)
            is_3b1b = "manim_imports_ext" in code or "OldTex" in code

            logger.debug(
                "[RAG] Example %d: source=%s, is_3b1b=%s, len=%d, score=%.3f",
                i, meta.get("source", "?"), is_3b1b, len(code), score,
            )
            # Don't truncate too aggressively - 3b1b examples can be 3000-4000 chars
            if len(code) > 4000:
                code = code[:4000] + "\n# ... (truncated)"

            source_label = " (3blue1brown original)" if is_3b1b else ""
            examples.append(f"Example {i}{source_label}:\n```python\n{code}\n```")

        # Add library documentation for correct API usage
        if doc_results:
            examples.append("\n" + "=" * 60)
            examples.append("MANIMGL LIBRARY REFERENCE (use these exact APIs):")
            examples.append("=" * 60)
            for doc in doc_results[:3]:
                doc_content = doc.get("content", "")
                doc_meta = doc.get("metadata", {})
                category = doc_meta.get("category", "")
                name = doc_meta.get("name", "")
                if len(doc_content) > 1500:
                    doc_content = doc_content[:1500] + "\n# ... (truncated)"
                examples.append(f"\n# {category}/{name}:\n```python\n{doc_content}\n```")

        return "\n\n".join(examples), high_quality_match

    async def _get_error_patterns(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> list[dict]:
        """Retrieve relevant error patterns to avoid common mistakes.

        Returns:
            List of error patterns with their fixes
        """
        if not self.rag_available:
            return []

        # Build a query that captures the visual elements being used
        visual_elements = analysis.visual_elements or []
        query_parts = [prompt]

        # Add specific method names that often have parameter issues
        problem_methods = ["get_riemann_rectangles", "get_area", "Axes", "NumberPlane"]
        for method in problem_methods:
            if any(method.lower() in elem.lower() for elem in visual_elements) or method.lower() in prompt.lower():
                query_parts.append(method)

        search_query = " ".join(query_parts)
        logger.debug("[RAG] Searching error patterns for: %s", search_query[:100])

        try:
            results = await self.rag.search_error_patterns(search_query, n_results=5)

            if results:
                logger.info("[RAG] Found %d error patterns to avoid", len(results))

            return results

        except Exception as e:
            logger.warning("[RAG] Error pattern search failed: %s", e)
            return []

    async def _get_animation_patterns(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> list[dict]:
        """Retrieve relevant 3b1b animation patterns for high-quality animations.

        IMPORTANT: Always returns at least 2 patterns for multi-animation videos.

        Returns:
            List of animation patterns with code templates (minimum 2)
        """
        if not self.rag_available:
            return []

        # Build a query that captures the math concepts and animation needs
        visual_elements = analysis.visual_elements or []
        key_concepts = analysis.key_concepts or []

        query_parts = [prompt]
        query_parts.extend(visual_elements)
        query_parts.extend(key_concepts)

        # Add domain-specific keywords
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["riemann", "rectangles", "integral", "area under"]):
            query_parts.append("riemann rectangles area integral approximation")
        if any(kw in prompt_lower for kw in ["derivative", "tangent", "slope", "secant"]):
            query_parts.append("derivative_definition secant tangent_line limit calculus")
        if any(kw in prompt_lower for kw in ["series", "sum", "convergence"]):
            query_parts.append("series partial_sums convergence")
        if any(kw in prompt_lower for kw in ["matrix", "transformation", "linear"]):
            query_parts.append("matrix_transformation linear_algebra")

        search_query = " ".join(query_parts)
        logger.debug("[RAG] Searching animation patterns for: %s", search_query[:100])

        results = []
        try:
            # Get primary patterns based on prompt (request 4 to ensure variety)
            results = await self.rag.search_animation_patterns(search_query, n_results=4)

            # ENSURE at least 2 patterns - add complementary technique patterns
            if len(results) < 2:
                # Add general technique patterns that work with any animation
                complementary_queries = [
                    "lagged_start_reveal staggered sequence",
                    "value_tracker_animation continuous smooth",
                    "indicate_with_flash highlight emphasis",
                    "build_up_construction step by step",
                    "fade_transition smooth animation",
                ]
                for fallback_query in complementary_queries:
                    if len(results) >= 2:
                        break
                    fallback_results = await self.rag.search_animation_patterns(
                        fallback_query, n_results=1
                    )
                    for r in fallback_results:
                        # Avoid duplicates
                        if not any(r.get("id") == existing.get("id") for existing in results):
                            results.append(r)
                            if len(results) >= 2:
                                break

            if results:
                logger.info("[RAG] Found %d animation patterns to apply (min 2 required)", len(results))

        except Exception as e:
            logger.warning("[RAG] Animation pattern search failed: %s", e)

        return results[:4]  # Return up to 4 patterns

    def _extract_method_names_from_prompt(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> list[str]:
        """Extract likely Manim method/class names from the user prompt.

        Uses keyword mapping and visual elements to identify specific API methods
        that should be looked up BEFORE code generation.

        Returns:
            List of specific method/class names to look up (e.g., ["Axes", "get_graph", "Transform"])
        """
        extracted = set()
        prompt_lower = prompt.lower()

        # 1. Map user keywords to manimgl API names
        for keyword, api_names in MANIM_KEYWORD_MAP.items():
            if keyword in prompt_lower:
                extracted.update(api_names)
                logger.debug("[API-LOOKUP] Keyword '%s' -> %s", keyword, api_names)

        # 2. Add visual elements from analysis (these are often actual class names)
        for elem in (analysis.visual_elements or []):
            # Clean and capitalize to match class names
            elem_clean = elem.strip()
            if elem_clean:
                extracted.add(elem_clean)
                # Also check if it maps to other methods
                elem_lower = elem_clean.lower()
                if elem_lower in MANIM_KEYWORD_MAP:
                    extracted.update(MANIM_KEYWORD_MAP[elem_lower])

        # 3. Extract CamelCase words that might be class names
        camel_case_pattern = re.compile(r'\b([A-Z][a-zA-Z]+)\b')
        for match in camel_case_pattern.findall(prompt):
            extracted.add(match)

        # 4. Extract method-like names (e.g., get_graph, add_coordinate_labels)
        method_pattern = re.compile(r'\b(get_\w+|add_\w+|set_\w+|animate\.\w+)\b')
        for match in method_pattern.findall(prompt):
            extracted.add(match)

        # 5. Add core classes that are almost always needed
        core_classes = {"Scene", "VGroup", "self.play", "self.wait"}
        extracted.update(core_classes)

        result = list(extracted)
        logger.info("[API-LOOKUP] Extracted %d method/class names from prompt: %s",
                   len(result), result[:10])  # Log first 10
        return result

    async def _get_api_signatures_for_methods(
        self,
        method_names: list[str],
    ) -> list[dict]:
        """Look up specific API signatures by method/class name.

        This is the PRE-GENERATION lookup that fetches exact signatures
        for methods we expect the LLM to use.

        Returns:
            List of API signatures with exact parameters
        """
        if not self.rag_available or not method_names:
            return []

        signatures = []
        seen_ids = set()

        # First, try to get exact matches by name
        for name in method_names[:15]:  # Limit to avoid too many lookups
            # Try common class prefixes for method names
            prefixes_to_try = ["", "Axes.", "Scene.", "VMobject.", "Mobject.", "Animation."]

            for prefix in prefixes_to_try:
                full_name = f"{prefix}{name}" if prefix else name
                try:
                    result = await self.rag.get_api_signature(full_name)
                    if result and result.get("document_id") not in seen_ids:
                        seen_ids.add(result["document_id"])
                        signatures.append(result)
                        logger.debug("[API-LOOKUP] Found exact signature: %s", full_name)
                        break  # Found it, don't try other prefixes
                except Exception:
                    pass

        # Then, do a semantic search for methods we couldn't find exactly
        if len(signatures) < 8:  # Want at least 8 signatures
            remaining_slots = 8 - len(signatures)
            search_query = " ".join(method_names[:10])
            try:
                search_results = await self.rag.search_api_signatures(
                    search_query, n_results=remaining_slots + 3
                )
                for result in search_results:
                    if result.get("document_id") not in seen_ids:
                        seen_ids.add(result["document_id"])
                        signatures.append(result)
                        if len(signatures) >= 12:  # Cap at 12 total
                            break
            except Exception as e:
                logger.warning("[API-LOOKUP] Search failed: %s", e)

        logger.info("[API-LOOKUP] Retrieved %d API signatures for pre-generation constraints",
                   len(signatures))
        return signatures

    async def _get_api_signatures(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
    ) -> list[dict]:
        """Retrieve relevant API signatures for correct parameter usage.

        This is a convenience method that combines extraction and lookup.

        Returns:
            List of API signatures with parameters and docstrings
        """
        if not self.rag_available:
            return []

        # Extract method names from the prompt
        method_names = self._extract_method_names_from_prompt(prompt, analysis)

        # Look up signatures for those specific methods
        return await self._get_api_signatures_for_methods(method_names)

    def _build_prompt(
        self,
        prompt: str,
        analysis: ConceptAnalysis,
        plan: ScenePlan,
        rag_context: str,
        high_quality_template: dict | None = None,
        error_patterns: list[dict] | None = None,
        animation_patterns: list[dict] | None = None,
        api_signatures: list[dict] | None = None,
        narration_script: list[str] | None = None,
        method_sequences: list[dict] | None = None,
        extracted_patterns: dict | None = None,
    ) -> str:
        """Build the code generation prompt with all context and extracted patterns."""
        parts = [
            f"Create a Manim animation: {prompt}",
            "",
        ]

        # CRITICAL: If narration script provided, code MUST match it exactly
        if narration_script:
            parts.extend([
                "=" * 60,
                "NARRATION SCRIPT - CODE MUST MATCH THIS EXACTLY",
                "=" * 60,
                "",
                "The animation will have audio narration. Each sentence below corresponds",
                "to ONE visual step. Your code MUST create visuals that sync with this script:",
                "",
            ])
            for i, sentence in enumerate(narration_script, 1):
                parts.append(f"  {i}. {sentence}")
            parts.extend([
                "",
                "REQUIREMENTS:",
                "- Each self.play() or animation should correspond to ONE narration sentence",
                "- Add self.wait(2-3) after each animation for narration time",
                "- The visual sequence must match the narration sequence exactly",
                "- Total animation duration should be ~{} seconds".format(len(narration_script) * 4),
                "",
                "=" * 60,
                "",
            ])

        parts.extend([
            f"Scene: {plan.title}",
            f"Total duration: ~{plan.total_duration:.1f} seconds",
            "",
            "Segments:",
        ])

        for i, seg in enumerate(plan.segments, 1):
            parts.append(f"{i}. {seg.name} ({seg.duration:.1f}s)")
            parts.append(f"   - {seg.description}")
            if seg.mobjects:
                parts.append(f"   - Objects: {', '.join(seg.mobjects)}")
            if seg.animations:
                parts.append(f"   - Animations: {', '.join(seg.animations)}")

        if analysis.visual_elements:
            parts.extend([
                "",
                f"Suggested visual elements: {', '.join(analysis.visual_elements)}",
            ])

        # If we have a high-quality 3b1b template, use it with API conversion
        if high_quality_template:
            template_code = high_quality_template.get("content", "")
            parts.extend([
                "",
                "=" * 60,
                "3BLUE1BROWN REFERENCE CODE - ADAPT THIS!",
                "=" * 60,
                "",
                "```python",
                template_code,
                "```",
                "",
                "MANDATORY API CONVERSIONS (apply ALL of these):",
                "- from manim_imports_ext import * → from manim import *",
                "- OldTex(...) → MathTex(r'...')",
                "- OldTexText(...) → Text(...)",
                "- .move_to(x, point_to_align=Y) → .move_to(x, aligned_edge=Y)",
                "- self.play(obj.method, val) → self.play(obj.animate.method(val))",
                "- RightAngle(triangle, ...) → Elbow() or just skip right angle marks",
                "- SIDE_COLORS → [RED, GREEN, BLUE]",
                "- compass_directions(4) → [UP, RIGHT, DOWN, LEFT]",
                "- get_corner(DL) works the same",
                "- Polygon, Square, Line, VGroup, MathTex, Text all work the same",
                "",
                "Keep the STRUCTURE, TIMING, and VISUAL QUALITY of the reference.",
                "The animation logic is excellent - just fix the API calls.",
            ])
        elif rag_context:
            parts.extend([
                "",
                "Reference examples (study their style and patterns):",
                rag_context,
            ])

        # Add extracted patterns from RAG examples (structured, not raw code)
        # These provide reusable idioms instead of raw code dumps
        if extracted_patterns and any(extracted_patterns.values()):
            formatted_patterns = self._format_patterns_for_prompt(extracted_patterns)
            if formatted_patterns.strip():
                parts.extend([
                    "",
                    "=" * 60,
                    "EXTRACTED PATTERNS FROM 3BLUE1BROWN CODE",
                    "=" * 60,
                    "",
                    "The following patterns were extracted from verified 3b1b code.",
                    "Use these idioms for consistent, high-quality animations:",
                    "",
                    formatted_patterns,
                ])

        # Add error patterns to avoid
        if error_patterns:
            parts.extend([
                "",
                "=" * 60,
                "MANIMGL PITFALLS TO AVOID (these will cause errors!):",
                "=" * 60,
            ])
            for pattern in error_patterns[:5]:  # Limit to 5 patterns
                content = pattern.get("content", "")
                # Parse the error/fix from the document format
                if "ERROR:" in content and "FIX:" in content:
                    # Extract just the key parts
                    error_part = content.split("FIX:")[0].replace("ERROR:", "").strip()
                    fix_part = content.split("FIX:")[-1].strip()
                    # Clean up for concise display
                    if len(error_part) > 100:
                        error_part = error_part[:100] + "..."
                    if len(fix_part) > 200:
                        fix_part = fix_part[:200] + "..."
                    parts.append(f"- DON'T: {error_part}")
                    parts.append(f"  DO: {fix_part}")
                else:
                    # Fallback for other formats
                    if len(content) > 300:
                        content = content[:300] + "..."
                    parts.append(f"- {content}")

        # Add animation patterns (3b1b-style techniques)
        # IMPORTANT: Use at least 2 patterns for professional multi-animation videos
        if animation_patterns:
            n_patterns = min(len(animation_patterns), 4)
            parts.extend([
                "",
                "=" * 60,
                "3BLUE1BROWN ANIMATION STYLE - CRITICAL REQUIREMENTS",
                "=" * 60,
                "",
                "The hallmark of 3b1b animations is REACTIVE/DECLARATIVE code:",
                "",
                "1. USE ValueTracker + updaters for smooth continuous animations:",
                "   ```python",
                "   t_tracker = ValueTracker(0)",
                "   dot.add_updater(lambda m: m.move_to(axes.c2p(t_tracker.get_value(), func(t_tracker.get_value()))))",
                "   self.play(t_tracker.animate.set_value(5), run_time=4)",
                "   ```",
                "",
                "2. USE always_redraw() for objects that depend on changing values:",
                "   ```python",
                "   area = always_redraw(lambda: axes.get_area_under_graph(graph, x_range=[0, t_tracker.get_value()]))",
                "   ```",
                "",
                "3. USE TracedPath for trajectories:",
                "   ```python",
                "   trace = TracedPath(dot.get_center, stroke_color=BLUE)",
                "   ```",
                "",
                "4. USE .become() in updaters for complex reconstructions:",
                "   ```python",
                "   area.add_updater(lambda m: m.become(axes.get_area_under_graph(graph)))",
                "   ```",
                "",
                "5. USE TransformMatchingTex for equation morphing:",
                "   ```python",
                "   self.play(TransformMatchingTex(eq1, eq2))",
                "   ```",
                "",
                "6. USE time_span for choreographed animations:",
                "   ```python",
                "   self.play(anim1, anim2.set_anim_args(time_span=(2, 5)), run_time=6)",
                "   ```",
                "",
                "DO NOT write static code with manual transforms between states!",
                "DO write reactive code where mobjects update automatically!",
                "",
                "3B1B VISUAL STYLE REQUIREMENTS:",
                "- ALWAYS use Dots to mark key points on curves/graphs",
                "- For derivatives/tangents: add Dot at base point AND Dot at second point",
                "- For secant lines: show BOTH intersection points with colored Dots",
                "- For Riemann sums: add Dots at sample points on the curve",
                "- Use color contrast: curve=BLUE, points=YELLOW/RED, lines=GREEN",
                "- Moving points should use always_redraw() to stay on the curve",
                "- Add subtle visual elements like dashed lines connecting key points",
                "",
                "LAYOUT (auto-fixed, but prefer these patterns):",
                "- Use next_to(obj, direction, buff=0.2) for labels near objects",
                "- Use VGroup().arrange(DOWN, buff=0.3) for multiple related items",
                "- Prefer relative positioning over absolute (next_to > move_to)",
                "",
                "CRITICAL LATEX/TEX RULES:",
                "- NEVER use '&' in TexText - it's a special character! Use 'and' instead",
                "- NEVER use set_to_corner() - use to_corner() instead",
                "- In Tex/MathTex, escape special chars: \\& \\% \\$ \\# \\_ \\{ \\}",
                "- For alignment in Tex, use align environment not raw &",
                "",
                "=" * 60,
                f"ANIMATION PATTERNS TO APPLY (use at least 2 of these {n_patterns}):",
                "=" * 60,
                "",
            ])
            for i, pattern in enumerate(animation_patterns[:4], 1):  # Up to 4 patterns
                content = pattern.get("content", "")
                meta = pattern.get("metadata", {})
                pattern_name = meta.get("name", "pattern")
                category = meta.get("category", "")

                parts.append(f"\n### PATTERN {i}: {pattern_name} ({category})")

                # Extract code template from content
                if "## Code Template" in content:
                    # Get everything after Code Template header
                    template_start = content.find("## Code Template")
                    if template_start != -1:
                        template_section = content[template_start:]
                        # Truncate if too long
                        if len(template_section) > 1200:
                            template_section = template_section[:1200] + "\n# ... (truncated)"
                        parts.append(template_section)
                else:
                    # Just use the content directly
                    if len(content) > 1200:
                        content = content[:1200] + "\n# ... (truncated)"
                    parts.append(content)

            # Reminder at the end
            parts.extend([
                "",
                "=" * 60,
                "REMEMBER: Your animation MUST use at least 2 of the above patterns!",
                "=" * 60,
            ])

        # Add API signatures as STRICT CONSTRAINTS for correct parameter usage
        if api_signatures:
            parts.extend([
                "",
                "=" * 60,
                "API CONSTRAINTS - MANDATORY METHOD SIGNATURES",
                "=" * 60,
                "",
                "CRITICAL: You MUST use ONLY these exact method signatures.",
                "DO NOT invent parameters. DO NOT guess parameter names.",
                "If a parameter is not listed below, it DOES NOT EXIST.",
                "",
            ])

            # Format each signature as a strict constraint
            for i, sig in enumerate(api_signatures[:12], 1):  # Up to 12 signatures
                content = sig.get("content", "")
                meta = sig.get("metadata", {})
                method_name = meta.get("name", "")
                class_name = meta.get("class_name", "")
                parameters = meta.get("parameters", "") or meta.get("valid_params", "")
                # Get REQUIRED params - critical for preventing errors!
                required_params = meta.get("required_params", "") or meta.get("required", "")

                # Build the header
                if class_name and method_name:
                    header = f"{class_name}.{method_name}"
                elif method_name:
                    header = method_name
                else:
                    header = f"Method {i}"

                parts.append(f"### {i}. {header}")

                # Extract and format the signature line if available
                if parameters:
                    parts.append(f"**Signature:** `{header}({parameters})`")

                # CRITICAL: Emphasize REQUIRED params to prevent TypeError
                if required_params:
                    parts.append(f"⚠️ **REQUIRED (must provide):** `{required_params}`")
                parts.append("")

                # Format the full content with parameter details
                # Truncate long content but preserve signature info
                if len(content) > 600:
                    # Try to preserve the signature and first few params
                    lines = content.split('\n')
                    kept_lines = []
                    char_count = 0
                    for line in lines:
                        if char_count + len(line) > 600:
                            kept_lines.append("# ... (see full docs for more parameters)")
                            break
                        kept_lines.append(line)
                        char_count += len(line)
                    content = '\n'.join(kept_lines)

                parts.append(f"```python\n{content}\n```")
                parts.append("")

            # Add strong closing reminder
            parts.extend([
                "=" * 60,
                "REMEMBER: Use ONLY the parameters shown above.",
                "Common mistakes to AVOID:",
                "- tips=True on Axes (WRONG - no such parameter)",
                "- x_length/y_length on Axes (WRONG - use width/height)",
                "- MathTex (WRONG - use Tex in manimgl)",
                "- Create (WRONG - use ShowCreation in manimgl)",
                "- axes.plot() (WRONG - use axes.get_graph())",
                "=" * 60,
            ])

        # Add method sequence suggestions from API graph
        if method_sequences:
            parts.extend([
                "",
                "=" * 60,
                "RECOMMENDED METHOD SEQUENCES (from 3b1b analysis)",
                "=" * 60,
                "",
                "These are common method chains observed in 3blue1brown code.",
                "Use these patterns for idiomatic animations:",
                "",
            ])

            for suggestion in method_sequences[:8]:  # Limit to 8 suggestions
                element = suggestion.get("element", "")
                segment = suggestion.get("segment", "")

                if "next_methods" in suggestion:
                    # Element -> next methods
                    next_methods = suggestion["next_methods"]
                    methods_str = ", ".join(
                        f"{m['name']} ({m['weight']:.1%})"
                        for m in next_methods[:3]
                    )
                    parts.append(f"- After {element}: typically use {methods_str}")

                elif "paired_with" in suggestion:
                    # Element commonly paired with
                    paired = suggestion["paired_with"]
                    paired_str = ", ".join(p["name"] for p in paired)
                    parts.append(f"- {element} often used with: {paired_str}")

                elif "can_follow_with" in suggestion:
                    # Animation can follow with
                    anim = suggestion.get("animation", "")
                    follow = suggestion["can_follow_with"]
                    parts.append(f"- In '{segment}': after {anim}, consider {', '.join(follow[:3])}")

            parts.extend([
                "",
                "These are suggestions - use them when appropriate for smooth transitions.",
                "",
            ])

        return "\n".join(parts)
