"""API Relationship Graph using NetworkX for smarter code generation.

This module builds a graph of Manim API relationships to enable smarter code generation:
- Nodes represent API methods (animations, mobjects, operations)
- Edges represent relationships (returns, commonly_follows, commonly_pairs)

The graph is built from:
1. Static analysis of manimgl API signatures (return types -> input types)
2. Analysis of indexed 3b1b scenes (method call sequences from real code)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim_mcp.core.rag import ChromaDBService

logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """Types of relationships between API methods."""

    RETURNS = "returns"  # Method A returns type that Method B accepts
    COMMONLY_FOLLOWS = "commonly_follows"  # Method A often appears after Method B
    COMMONLY_PAIRS = "commonly_pairs"  # Methods often used together in same scene


@dataclass
class APINode:
    """Represents a Manim API method or class in the graph."""

    name: str  # e.g., "FadeIn", "Transform", "Circle"
    category: str = "animation"  # animation, mobject, method, property
    input_types: list[str] = field(default_factory=list)  # Types this method accepts
    return_type: str | None = None  # Type this method returns
    description: str = ""
    usage_count: int = 0  # How often seen in 3b1b code


@dataclass
class APIEdge:
    """Represents a relationship between two API methods."""

    source: str  # Source method name
    target: str  # Target method name
    relation: RelationType
    weight: float = 1.0  # Strength of relationship (frequency in 3b1b code)
    examples: list[str] = field(default_factory=list)  # Code examples showing this pattern


class ManimAPIGraph:
    """NetworkX-based graph of Manim API relationships.

    Enables:
    - Finding valid method sequences
    - Suggesting commonly-paired methods
    - Generating idiomatic 3b1b-style code patterns
    """

    def __init__(self) -> None:
        """Initialize the API graph."""
        self._graph = None
        self._nodes: dict[str, APINode] = {}
        self._initialized = False

    async def initialize(self, rag: ChromaDBService | None = None) -> None:
        """Initialize the graph with static API data and optionally 3b1b scene analysis.

        Args:
            rag: Optional ChromaDB service to analyze indexed 3b1b scenes
        """
        try:
            import networkx as nx
            self._graph = nx.DiGraph()
        except ImportError:
            logger.warning("networkx not installed - API graph unavailable. "
                         "Install with: pip install networkx")
            return

        # Build static API knowledge
        self._build_static_graph()

        # Enhance with 3b1b scene analysis if RAG available
        if rag and rag.available:
            await self._analyze_3b1b_scenes(rag)

        self._initialized = True
        logger.info("ManimAPIGraph initialized with %d nodes, %d edges",
                   self._graph.number_of_nodes(),
                   self._graph.number_of_edges())

    @property
    def available(self) -> bool:
        """Check if the graph is available for use."""
        return self._initialized and self._graph is not None

    def _build_static_graph(self) -> None:
        """Build graph from static knowledge of manimgl API."""
        import networkx as nx

        # === ANIMATION NODES ===
        animations = {
            # Basic animations
            "FadeIn": APINode("FadeIn", "animation", ["Mobject"], "AnimationGroup",
                            "Fade in a mobject with optional direction"),
            "FadeOut": APINode("FadeOut", "animation", ["Mobject"], "AnimationGroup",
                             "Fade out a mobject"),
            "Create": APINode("Create", "animation", ["VMobject"], "AnimationGroup",
                            "Draw a VMobject from nothing"),
            "Write": APINode("Write", "animation", ["VMobject"], "AnimationGroup",
                           "Write text or TeX"),
            "Uncreate": APINode("Uncreate", "animation", ["VMobject"], "AnimationGroup",
                              "Reverse of Create"),
            "DrawBorderThenFill": APINode("DrawBorderThenFill", "animation",
                                         ["VMobject"], "AnimationGroup",
                                         "Draw border then fill in"),
            "ShowCreation": APINode("ShowCreation", "animation", ["VMobject"], "AnimationGroup",
                                   "Show creation of VMobject"),
            "GrowFromCenter": APINode("GrowFromCenter", "animation", ["Mobject"], "AnimationGroup",
                                     "Grow mobject from center"),
            "GrowFromPoint": APINode("GrowFromPoint", "animation", ["Mobject"], "AnimationGroup",
                                    "Grow mobject from a point"),
            "GrowArrow": APINode("GrowArrow", "animation", ["Arrow"], "AnimationGroup",
                                "Grow an arrow"),
            "SpinInFromNothing": APINode("SpinInFromNothing", "animation", ["Mobject"], "AnimationGroup",
                                        "Spin in from nothing"),

            # Transform animations
            "Transform": APINode("Transform", "animation", ["Mobject", "Mobject"], "AnimationGroup",
                               "Transform one mobject into another"),
            "ReplacementTransform": APINode("ReplacementTransform", "animation",
                                           ["Mobject", "Mobject"], "AnimationGroup",
                                           "Transform and replace (different from Transform)"),
            "TransformMatchingTex": APINode("TransformMatchingTex", "animation",
                                           ["MathTex", "MathTex"], "AnimationGroup",
                                           "Transform matching TeX parts"),
            "TransformMatchingShapes": APINode("TransformMatchingShapes", "animation",
                                              ["VMobject", "VMobject"], "AnimationGroup",
                                              "Transform matching shapes"),
            "ClockwiseTransform": APINode("ClockwiseTransform", "animation",
                                         ["Mobject", "Mobject"], "AnimationGroup",
                                         "Transform with clockwise rotation"),
            "CounterclockwiseTransform": APINode("CounterclockwiseTransform", "animation",
                                                ["Mobject", "Mobject"], "AnimationGroup",
                                                "Transform with counter-clockwise rotation"),
            "MoveToTarget": APINode("MoveToTarget", "animation", ["Mobject"], "AnimationGroup",
                                   "Move mobject to its target"),

            # Movement animations
            "MoveAlongPath": APINode("MoveAlongPath", "animation", ["Mobject", "VMobject"],
                                    "AnimationGroup", "Move along a path"),
            "Rotate": APINode("Rotate", "animation", ["Mobject"], "AnimationGroup",
                            "Rotate a mobject"),
            "Rotating": APINode("Rotating", "animation", ["Mobject"], "AnimationGroup",
                              "Continuous rotation"),
            "ApplyMethod": APINode("ApplyMethod", "animation", ["method"], "AnimationGroup",
                                  "Apply a method as animation"),

            # Indication animations
            "Indicate": APINode("Indicate", "animation", ["Mobject"], "AnimationGroup",
                              "Briefly highlight a mobject"),
            "Flash": APINode("Flash", "animation", ["Mobject"], "AnimationGroup",
                           "Flash effect"),
            "Circumscribe": APINode("Circumscribe", "animation", ["Mobject"], "AnimationGroup",
                                   "Draw a shape around mobject"),
            "ShowPassingFlash": APINode("ShowPassingFlash", "animation", ["VMobject"], "AnimationGroup",
                                       "Show a passing flash along path"),
            "ShowCreationThenFadeOut": APINode("ShowCreationThenFadeOut", "animation",
                                              ["VMobject"], "AnimationGroup",
                                              "Create then fade out"),
            "FocusOn": APINode("FocusOn", "animation", ["Mobject"], "AnimationGroup",
                             "Focus camera on mobject"),
            "Wiggle": APINode("Wiggle", "animation", ["Mobject"], "AnimationGroup",
                            "Wiggle a mobject"),

            # Update animations
            "UpdateFromFunc": APINode("UpdateFromFunc", "animation", ["Mobject", "function"],
                                     "AnimationGroup", "Update mobject from function"),
            "UpdateFromAlphaFunc": APINode("UpdateFromAlphaFunc", "animation",
                                          ["Mobject", "function"], "AnimationGroup",
                                          "Update mobject from alpha function"),
            "MaintainPositionRelativeTo": APINode("MaintainPositionRelativeTo", "animation",
                                                 ["Mobject", "Mobject"], "AnimationGroup",
                                                 "Maintain relative position"),

            # Animation groups
            "AnimationGroup": APINode("AnimationGroup", "animation", ["Animation..."],
                                     "AnimationGroup", "Group multiple animations"),
            "Succession": APINode("Succession", "animation", ["Animation..."],
                                 "AnimationGroup", "Play animations in succession"),
            "LaggedStart": APINode("LaggedStart", "animation", ["Animation..."],
                                  "AnimationGroup", "Staggered animation start"),
            "LaggedStartMap": APINode("LaggedStartMap", "animation",
                                     ["Animation", "Mobject..."], "AnimationGroup",
                                     "Map animation with lag"),
        }

        # === MOBJECT NODES ===
        mobjects = {
            # Basic shapes
            "Circle": APINode("Circle", "mobject", [], "VMobject", "A circle"),
            "Square": APINode("Square", "mobject", [], "VMobject", "A square"),
            "Rectangle": APINode("Rectangle", "mobject", [], "VMobject", "A rectangle"),
            "Triangle": APINode("Triangle", "mobject", [], "VMobject", "A triangle"),
            "Polygon": APINode("Polygon", "mobject", ["point..."], "VMobject", "A polygon"),
            "RegularPolygon": APINode("RegularPolygon", "mobject", ["n"], "VMobject",
                                     "Regular polygon with n sides"),
            "Dot": APINode("Dot", "mobject", [], "VMobject", "A small dot"),
            "Line": APINode("Line", "mobject", ["point", "point"], "VMobject", "A line"),
            "Arrow": APINode("Arrow", "mobject", ["point", "point"], "VMobject", "An arrow"),
            "DoubleArrow": APINode("DoubleArrow", "mobject", ["point", "point"], "VMobject",
                                  "Arrow with heads on both ends"),
            "Vector": APINode("Vector", "mobject", ["direction"], "VMobject", "A vector"),
            "Arc": APINode("Arc", "mobject", [], "VMobject", "An arc"),
            "ArcBetweenPoints": APINode("ArcBetweenPoints", "mobject", ["point", "point"],
                                       "VMobject", "Arc between two points"),
            "CurvedArrow": APINode("CurvedArrow", "mobject", ["point", "point"], "VMobject",
                                  "A curved arrow"),
            "Elbow": APINode("Elbow", "mobject", [], "VMobject", "Right angle indicator"),

            # Text/Math
            "Text": APINode("Text", "mobject", ["string"], "VMobject", "Plain text"),
            "MathTex": APINode("MathTex", "mobject", ["latex"], "VMobject", "LaTeX math"),
            "Tex": APINode("Tex", "mobject", ["latex"], "VMobject", "LaTeX text"),
            "TexText": APINode("TexText", "mobject", ["text"], "VMobject", "Text with TeX"),
            "Integer": APINode("Integer", "mobject", ["number"], "VMobject", "Integer display"),
            "DecimalNumber": APINode("DecimalNumber", "mobject", ["number"], "VMobject",
                                    "Decimal number display"),

            # Graphs and plots
            "Axes": APINode("Axes", "mobject", [], "CoordinateSystem", "2D axes"),
            "ThreeDAxes": APINode("ThreeDAxes", "mobject", [], "CoordinateSystem", "3D axes"),
            "NumberPlane": APINode("NumberPlane", "mobject", [], "CoordinateSystem",
                                  "Number plane with grid"),
            "ComplexPlane": APINode("ComplexPlane", "mobject", [], "CoordinateSystem",
                                   "Complex number plane"),
            "NumberLine": APINode("NumberLine", "mobject", [], "VMobject", "Number line"),
            "ParametricCurve": APINode("ParametricCurve", "mobject", ["function"], "VMobject",
                                      "Parametric curve"),
            "FunctionGraph": APINode("FunctionGraph", "mobject", ["function"], "VMobject",
                                    "Graph of a function"),

            # Groups
            "VGroup": APINode("VGroup", "mobject", ["VMobject..."], "VGroup",
                            "Group of VMobjects"),
            "Group": APINode("Group", "mobject", ["Mobject..."], "Group",
                           "Group of Mobjects"),

            # 3D
            "Sphere": APINode("Sphere", "mobject", [], "Surface", "A sphere"),
            "Cube": APINode("Cube", "mobject", [], "Surface", "A cube"),
            "Prism": APINode("Prism", "mobject", [], "Surface", "A prism"),
            "Cylinder": APINode("Cylinder", "mobject", [], "Surface", "A cylinder"),
            "Cone": APINode("Cone", "mobject", [], "Surface", "A cone"),
            "Torus": APINode("Torus", "mobject", [], "Surface", "A torus"),
            "Surface": APINode("Surface", "mobject", ["function"], "Surface", "A surface"),

            # Special
            "TracedPath": APINode("TracedPath", "mobject", ["function"], "VMobject",
                                 "Path traced by a point"),
            "ValueTracker": APINode("ValueTracker", "mobject", ["number"], "ValueTracker",
                                   "Track a value for animations"),
            "Brace": APINode("Brace", "mobject", ["Mobject"], "VMobject", "A brace"),
            "BraceBetweenPoints": APINode("BraceBetweenPoints", "mobject", ["point", "point"],
                                         "VMobject", "Brace between two points"),
            "SurroundingRectangle": APINode("SurroundingRectangle", "mobject", ["Mobject"],
                                           "VMobject", "Rectangle around mobject"),
            "BackgroundRectangle": APINode("BackgroundRectangle", "mobject", ["Mobject"],
                                          "VMobject", "Background for mobject"),
        }

        # === METHOD NODES ===
        methods = {
            # Positioning
            "move_to": APINode("move_to", "method", ["Mobject", "point"], "Mobject",
                             "Move center to point"),
            "next_to": APINode("next_to", "method", ["Mobject", "Mobject", "direction"],
                             "Mobject", "Position next to another mobject"),
            "shift": APINode("shift", "method", ["Mobject", "vector"], "Mobject",
                           "Shift by vector"),
            "to_corner": APINode("to_corner", "method", ["Mobject", "corner"], "Mobject",
                               "Move to corner"),
            "to_edge": APINode("to_edge", "method", ["Mobject", "edge"], "Mobject",
                             "Move to edge"),
            "align_to": APINode("align_to", "method", ["Mobject", "Mobject", "direction"],
                              "Mobject", "Align to another mobject"),
            "arrange": APINode("arrange", "method", ["VGroup", "direction"], "VGroup",
                             "Arrange submobjects"),
            "arrange_in_grid": APINode("arrange_in_grid", "method", ["VGroup"], "VGroup",
                                      "Arrange in a grid"),

            # Transformation
            "scale": APINode("scale", "method", ["Mobject", "factor"], "Mobject",
                           "Scale mobject"),
            "rotate": APINode("rotate", "method", ["Mobject", "angle"], "Mobject",
                            "Rotate mobject"),
            "flip": APINode("flip", "method", ["Mobject", "axis"], "Mobject",
                          "Flip mobject"),
            "stretch": APINode("stretch", "method", ["Mobject", "factor", "dim"], "Mobject",
                             "Stretch mobject"),

            # Appearance
            "set_color": APINode("set_color", "method", ["Mobject", "color"], "Mobject",
                               "Set color"),
            "set_fill": APINode("set_fill", "method", ["VMobject", "color", "opacity"],
                              "VMobject", "Set fill color and opacity"),
            "set_stroke": APINode("set_stroke", "method", ["VMobject", "color", "width"],
                                "VMobject", "Set stroke properties"),
            "set_opacity": APINode("set_opacity", "method", ["Mobject", "opacity"], "Mobject",
                                 "Set opacity"),
            "fade": APINode("fade", "method", ["Mobject", "darkness"], "Mobject",
                          "Fade mobject"),

            # Graph methods
            "plot": APINode("plot", "method", ["Axes", "function"], "ParametricCurve",
                          "Plot a function"),
            "get_graph": APINode("get_graph", "method", ["Axes", "function"], "ParametricCurve",
                               "Get graph of function"),
            "get_area": APINode("get_area", "method", ["Axes", "graph"], "VMobject",
                              "Get area under graph"),
            "get_riemann_rectangles": APINode("get_riemann_rectangles", "method",
                                             ["Axes", "graph"], "VGroup",
                                             "Get Riemann sum rectangles"),
            "get_secant_slope_group": APINode("get_secant_slope_group", "method",
                                             ["Axes", "graph", "x"], "VGroup",
                                             "Get secant line visualization"),
            "get_derivative_graph": APINode("get_derivative_graph", "method",
                                           ["Axes", "graph"], "ParametricCurve",
                                           "Get derivative graph"),
            "input_to_graph_point": APINode("input_to_graph_point", "method",
                                           ["Axes", "x", "graph"], "point",
                                           "Get point on graph"),
            "c2p": APINode("c2p", "method", ["CoordinateSystem", "x", "y"], "point",
                         "Coordinates to point"),
            "p2c": APINode("p2c", "method", ["CoordinateSystem", "point"], "coords",
                         "Point to coordinates"),

            # Updaters
            "add_updater": APINode("add_updater", "method", ["Mobject", "function"],
                                  "Mobject", "Add update function"),
            "remove_updater": APINode("remove_updater", "method", ["Mobject", "function"],
                                     "Mobject", "Remove update function"),
            "clear_updaters": APINode("clear_updaters", "method", ["Mobject"], "Mobject",
                                     "Clear all updaters"),
            "always_redraw": APINode("always_redraw", "method", ["function"], "Mobject",
                                    "Create auto-updating mobject"),

            # Target/copy
            "copy": APINode("copy", "method", ["Mobject"], "Mobject", "Copy mobject"),
            "generate_target": APINode("generate_target", "method", ["Mobject"], "Mobject",
                                      "Generate animation target"),
            "become": APINode("become", "method", ["Mobject", "Mobject"], "Mobject",
                            "Become another mobject"),
            "save_state": APINode("save_state", "method", ["Mobject"], "Mobject",
                                "Save current state"),
            "restore": APINode("restore", "method", ["Mobject"], "Mobject",
                             "Restore saved state"),

            # ValueTracker methods
            "get_value": APINode("get_value", "method", ["ValueTracker"], "float",
                               "Get current value"),
            "set_value": APINode("set_value", "method", ["ValueTracker", "value"],
                               "ValueTracker", "Set value"),
            "increment_value": APINode("increment_value", "method", ["ValueTracker", "delta"],
                                      "ValueTracker", "Increment value"),
        }

        # Add all nodes to graph
        for name, node in {**animations, **mobjects, **methods}.items():
            self._nodes[name] = node
            self._graph.add_node(name, **{
                "category": node.category,
                "input_types": node.input_types,
                "return_type": node.return_type,
                "description": node.description,
            })

        # === BUILD STATIC EDGES ===

        # Type-based edges (returns -> accepts)
        self._add_type_based_edges()

        # Common pattern edges (3b1b style)
        self._add_common_pattern_edges()

    def _add_type_based_edges(self) -> None:
        """Add edges based on return type -> input type matching."""
        for source_name, source_node in self._nodes.items():
            if not source_node.return_type:
                continue

            for target_name, target_node in self._nodes.items():
                if source_name == target_name:
                    continue

                # Check if source return type matches target input types
                for input_type in target_node.input_types:
                    if self._types_compatible(source_node.return_type, input_type):
                        self._graph.add_edge(
                            source_name, target_name,
                            relation=RelationType.RETURNS.value,
                            weight=0.5,
                        )
                        break

    def _types_compatible(self, return_type: str, input_type: str) -> bool:
        """Check if return type is compatible with input type."""
        # Direct match
        if return_type == input_type:
            return True

        # Inheritance-like relationships
        type_hierarchy = {
            "VMobject": ["Mobject"],
            "VGroup": ["VMobject", "Mobject"],
            "Group": ["Mobject"],
            "ParametricCurve": ["VMobject", "Mobject"],
            "CoordinateSystem": ["VMobject", "Mobject"],
            "Surface": ["Mobject"],
            "AnimationGroup": ["Animation"],
        }

        if return_type in type_hierarchy:
            if input_type in type_hierarchy[return_type]:
                return True

        # Generic matches
        if input_type == "Mobject" and return_type.endswith("Mobject"):
            return True
        if input_type == "VMobject" and return_type == "VGroup":
            return True

        return False

    def _add_common_pattern_edges(self) -> None:
        """Add edges for common 3b1b animation patterns."""
        # These patterns are derived from analyzing 3b1b videos

        # Create/Write -> Wait -> Transform sequence
        common_sequences = [
            # Setup patterns
            ("Circle", "Create", RelationType.COMMONLY_FOLLOWS, 0.9),
            ("Square", "Create", RelationType.COMMONLY_FOLLOWS, 0.9),
            ("MathTex", "Write", RelationType.COMMONLY_FOLLOWS, 0.95),
            ("Tex", "Write", RelationType.COMMONLY_FOLLOWS, 0.95),
            ("Text", "Write", RelationType.COMMONLY_FOLLOWS, 0.9),
            ("Axes", "Create", RelationType.COMMONLY_FOLLOWS, 0.8),
            ("NumberPlane", "Create", RelationType.COMMONLY_FOLLOWS, 0.8),

            # Animation sequences
            ("Write", "Transform", RelationType.COMMONLY_FOLLOWS, 0.7),
            ("Create", "Transform", RelationType.COMMONLY_FOLLOWS, 0.7),
            ("FadeIn", "Transform", RelationType.COMMONLY_FOLLOWS, 0.6),
            ("Transform", "FadeOut", RelationType.COMMONLY_FOLLOWS, 0.5),
            ("Create", "FadeOut", RelationType.COMMONLY_FOLLOWS, 0.4),

            # Indication after creation
            ("Write", "Indicate", RelationType.COMMONLY_FOLLOWS, 0.5),
            ("Create", "Indicate", RelationType.COMMONLY_FOLLOWS, 0.5),
            ("FadeIn", "Indicate", RelationType.COMMONLY_FOLLOWS, 0.4),

            # Transform matching for equations
            ("MathTex", "TransformMatchingTex", RelationType.COMMONLY_PAIRS, 0.9),
            ("TransformMatchingTex", "TransformMatchingTex", RelationType.COMMONLY_FOLLOWS, 0.7),

            # Graph plotting patterns
            ("Axes", "plot", RelationType.COMMONLY_FOLLOWS, 0.95),
            ("Axes", "get_graph", RelationType.COMMONLY_FOLLOWS, 0.95),
            ("plot", "Create", RelationType.COMMONLY_FOLLOWS, 0.9),
            ("get_graph", "Create", RelationType.COMMONLY_FOLLOWS, 0.9),
            ("get_graph", "get_area", RelationType.COMMONLY_FOLLOWS, 0.7),
            ("get_graph", "get_riemann_rectangles", RelationType.COMMONLY_FOLLOWS, 0.8),
            ("get_graph", "get_derivative_graph", RelationType.COMMONLY_FOLLOWS, 0.6),

            # ValueTracker patterns
            ("ValueTracker", "add_updater", RelationType.COMMONLY_FOLLOWS, 0.95),
            ("ValueTracker", "always_redraw", RelationType.COMMONLY_PAIRS, 0.9),
            ("add_updater", "set_value", RelationType.COMMONLY_FOLLOWS, 0.8),

            # Positioning patterns
            ("next_to", "Write", RelationType.COMMONLY_FOLLOWS, 0.6),
            ("next_to", "FadeIn", RelationType.COMMONLY_FOLLOWS, 0.6),
            ("move_to", "Create", RelationType.COMMONLY_FOLLOWS, 0.5),

            # VGroup patterns
            ("VGroup", "arrange", RelationType.COMMONLY_FOLLOWS, 0.9),
            ("VGroup", "Create", RelationType.COMMONLY_FOLLOWS, 0.7),
            ("VGroup", "LaggedStart", RelationType.COMMONLY_PAIRS, 0.8),
            ("VGroup", "LaggedStartMap", RelationType.COMMONLY_PAIRS, 0.8),

            # TracedPath patterns
            ("TracedPath", "add_updater", RelationType.COMMONLY_PAIRS, 0.9),
            ("TracedPath", "ValueTracker", RelationType.COMMONLY_PAIRS, 0.85),

            # Brace patterns
            ("Brace", "next_to", RelationType.COMMONLY_FOLLOWS, 0.8),
            ("Brace", "Write", RelationType.COMMONLY_FOLLOWS, 0.7),
            ("Brace", "FadeIn", RelationType.COMMONLY_FOLLOWS, 0.6),

            # Dot patterns (3b1b loves dots on curves)
            ("Dot", "add_updater", RelationType.COMMONLY_FOLLOWS, 0.8),
            ("Dot", "move_to", RelationType.COMMONLY_FOLLOWS, 0.9),
            ("get_graph", "Dot", RelationType.COMMONLY_PAIRS, 0.85),

            # Target/animation patterns
            ("generate_target", "MoveToTarget", RelationType.COMMONLY_FOLLOWS, 0.95),
            ("save_state", "restore", RelationType.COMMONLY_PAIRS, 0.9),
            ("copy", "Transform", RelationType.COMMONLY_FOLLOWS, 0.7),

            # Succession patterns
            ("LaggedStart", "FadeIn", RelationType.COMMONLY_PAIRS, 0.85),
            ("LaggedStartMap", "FadeIn", RelationType.COMMONLY_PAIRS, 0.85),
            ("LaggedStartMap", "Create", RelationType.COMMONLY_PAIRS, 0.85),
            ("Succession", "Transform", RelationType.COMMONLY_PAIRS, 0.7),
        ]

        for source, target, relation, weight in common_sequences:
            if source in self._nodes and target in self._nodes:
                self._graph.add_edge(
                    source, target,
                    relation=relation.value,
                    weight=weight,
                )

    async def _analyze_3b1b_scenes(self, rag: ChromaDBService) -> None:
        """Analyze indexed 3b1b scenes to extract method call sequences."""
        logger.info("Analyzing 3b1b scenes for API patterns...")

        # Get all indexed scenes
        try:
            results = await rag.search_similar_scenes(
                query="manim animation scene",
                n_results=100,
                prioritize_3b1b=True,
            )
        except Exception as e:
            logger.warning("Failed to search 3b1b scenes: %s", e)
            return

        if not results:
            logger.info("No 3b1b scenes found for analysis")
            return

        # Extract method sequences from each scene
        method_sequences = []
        for result in results:
            code = result.get("content", "")
            if "manim_imports_ext" in code or "OldTex" in code:
                sequences = self._extract_method_sequences(code)
                method_sequences.extend(sequences)

        logger.info("Extracted %d method sequences from %d 3b1b scenes",
                   len(method_sequences), len(results))

        # Count co-occurrences to strengthen edges
        co_occurrence = defaultdict(lambda: defaultdict(int))
        for seq in method_sequences:
            for i, method in enumerate(seq):
                # Count what follows this method
                if i + 1 < len(seq):
                    next_method = seq[i + 1]
                    co_occurrence[method][next_method] += 1

        # Update graph edges with observed frequencies
        for source, targets in co_occurrence.items():
            for target, count in targets.items():
                if source in self._nodes and target in self._nodes:
                    # Add or update edge
                    if self._graph.has_edge(source, target):
                        # Strengthen existing edge
                        current_weight = self._graph[source][target].get("weight", 0.5)
                        new_weight = min(1.0, current_weight + count * 0.05)
                        self._graph[source][target]["weight"] = new_weight
                    else:
                        # Add new edge from observation
                        self._graph.add_edge(
                            source, target,
                            relation=RelationType.COMMONLY_FOLLOWS.value,
                            weight=min(0.9, 0.3 + count * 0.1),
                        )

        logger.info("Updated graph with 3b1b scene analysis")

    def _extract_method_sequences(self, code: str) -> list[list[str]]:
        """Extract method call sequences from Manim code."""
        sequences = []
        current_sequence = []

        # Pattern to match method calls and class instantiations
        patterns = [
            r"self\.play\s*\(\s*(\w+)\s*\(",  # self.play(Animation(...
            r"(\w+)\s*\(",  # ClassName( or method(
            r"\.(\w+)\s*\(",  # .method(
        ]

        # Known API names to track
        known_apis = set(self._nodes.keys())

        for line in code.split('\n'):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = []
                continue

            # Extract method calls from line
            for pattern in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if match in known_apis:
                        current_sequence.append(match)

        if current_sequence:
            sequences.append(current_sequence)

        return sequences

    # === PUBLIC QUERY METHODS ===

    def get_valid_next_methods(
        self,
        current_method: str,
        limit: int = 10,
        relation_filter: RelationType | None = None,
    ) -> list[tuple[str, float, str]]:
        """Get methods that can validly follow the current method.

        Args:
            current_method: The method just used
            limit: Maximum number of results
            relation_filter: Only return edges of this type

        Returns:
            List of (method_name, weight, relation_type) tuples, sorted by weight
        """
        if not self.available or current_method not in self._graph:
            return []

        results = []
        for successor in self._graph.successors(current_method):
            edge_data = self._graph[current_method][successor]
            relation = edge_data.get("relation", RelationType.RETURNS.value)
            weight = edge_data.get("weight", 0.5)

            if relation_filter and relation != relation_filter.value:
                continue

            results.append((successor, weight, relation))

        # Sort by weight descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_method_chain(
        self,
        start: str,
        end: str,
        max_length: int = 5,
    ) -> list[list[str]] | None:
        """Find paths between two methods.

        Args:
            start: Starting method
            end: Target method
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of method names), or None if no path
        """
        if not self.available:
            return None

        if start not in self._graph or end not in self._graph:
            return None

        import networkx as nx

        try:
            # Get all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self._graph, start, end, cutoff=max_length
            ))

            if not paths:
                return None

            # Sort by path length and total weight
            def path_score(path: list[str]) -> float:
                total_weight = 0.0
                for i in range(len(path) - 1):
                    edge_data = self._graph[path[i]][path[i+1]]
                    total_weight += edge_data.get("weight", 0.5)
                # Prefer shorter paths with higher weights
                return total_weight / len(path)

            paths.sort(key=path_score, reverse=True)
            return paths[:5]  # Return top 5 paths

        except nx.NetworkXNoPath:
            return None

    def get_commonly_paired(
        self,
        method: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Get methods commonly used together with this one.

        Args:
            method: The method to find pairings for
            limit: Maximum number of results

        Returns:
            List of (method_name, weight) tuples
        """
        if not self.available or method not in self._graph:
            return []

        paired = []

        # Check outgoing edges for COMMONLY_PAIRS
        for successor in self._graph.successors(method):
            edge_data = self._graph[method][successor]
            if edge_data.get("relation") == RelationType.COMMONLY_PAIRS.value:
                paired.append((successor, edge_data.get("weight", 0.5)))

        # Also check incoming edges (pairing is bidirectional)
        for predecessor in self._graph.predecessors(method):
            edge_data = self._graph[predecessor][method]
            if edge_data.get("relation") == RelationType.COMMONLY_PAIRS.value:
                if predecessor not in [p[0] for p in paired]:
                    paired.append((predecessor, edge_data.get("weight", 0.5)))

        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:limit]

    def get_method_info(self, method: str) -> APINode | None:
        """Get information about a specific method.

        Args:
            method: Method name

        Returns:
            APINode with method information, or None
        """
        return self._nodes.get(method)

    def get_methods_by_category(self, category: str) -> list[str]:
        """Get all methods in a category.

        Args:
            category: One of 'animation', 'mobject', 'method'

        Returns:
            List of method names in that category
        """
        return [
            name for name, node in self._nodes.items()
            if node.category == category
        ]

    def suggest_animation_sequence(
        self,
        start_mobject: str,
        goal: str = "transform",
        max_steps: int = 4,
    ) -> list[tuple[str, str]]:
        """Suggest an animation sequence for a given goal.

        Args:
            start_mobject: The initial mobject type (e.g., "Circle", "MathTex")
            goal: The animation goal ("transform", "indicate", "fadeout", etc.)
            max_steps: Maximum sequence length

        Returns:
            List of (method, description) tuples for suggested sequence
        """
        if not self.available:
            return []

        # Map goals to target animations
        goal_targets = {
            "transform": ["Transform", "ReplacementTransform", "TransformMatchingTex"],
            "indicate": ["Indicate", "Flash", "Circumscribe", "Wiggle"],
            "fadeout": ["FadeOut", "Uncreate"],
            "create": ["Create", "Write", "FadeIn", "DrawBorderThenFill"],
        }

        targets = goal_targets.get(goal, ["Transform"])

        sequence = []

        # Start with creation animation for the mobject
        if start_mobject in self._nodes:
            creation_anims = self.get_valid_next_methods(
                start_mobject,
                limit=3,
                relation_filter=RelationType.COMMONLY_FOLLOWS,
            )
            if creation_anims:
                best_create = creation_anims[0]
                node = self._nodes.get(best_create[0])
                desc = node.description if node else ""
                sequence.append((best_create[0], desc))

        # Add intermediate steps if needed
        if len(sequence) < max_steps - 1:
            # Find path to target
            for target in targets:
                if sequence:
                    last_method = sequence[-1][0]
                else:
                    last_method = start_mobject

                paths = self.get_method_chain(last_method, target, max_steps - len(sequence))
                if paths:
                    for method in paths[0][1:]:  # Skip first (already in sequence)
                        node = self._nodes.get(method)
                        desc = node.description if node else ""
                        sequence.append((method, desc))
                        if len(sequence) >= max_steps:
                            break
                    break

        return sequence[:max_steps]

    def get_graph_stats(self) -> dict:
        """Get statistics about the API graph.

        Returns:
            Dictionary with graph statistics
        """
        if not self.available:
            return {"available": False}

        import networkx as nx

        return {
            "available": True,
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "animations": len(self.get_methods_by_category("animation")),
            "mobjects": len(self.get_methods_by_category("mobject")),
            "methods": len(self.get_methods_by_category("method")),
            "density": nx.density(self._graph),
            "avg_out_degree": sum(d for _, d in self._graph.out_degree()) / self._graph.number_of_nodes(),
        }


# Module-level singleton for easy access
_api_graph: ManimAPIGraph | None = None


async def get_api_graph(rag: ChromaDBService | None = None) -> ManimAPIGraph:
    """Get or create the singleton API graph instance.

    Args:
        rag: Optional ChromaDB service for 3b1b scene analysis

    Returns:
        Initialized ManimAPIGraph instance
    """
    global _api_graph

    if _api_graph is None:
        _api_graph = ManimAPIGraph()
        await _api_graph.initialize(rag)

    return _api_graph
