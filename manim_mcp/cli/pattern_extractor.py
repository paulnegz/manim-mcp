"""Pattern extractor for 3b1b animation techniques.

Extracts reusable animation patterns from indexed scenes and stores them
in a structured format for better code generation.
"""

from __future__ import annotations

import ast
import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnimationPattern:
    """A reusable animation pattern extracted from 3b1b code."""

    name: str
    category: str  # "transform", "progression", "updater", "camera", "sequence"
    description: str
    code_template: str
    math_concepts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source_scene: str | None = None
    source_file: str | None = None

    def to_document(self) -> str:
        """Convert to document for RAG indexing."""
        parts = [
            f"# Animation Pattern: {self.name}",
            f"Category: {self.category}",
            f"",
            f"## Description",
            self.description,
            f"",
            f"## Code Template",
            "```python",
            self.code_template,
            "```",
        ]

        if self.math_concepts:
            parts.extend([
                "",
                f"## Math Concepts",
                ", ".join(self.math_concepts),
            ])

        if self.keywords:
            parts.extend([
                "",
                f"## Keywords",
                ", ".join(self.keywords),
            ])

        return "\n".join(parts)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "id": f"pattern:{self.category}:{self.name}",
            "name": self.name,
            "category": self.category,
            "math_concepts": ",".join(self.math_concepts),
            "keywords": ",".join(self.keywords),
            "source_scene": self.source_scene or "",
            "source_file": self.source_file or "",
        }


# Pre-defined animation patterns extracted from 3b1b code analysis
ANIMATION_PATTERNS = [
    # === PROGRESSION PATTERNS ===
    AnimationPattern(
        name="riemann_sum_convergence",
        category="progression",
        description="""
Show Riemann sum rectangles converging to the exact integral by progressively
increasing the number of rectangles. This is the signature 3b1b style for
integral approximation animations.

Key technique: Pre-generate all rectangle sets, then Transform between them.
""",
        code_template="""
# Pre-generate rectangles with decreasing dx (more rectangles)
all_rects = VGroup(*(
    axes.get_riemann_rectangles(
        graph,
        x_range=[a, b],
        dx=(b-a)/(2**n),  # 4, 8, 16, 32, 64... rectangles
        colors=(BLUE, GREEN),
        fill_opacity=0.7,
    )
    for n in range(2, 7)
))

# Show first set
rects = all_rects[0].copy()
self.play(ShowCreation(rects))
self.wait()

# Transform through increasing rectangle counts
for new_rects in all_rects[1:]:
    self.play(Transform(rects, new_rects))
    self.wait(0.5)

# Optional: Show exact area at the end
exact_area = axes.get_area_under_graph(graph, x_range=[a, b], fill_color=BLUE, fill_opacity=0.5)
self.play(FadeOut(rects), FadeIn(exact_area))
""",
        math_concepts=["integral", "riemann sum", "area under curve", "limit", "approximation"],
        keywords=["get_riemann_rectangles", "Transform", "convergence", "dx", "rectangles"],
        source_scene="IntegralError",
        source_file="_2022/visual_proofs/lies.py",
    ),

    AnimationPattern(
        name="series_partial_sums",
        category="progression",
        description="""
Animate a series by showing partial sums with braces underneath.
Each step adds the next term and shows the running total converging.
""",
        code_template="""
# Create series terms
terms = VGroup(*[
    Tex(f"\\\\frac{{1}}{{{n}}}") for n in range(1, max_terms+1)
])
plus_signs = VGroup(*[Tex("+") for _ in range(len(terms)-1)])
series = VGroup(*[item for pair in zip(terms, plus_signs) for item in pair] + [terms[-1]])
series.arrange(RIGHT)

# Calculate partial sums
partial_sums = np.cumsum([1/n for n in range(1, max_terms+1)])

# Animate with braces showing partial sums
brace = Brace(terms[0], DOWN)
sum_label = brace.get_tex(f"{partial_sums[0]:.4f}")

self.play(FadeIn(terms[0]), GrowFromCenter(brace), Write(sum_label))

for i in range(1, len(terms)):
    new_brace = Brace(VGroup(*terms[:i+1]), DOWN)
    new_label = new_brace.get_tex(f"{partial_sums[i]:.4f}")
    self.play(
        FadeIn(plus_signs[i-1]), FadeIn(terms[i]),
        Transform(brace, new_brace),
        Transform(sum_label, new_label),
    )
    self.wait(0.3)
""",
        math_concepts=["series", "convergence", "partial sum", "infinite series"],
        keywords=["Brace", "partial_sums", "cumsum", "convergence"],
        source_scene="ConvergenceExample",
        source_file="_2017/eoc/chapter10.py",
    ),

    # === TRANSFORM PATTERNS ===
    AnimationPattern(
        name="equation_transformation",
        category="transform",
        description="""
Transform one equation into another, showing algebraic manipulation.
Use ReplacementTransform or TransformMatchingTex for smooth transitions.
""",
        code_template="""
# Original equation
eq1 = Tex("f(x)", "=", "g(x)")
eq1.to_edge(UP)

# Transformed equation
eq2 = Tex("f(x)", "-", "g(x)", "=", "0")
eq2.move_to(eq1, LEFT)

self.play(Write(eq1))
self.wait()

# Transform matching parts, create new parts
self.play(
    ReplacementTransform(eq1[0], eq2[0]),  # f(x) -> f(x)
    ReplacementTransform(eq1[1], eq2[3]),  # = -> =
    ReplacementTransform(eq1[2], eq2[2]),  # g(x) -> g(x)
    FadeIn(eq2[1]),  # - appears
    FadeIn(eq2[4]),  # 0 appears
)
self.wait()
""",
        math_concepts=["algebra", "equation", "manipulation"],
        keywords=["ReplacementTransform", "TransformMatchingTex", "equation"],
        source_scene="RewriteEquation",
        source_file="_2018/WindingNumber.py",
    ),

    AnimationPattern(
        name="morph_graph_to_graph",
        category="transform",
        description="""
Transform one graph into another smoothly. Useful for showing
function transformations or comparing functions.
""",
        code_template="""
# Create two graphs
graph1 = axes.get_graph(lambda x: x**2, color=BLUE)
graph2 = axes.get_graph(lambda x: np.sin(x), color=GREEN)

# Show first graph being drawn
self.play(ShowCreation(graph1))
self.wait()

# Transform to second graph
self.play(Transform(graph1, graph2), run_time=2)
self.wait()
""",
        math_concepts=["function", "transformation", "graph"],
        keywords=["Transform", "get_graph", "morph"],
    ),

    # === UPDATER PATTERNS ===
    AnimationPattern(
        name="value_tracker_animation",
        category="updater",
        description="""
Use ValueTracker with updaters for smooth parameter-based animations.
Objects update automatically as the tracker value changes.
""",
        code_template="""
# Create tracker for parameter
param = ValueTracker(0)

# Create objects that depend on parameter
dot = Dot()
dot.add_updater(lambda d: d.move_to(axes.c2p(param.get_value(), func(param.get_value()))))

label = DecimalNumber()
label.add_updater(lambda l: l.set_value(param.get_value()))
label.add_updater(lambda l: l.next_to(dot, UP))

self.add(dot, label)

# Animate by changing tracker value
self.play(param.animate.set_value(5), run_time=3)
self.wait()
self.play(param.animate.set_value(2), run_time=2)
""",
        math_concepts=["parameter", "continuous", "animation"],
        keywords=["ValueTracker", "add_updater", "get_value", "animate"],
        source_scene="TwoValuesEvenlySpaceAroundZero",
        source_file="_2021/quick_eigen.py",
    ),

    AnimationPattern(
        name="always_redraw_dynamic",
        category="updater",
        description="""
Use always_redraw for objects that need to be completely redrawn
based on other changing objects. More powerful than simple updaters.
""",
        code_template="""
# Moving point
point = Dot()
point_tracker = ValueTracker(0)
point.add_updater(lambda p: p.move_to(curve.pfp(point_tracker.get_value())))

# Tangent line that redraws completely
def get_tangent_line():
    t = point_tracker.get_value()
    p = curve.pfp(t)
    # Calculate tangent direction
    tangent = curve.pfp(t + 0.01) - curve.pfp(t - 0.01)
    tangent = tangent / np.linalg.norm(tangent)
    line = Line(p - tangent, p + tangent, color=YELLOW)
    return line

tangent = always_redraw(get_tangent_line)

self.add(curve, point, tangent)
self.play(point_tracker.animate.set_value(1), run_time=5)
""",
        math_concepts=["tangent", "derivative", "dynamic"],
        keywords=["always_redraw", "pfp", "tangent"],
        source_scene="PortionOfRadialLineInTriangle",
        source_file="_2021/bertrands_paradox.py",
    ),

    # === SEQUENCE PATTERNS ===
    AnimationPattern(
        name="lagged_start_reveal",
        category="sequence",
        description="""
Reveal multiple objects in a staggered sequence using LaggedStart.
Creates a flowing, professional animation feel.
""",
        code_template="""
# Create group of objects
objects = VGroup(*[
    Square().shift(i * RIGHT) for i in range(5)
])

# Staggered reveal
self.play(LaggedStart(*[
    FadeIn(obj, shift=UP) for obj in objects
], lag_ratio=0.2))

# Or use LaggedStartMap for cleaner syntax
self.play(LaggedStartMap(ShowCreation, objects, lag_ratio=0.3))

# Staggered fade out
self.play(LaggedStartMap(FadeOut, objects, lag_ratio=0.1))
""",
        math_concepts=["sequence", "reveal", "multiple objects"],
        keywords=["LaggedStart", "LaggedStartMap", "lag_ratio", "stagger"],
        source_scene="BothPositiveNumbers",
        source_file="_2025/laplace/prequel_equations.py",
    ),

    AnimationPattern(
        name="build_up_construction",
        category="sequence",
        description="""
Build up a complex construction step by step, showing each component
being added. Common in geometry proofs and explanations.
""",
        code_template="""
# Step 1: Draw axes
axes = Axes(x_range=[-3, 3], y_range=[-2, 2])
self.play(ShowCreation(axes))
self.wait(0.5)

# Step 2: Draw graph
graph = axes.get_graph(lambda x: x**2, color=BLUE)
self.play(ShowCreation(graph), run_time=2)
self.wait(0.5)

# Step 3: Add point on graph
x_val = 1
dot = Dot(axes.c2p(x_val, x_val**2), color=YELLOW)
self.play(FadeIn(dot, scale=0.5))

# Step 4: Add tangent line
tangent = axes.get_graph(lambda x: 2*x_val*(x - x_val) + x_val**2, color=GREEN)
self.play(ShowCreation(tangent))

# Step 5: Add label
label = Tex("slope = 2x").next_to(tangent, UP)
self.play(Write(label))
""",
        math_concepts=["construction", "step-by-step", "geometry"],
        keywords=["ShowCreation", "FadeIn", "Write", "sequential"],
    ),

    # === CAMERA PATTERNS ===
    AnimationPattern(
        name="zoom_to_detail",
        category="camera",
        description="""
Zoom the camera into a specific region to show detail,
then zoom back out.
""",
        code_template="""
# Save original camera state
frame = self.camera.frame

# Zoom in to region of interest
target_point = axes.c2p(2, 4)
self.play(
    frame.animate.set_height(3).move_to(target_point),
    run_time=2
)
self.wait()

# Do something at zoomed level
# ...

# Zoom back out
self.play(frame.animate.set_height(FRAME_HEIGHT).move_to(ORIGIN), run_time=2)
""",
        math_concepts=["zoom", "detail", "focus"],
        keywords=["camera", "frame", "set_height", "move_to", "zoom"],
        source_scene="ShowJacobianZoomedIn",
        source_file="_2018/alt_calc.py",
    ),

    # === HIGHLIGHT PATTERNS ===
    AnimationPattern(
        name="indicate_with_flash",
        category="highlight",
        description="""
Draw attention to an object using Flash, Indicate, or
SurroundingRectangle animations.
""",
        code_template="""
# Flash effect
self.play(Flash(important_object, color=YELLOW, run_time=1.5))

# Indicate (pulse/wiggle)
self.play(Indicate(important_object, color=RED))

# Surrounding rectangle
rect = SurroundingRectangle(important_object, color=YELLOW, buff=0.1)
self.play(ShowCreation(rect))
self.wait()
self.play(FadeOut(rect))

# Circumscribe (draw around)
self.play(Circumscribe(important_object, color=BLUE))
""",
        math_concepts=["emphasis", "highlight", "attention"],
        keywords=["Flash", "Indicate", "SurroundingRectangle", "Circumscribe"],
    ),

    # === DERIVATIVE/CALCULUS PATTERNS ===
    AnimationPattern(
        name="tangent_line_animation",
        category="calculus",
        description="""
Animate a tangent line moving along a curve, showing the derivative
at each point. Classic calculus visualization.
""",
        code_template="""
# Setup
axes = Axes(x_range=[-3, 3], y_range=[-2, 5])
func = lambda x: x**2
graph = axes.get_graph(func, color=BLUE)

# Tracker for x position
x_tracker = ValueTracker(-2)

# Dot on curve
dot = always_redraw(lambda: Dot(
    axes.c2p(x_tracker.get_value(), func(x_tracker.get_value())),
    color=YELLOW
))

# Tangent line
def get_tangent():
    x = x_tracker.get_value()
    slope = 2 * x  # derivative of x^2
    y = func(x)
    # Line from x-1 to x+1
    p1 = axes.c2p(x - 1, y - slope)
    p2 = axes.c2p(x + 1, y + slope)
    return Line(p1, p2, color=GREEN)

tangent = always_redraw(get_tangent)

# Slope label
slope_label = always_redraw(lambda: DecimalNumber(
    2 * x_tracker.get_value(),
    num_decimal_places=2,
).next_to(dot, UR))

self.add(axes, graph, dot, tangent, slope_label)

# Animate x moving
self.play(x_tracker.animate.set_value(2), run_time=5)
self.wait()
self.play(x_tracker.animate.set_value(0), run_time=3)
""",
        math_concepts=["derivative", "tangent line", "slope", "calculus"],
        keywords=["tangent", "derivative", "slope", "ValueTracker", "always_redraw"],
    ),

    # === LINEAR ALGEBRA PATTERNS ===
    AnimationPattern(
        name="matrix_transformation",
        category="linear_algebra",
        description="""
Show a 2D linear transformation by applying a matrix to the plane.
Eigenvectors stay on their span, grid lines transform smoothly.
""",
        code_template="""
# Setup plane
plane = NumberPlane()
plane.add_coordinates()

# Basis vectors
i_hat = Vector([1, 0], color=GREEN)
j_hat = Vector([0, 1], color=RED)

# Matrix to apply
matrix = [[2, 1], [1, 2]]

self.add(plane, i_hat, j_hat)
self.wait()

# Apply transformation
self.play(
    plane.animate.apply_matrix(matrix),
    i_hat.animate.apply_matrix(matrix),
    j_hat.animate.apply_matrix(matrix),
    run_time=3
)
self.wait()
""",
        math_concepts=["matrix", "linear transformation", "eigenvector", "linear algebra"],
        keywords=["apply_matrix", "NumberPlane", "Vector", "transformation"],
        source_scene="ShowSquishingAndStretching",
        source_file="_2021/quick_eigen.py",
    ),

    # === MORE GEOMETRY PATTERNS ===
    AnimationPattern(
        name="triangle_construction",
        category="geometry",
        description="""
Construct a triangle step by step, showing vertices, sides, and
optionally angles. Classic geometry construction animation.
""",
        code_template="""
# Define vertices
A = np.array([-2, -1, 0])
B = np.array([2, -1, 0])
C = np.array([0, 2, 0])

# Create triangle
triangle = Polygon(A, B, C, color=WHITE)

# Vertex labels
labels = VGroup(
    Tex("A").next_to(A, DL),
    Tex("B").next_to(B, DR),
    Tex("C").next_to(C, UP),
)

# Dots at vertices
dots = VGroup(*[Dot(p, color=YELLOW) for p in [A, B, C]])

# Animate construction
self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.3))
self.play(Write(labels))
self.play(ShowCreation(triangle), run_time=2)

# Optionally add altitude/median/etc
altitude = Line(C, np.array([0, -1, 0]), color=GREEN)
self.play(ShowCreation(altitude))
""",
        math_concepts=["triangle", "geometry", "construction", "vertices"],
        keywords=["Polygon", "vertices", "triangle", "construction"],
    ),

    AnimationPattern(
        name="circle_theorem",
        category="geometry",
        description="""
Demonstrate a circle theorem (inscribed angle, tangent, etc.)
by showing the construction and highlighting key relationships.
""",
        code_template="""
# Create circle
circle = Circle(radius=2, color=BLUE)
center = Dot(ORIGIN, color=WHITE)
center_label = Tex("O").next_to(center, DR, buff=0.1)

# Points on circle
angle_A = 0.5
angle_B = 2.0
angle_C = 4.0
A = circle.point_at_angle(angle_A)
B = circle.point_at_angle(angle_B)
C = circle.point_at_angle(angle_C)

dots = VGroup(*[Dot(p, color=YELLOW) for p in [A, B, C]])
labels = VGroup(
    Tex("A").next_to(A, normalize(A)),
    Tex("B").next_to(B, normalize(B)),
    Tex("C").next_to(C, normalize(C)),
)

# Inscribed angle and central angle
inscribed = VGroup(Line(A, B), Line(A, C)).set_color(GREEN)
central = VGroup(Line(ORIGIN, B), Line(ORIGIN, C)).set_color(RED)

# Animate
self.play(ShowCreation(circle), FadeIn(center))
self.play(Write(center_label))
self.play(LaggedStartMap(FadeIn, dots), Write(labels))
self.play(ShowCreation(inscribed))
self.wait()
self.play(ShowCreation(central))

# Show angle relationship
arc = Arc(
    start_angle=angle_B, angle=angle_C - angle_B,
    radius=0.5, color=RED
).move_arc_center_to(ORIGIN)
self.play(ShowCreation(arc))
""",
        math_concepts=["circle", "inscribed angle", "central angle", "geometry"],
        keywords=["Circle", "point_at_angle", "Arc", "inscribed"],
    ),

    AnimationPattern(
        name="rotating_shape",
        category="geometry",
        description="""
Rotate a shape around a point or axis, showing symmetry or
transformation properties.
""",
        code_template="""
# Create shape
square = Square(side_length=2, color=BLUE, fill_opacity=0.5)

# Mark center of rotation
center = Dot(ORIGIN, color=YELLOW)
center_label = Tex("O").next_to(center, DR)

# Add reference point to track rotation
ref_point = Dot(square.get_corner(UR), color=RED)

self.play(ShowCreation(square), FadeIn(center, ref_point))
self.play(Write(center_label))

# Rotate 90 degrees
for _ in range(4):
    self.play(
        Rotate(square, PI/2, about_point=ORIGIN),
        Rotate(ref_point, PI/2, about_point=ORIGIN),
        run_time=1
    )
    self.wait(0.3)

# Or use animate for smooth rotation
self.play(Rotate(square, 2*PI, about_point=ORIGIN), run_time=3)
""",
        math_concepts=["rotation", "symmetry", "transformation", "geometry"],
        keywords=["Rotate", "about_point", "symmetry", "rotation"],
    ),

    # === MORE CALCULUS PATTERNS ===
    AnimationPattern(
        name="limit_visualization",
        category="calculus",
        description="""
Visualize a limit by showing values approaching from both sides,
with the function value approaching the limit.
""",
        code_template="""
# Setup
axes = Axes(x_range=[-1, 3], y_range=[-1, 5])
func = lambda x: (x**2 - 1) / (x - 1) if abs(x - 1) > 0.01 else 2

# Graph with hole at x=1
left_graph = axes.get_graph(func, x_range=[-1, 0.99], color=BLUE)
right_graph = axes.get_graph(func, x_range=[1.01, 3], color=BLUE)

# Point where limit exists
limit_point = Dot(axes.c2p(1, 2), color=WHITE, fill_opacity=0)
limit_point.set_stroke(WHITE, width=2)

# Moving dots approaching from both sides
left_tracker = ValueTracker(0)
right_tracker = ValueTracker(2)

left_dot = always_redraw(lambda: Dot(
    axes.c2p(left_tracker.get_value(), func(left_tracker.get_value())),
    color=GREEN
))
right_dot = always_redraw(lambda: Dot(
    axes.c2p(right_tracker.get_value(), func(right_tracker.get_value())),
    color=RED
))

# Limit label
limit_label = Tex(r"\\lim_{x \\to 1} f(x) = 2").to_edge(UP)

self.add(axes, left_graph, right_graph, limit_point)
self.add(left_dot, right_dot)
self.wait()

# Animate approach
self.play(
    left_tracker.animate.set_value(0.99),
    right_tracker.animate.set_value(1.01),
    run_time=3
)
self.play(Write(limit_label))
""",
        math_concepts=["limit", "continuity", "approach", "calculus"],
        keywords=["limit", "ValueTracker", "approach", "discontinuity"],
    ),

    AnimationPattern(
        name="area_accumulation",
        category="calculus",
        description="""
Show area accumulating under a curve as x increases, demonstrating
the integral as accumulated area (Fundamental Theorem of Calculus).
""",
        code_template="""
# Setup
axes = Axes(x_range=[0, 5], y_range=[0, 4])
func = lambda x: 0.2 * x**2
graph = axes.get_graph(func, color=BLUE)

# Tracker for upper bound of integration
x_tracker = ValueTracker(0.5)

# Dynamic shaded area
def get_area():
    return axes.get_area_under_graph(
        graph,
        x_range=[0, x_tracker.get_value()],
        fill_color=BLUE,
        fill_opacity=0.5
    )

area = always_redraw(get_area)

# Running total label
def get_label():
    x = x_tracker.get_value()
    integral_value = (0.2 * x**3) / 3  # Antiderivative
    label = DecimalNumber(integral_value, num_decimal_places=2)
    label.next_to(axes.c2p(x, func(x)/2), RIGHT)
    return label

total_label = always_redraw(get_label)

# Vertical line at x
v_line = always_redraw(lambda: axes.get_v_line_to_graph(
    x_tracker.get_value(), graph, color=YELLOW
))

self.add(axes, graph)
self.play(FadeIn(area, v_line, total_label))

# Animate accumulation
self.play(x_tracker.animate.set_value(4), run_time=5)
self.wait()
""",
        math_concepts=["integral", "area", "accumulation", "FTC", "fundamental theorem"],
        keywords=["get_area_under_graph", "accumulation", "integral", "ValueTracker"],
    ),

    AnimationPattern(
        name="derivative_definition",
        category="calculus",
        description="""
Show the definition of derivative as a limit of secant lines
approaching the tangent line as h -> 0.
""",
        code_template="""
# Setup
axes = Axes(x_range=[-1, 4], y_range=[-1, 5])
func = lambda x: 0.5 * x**2
graph = axes.get_graph(func, color=BLUE)

x0 = 2  # Point of tangency
y0 = func(x0)
base_point = Dot(axes.c2p(x0, y0), color=YELLOW)

# h tracker (distance to second point)
h_tracker = ValueTracker(1.5)

# Secant line
def get_secant():
    h = h_tracker.get_value()
    if abs(h) < 0.01:
        h = 0.01
    x1, x2 = x0, x0 + h
    y1, y2 = func(x1), func(x2)
    slope = (y2 - y1) / h
    # Extend line
    p1 = axes.c2p(x0 - 1, y0 - slope)
    p2 = axes.c2p(x0 + 2, y0 + slope * 2)
    return Line(p1, p2, color=GREEN)

secant = always_redraw(get_secant)

# Second point on curve
second_point = always_redraw(lambda: Dot(
    axes.c2p(x0 + h_tracker.get_value(), func(x0 + h_tracker.get_value())),
    color=RED
))

# h label
h_brace = always_redraw(lambda: BraceBetweenPoints(
    axes.c2p(x0, 0), axes.c2p(x0 + h_tracker.get_value(), 0), UP
))
h_label = always_redraw(lambda: Tex("h").next_to(h_brace, UP))

self.add(axes, graph, base_point, secant, second_point, h_brace, h_label)

# Shrink h towards 0
self.play(h_tracker.animate.set_value(0.1), run_time=4)
self.wait()

# Label as derivative
deriv_label = Tex(r"f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}")
deriv_label.to_edge(UP)
self.play(Write(deriv_label))
""",
        math_concepts=["derivative", "limit", "secant", "tangent", "definition"],
        keywords=["secant", "tangent", "limit", "derivative", "definition"],
    ),

    # === PROBABILITY PATTERNS ===
    AnimationPattern(
        name="probability_distribution",
        category="probability",
        description="""
Visualize a probability distribution with bars and the continuous
PDF overlaid. Show mean, standard deviation markers.
""",
        code_template="""
# Setup axes
axes = Axes(
    x_range=[-4, 4],
    y_range=[0, 0.5],
    x_length=10,
    y_length=4,
)
axes_labels = axes.get_axis_labels(x_label="x", y_label="P(x)")

# Normal distribution PDF
import numpy as np
mu, sigma = 0, 1
pdf = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Discrete bars (histogram approximation)
bar_width = 0.4
bars = VGroup()
for x in np.arange(-3, 3.5, bar_width):
    height = pdf(x + bar_width/2)
    bar = Rectangle(
        width=bar_width * axes.x_length / 8,
        height=height * axes.y_length / 0.5,
        fill_color=BLUE,
        fill_opacity=0.7,
        stroke_width=1
    )
    bar.move_to(axes.c2p(x + bar_width/2, height/2))
    bars.add(bar)

# Smooth PDF curve
pdf_curve = axes.get_graph(pdf, color=YELLOW, x_range=[-3.5, 3.5])

# Mean and std markers
mean_line = DashedLine(axes.c2p(mu, 0), axes.c2p(mu, 0.5), color=RED)
std_lines = VGroup(
    DashedLine(axes.c2p(mu - sigma, 0), axes.c2p(mu - sigma, pdf(mu - sigma)), color=GREEN),
    DashedLine(axes.c2p(mu + sigma, 0), axes.c2p(mu + sigma, pdf(mu + sigma)), color=GREEN),
)

self.play(ShowCreation(axes), Write(axes_labels))
self.play(LaggedStartMap(FadeIn, bars, lag_ratio=0.05))
self.wait()
self.play(ShowCreation(pdf_curve), run_time=2)
self.play(ShowCreation(mean_line))
self.play(ShowCreation(std_lines))
""",
        math_concepts=["probability", "distribution", "normal", "gaussian", "statistics"],
        keywords=["histogram", "pdf", "normal", "distribution", "bars"],
    ),

    AnimationPattern(
        name="random_walk",
        category="probability",
        description="""
Animate a random walk, showing the path building up step by step.
Can be used for Brownian motion or coin flip demonstrations.
""",
        code_template="""
import numpy as np
np.random.seed(42)

# Setup
axes = Axes(x_range=[0, 50], y_range=[-10, 10])

# Generate random walk
n_steps = 50
steps = np.random.choice([-1, 1], size=n_steps)
positions = np.cumsum(steps)
positions = np.insert(positions, 0, 0)  # Start at 0

# Create path
path_points = [axes.c2p(i, positions[i]) for i in range(n_steps + 1)]

# Build path incrementally
current_path = VMobject(color=BLUE)
current_path.set_points_as_corners([path_points[0], path_points[0]])

dot = Dot(path_points[0], color=YELLOW)

self.add(axes, current_path, dot)

# Animate walk
for i in range(1, len(path_points)):
    new_path = VMobject(color=BLUE)
    new_path.set_points_as_corners(path_points[:i+1])
    self.play(
        Transform(current_path, new_path),
        dot.animate.move_to(path_points[i]),
        run_time=0.1
    )

# Show final position
final_label = Tex(f"Final: {positions[-1]}").next_to(dot, UP)
self.play(Write(final_label))
""",
        math_concepts=["random walk", "probability", "stochastic", "Brownian motion"],
        keywords=["random", "walk", "path", "stochastic", "cumsum"],
    ),

    # === COMPLEX NUMBER PATTERNS ===
    AnimationPattern(
        name="complex_plane_multiplication",
        category="complex",
        description="""
Visualize complex number multiplication as rotation and scaling
on the complex plane.
""",
        code_template="""
# Complex plane setup
plane = ComplexPlane(x_range=[-3, 3], y_range=[-3, 3])
plane.add_coordinates()

# Original complex number
z1 = complex(1, 1)  # 1 + i
z1_dot = Dot(plane.n2p(z1), color=BLUE)
z1_label = Tex("z = 1 + i").next_to(z1_dot, UR)
z1_vector = Arrow(plane.n2p(0), plane.n2p(z1), buff=0, color=BLUE)

# Multiplier (causes rotation by 90 degrees)
w = complex(0, 1)  # i

# Result
z2 = z1 * w  # -1 + i
z2_dot = Dot(plane.n2p(z2), color=GREEN)
z2_label = Tex("iz = -1 + i").next_to(z2_dot, UL)
z2_vector = Arrow(plane.n2p(0), plane.n2p(z2), buff=0, color=GREEN)

self.play(ShowCreation(plane))
self.play(ShowCreation(z1_vector), FadeIn(z1_dot), Write(z1_label))
self.wait()

# Show rotation
arc = Arc(
    start_angle=np.angle(z1),
    angle=np.angle(w),
    radius=0.5,
    color=YELLOW
)
self.play(ShowCreation(arc))

# Transform to result
self.play(
    ReplacementTransform(z1_vector.copy(), z2_vector),
    ReplacementTransform(z1_dot.copy(), z2_dot),
    run_time=2
)
self.play(Write(z2_label))
""",
        math_concepts=["complex numbers", "multiplication", "rotation", "complex plane"],
        keywords=["ComplexPlane", "complex", "rotation", "n2p"],
    ),

    AnimationPattern(
        name="euler_formula",
        category="complex",
        description="""
Visualize Euler's formula e^(i*theta) as a point moving on the unit circle,
connecting exponential, trigonometric, and complex representations.
""",
        code_template="""
# Complex plane with unit circle
plane = ComplexPlane(x_range=[-2, 2], y_range=[-2, 2])
circle = Circle(radius=plane.x_length/4, color=BLUE)

# Angle tracker
theta = ValueTracker(0)

# Point on circle
def get_point():
    t = theta.get_value()
    return Dot(plane.n2p(complex(np.cos(t), np.sin(t))), color=YELLOW)

point = always_redraw(get_point)

# Radius line
radius = always_redraw(lambda: Line(
    plane.n2p(0),
    plane.n2p(complex(np.cos(theta.get_value()), np.sin(theta.get_value()))),
    color=GREEN
))

# Projections onto axes
cos_line = always_redraw(lambda: DashedLine(
    plane.n2p(complex(np.cos(theta.get_value()), np.sin(theta.get_value()))),
    plane.n2p(complex(np.cos(theta.get_value()), 0)),
    color=RED
))
sin_line = always_redraw(lambda: DashedLine(
    plane.n2p(complex(np.cos(theta.get_value()), np.sin(theta.get_value()))),
    plane.n2p(complex(0, np.sin(theta.get_value()))),
    color=BLUE_C
))

# Euler's formula label
formula = Tex(r"e^{i\\theta} = \\cos\\theta + i\\sin\\theta")
formula.to_edge(UP)

self.add(plane, circle, radius, point, cos_line, sin_line)
self.play(Write(formula))

# Animate full rotation
self.play(theta.animate.set_value(2 * PI), run_time=6, rate_func=linear)
""",
        math_concepts=["Euler formula", "complex exponential", "unit circle", "trigonometry"],
        keywords=["euler", "e^i", "unit circle", "cos", "sin", "complex"],
    ),

    # === VECTOR PATTERNS ===
    AnimationPattern(
        name="vector_addition",
        category="vectors",
        description="""
Show vector addition using head-to-tail method, with the resultant
vector shown as the diagonal of the parallelogram.
""",
        code_template="""
# Setup
plane = NumberPlane(x_range=[-5, 5], y_range=[-3, 3])

# Vectors
v1 = np.array([3, 1, 0])
v2 = np.array([1, 2, 0])

vec1 = Arrow(ORIGIN, v1, buff=0, color=RED)
vec2 = Arrow(ORIGIN, v2, buff=0, color=BLUE)

vec1_label = Tex(r"\\vec{a}").next_to(vec1, DOWN)
vec2_label = Tex(r"\\vec{b}").next_to(vec2, LEFT)

self.play(ShowCreation(plane))
self.play(ShowCreation(vec1), Write(vec1_label))
self.play(ShowCreation(vec2), Write(vec2_label))

# Move v2 to tip of v1 (head-to-tail)
vec2_shifted = Arrow(v1, v1 + v2, buff=0, color=BLUE)
self.play(
    ReplacementTransform(vec2.copy(), vec2_shifted),
    run_time=1.5
)

# Show resultant
resultant = Arrow(ORIGIN, v1 + v2, buff=0, color=GREEN)
result_label = Tex(r"\\vec{a} + \\vec{b}").next_to(resultant, RIGHT)

self.play(ShowCreation(resultant))
self.play(Write(result_label))

# Optional: show parallelogram
parallelogram = Polygon(
    ORIGIN, v1, v1 + v2, v2,
    fill_color=YELLOW, fill_opacity=0.2, stroke_color=YELLOW
)
self.play(FadeIn(parallelogram))
""",
        math_concepts=["vectors", "addition", "head-to-tail", "parallelogram"],
        keywords=["Arrow", "vector", "addition", "parallelogram"],
    ),

    AnimationPattern(
        name="dot_product_projection",
        category="vectors",
        description="""
Visualize dot product as the projection of one vector onto another,
showing the geometric interpretation.
""",
        code_template="""
# Setup
plane = NumberPlane(x_range=[-4, 4], y_range=[-2, 3])

# Vectors
a = np.array([3, 0, 0])
b = np.array([2, 2, 0])

vec_a = Arrow(ORIGIN, a, buff=0, color=RED)
vec_b = Arrow(ORIGIN, b, buff=0, color=BLUE)

# Labels
a_label = Tex(r"\\vec{a}").next_to(vec_a, DOWN)
b_label = Tex(r"\\vec{b}").next_to(vec_b, UL)

self.play(ShowCreation(plane))
self.play(ShowCreation(vec_a), Write(a_label))
self.play(ShowCreation(vec_b), Write(b_label))

# Projection of b onto a
proj_length = np.dot(b, a) / np.linalg.norm(a)
proj_point = proj_length * a / np.linalg.norm(a)

# Dashed perpendicular line
perp_line = DashedLine(b, proj_point, color=GREEN)
self.play(ShowCreation(perp_line))

# Projection vector
proj_vec = Arrow(ORIGIN, proj_point, buff=0, color=YELLOW)
proj_label = Tex(r"proj_{\\vec{a}}\\vec{b}").next_to(proj_vec, DOWN)

self.play(ShowCreation(proj_vec))
self.play(Write(proj_label))

# Dot product formula
formula = Tex(r"\\vec{a} \\cdot \\vec{b} = |\\vec{a}| |\\vec{b}| \\cos\\theta")
formula.to_edge(UP)
self.play(Write(formula))
""",
        math_concepts=["dot product", "projection", "vectors", "inner product"],
        keywords=["dot product", "projection", "cos", "vectors"],
    ),

    # === TEXT/EQUATION PATTERNS ===
    AnimationPattern(
        name="equation_derivation",
        category="equation",
        description="""
Show a multi-step equation derivation, transforming from one
form to the next with clear intermediate steps.
""",
        code_template="""
# Steps of derivation
steps = [
    Tex(r"(a + b)^2"),
    Tex(r"(a + b)(a + b)"),
    Tex(r"a \\cdot a + a \\cdot b + b \\cdot a + b \\cdot b"),
    Tex(r"a^2 + ab + ba + b^2"),
    Tex(r"a^2 + 2ab + b^2"),
]

# Position all at same location
for step in steps:
    step.move_to(ORIGIN)

# Show first step
self.play(Write(steps[0]))
self.wait()

# Transform through each step
for i in range(len(steps) - 1):
    self.play(TransformMatchingShapes(steps[i], steps[i + 1]))
    self.wait(0.5)

# Highlight final answer
box = SurroundingRectangle(steps[-1], color=YELLOW)
self.play(ShowCreation(box))
""",
        math_concepts=["algebra", "derivation", "proof", "simplification"],
        keywords=["TransformMatchingShapes", "derivation", "equation", "steps"],
    ),

    AnimationPattern(
        name="text_highlight",
        category="equation",
        description="""
Highlight parts of text or equations using color changes,
underlines, or boxes to draw attention.
""",
        code_template="""
# Create equation with parts
equation = Tex(
    "E", "=", "m", "c^2",
    tex_to_color_map={"E": WHITE, "m": WHITE, "c^2": WHITE}
)
equation.scale(2)

self.play(Write(equation))
self.wait()

# Highlight E (energy)
self.play(equation[0].animate.set_color(YELLOW))
energy_label = Tex("Energy").next_to(equation[0], UP)
self.play(FadeIn(energy_label, shift=DOWN))
self.wait(0.5)
self.play(FadeOut(energy_label), equation[0].animate.set_color(WHITE))

# Highlight m (mass)
self.play(equation[2].animate.set_color(RED))
mass_label = Tex("Mass").next_to(equation[2], UP)
self.play(FadeIn(mass_label, shift=DOWN))
self.wait(0.5)
self.play(FadeOut(mass_label), equation[2].animate.set_color(WHITE))

# Highlight c^2 (speed of light squared)
self.play(equation[3].animate.set_color(BLUE))
c_label = Tex("Speed of light$^2$").next_to(equation[3], UP)
self.play(FadeIn(c_label, shift=DOWN))

# Flash the whole equation
self.play(Flash(equation, color=WHITE, run_time=1))
""",
        math_concepts=["highlight", "emphasis", "annotation"],
        keywords=["tex_to_color_map", "set_color", "highlight", "Flash"],
    ),

    # === 3D PATTERNS ===
    AnimationPattern(
        name="surface_plot_3d",
        category="3d",
        description="""
Create and animate a 3D surface plot, showing the function
from multiple angles.
""",
        code_template="""
# This is a ThreeDScene
axes = ThreeDAxes(
    x_range=[-3, 3],
    y_range=[-3, 3],
    z_range=[-1, 5],
)

# Surface function
surface = Surface(
    lambda u, v: axes.c2p(u, v, u**2 + v**2),
    u_range=[-2, 2],
    v_range=[-2, 2],
    resolution=(20, 20),
    fill_opacity=0.7,
    checkerboard_colors=[BLUE_D, BLUE_E],
)

# Initial camera angle
self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)

self.play(ShowCreation(axes))
self.play(ShowCreation(surface), run_time=3)

# Rotate camera around
self.begin_ambient_camera_rotation(rate=0.2)
self.wait(5)
self.stop_ambient_camera_rotation()

# Move camera to top-down view
self.move_camera(phi=0, theta=0, run_time=2)
self.wait()
""",
        math_concepts=["3D", "surface", "multivariable", "calculus"],
        keywords=["ThreeDAxes", "Surface", "camera", "3D", "surface plot"],
    ),

    AnimationPattern(
        name="vector_field_3d",
        category="3d",
        description="""
Animate a 3D vector field showing flow or force directions
at each point in space.
""",
        code_template="""
# This is a ThreeDScene
axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3])

# Vector field function
def field_func(point):
    x, y, z = point
    return np.array([-y, x, 0])  # Rotation around z-axis

# Create vector field
field = ArrowVectorField(
    field_func,
    x_range=[-2, 2, 1],
    y_range=[-2, 2, 1],
    z_range=[-1, 1, 1],
    length_func=lambda norm: 0.3 * norm,
)

self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
self.play(ShowCreation(axes))
self.play(ShowCreation(field), run_time=3)

# Rotate to show from different angles
self.begin_ambient_camera_rotation(rate=0.3)
self.wait(5)
""",
        math_concepts=["vector field", "3D", "flow", "differential equations"],
        keywords=["ArrowVectorField", "3D", "vector field", "flow"],
    ),

    # === WAVE PATTERNS ===
    AnimationPattern(
        name="standing_wave",
        category="waves",
        description="""
Animate a standing wave showing nodes and antinodes,
with the wave oscillating in place.
""",
        code_template="""
# Setup
axes = Axes(x_range=[0, 4*PI], y_range=[-2, 2])

# Time tracker for animation
time = ValueTracker(0)

# Standing wave: sin(x) * cos(t)
def wave_func(x):
    return np.sin(x) * np.cos(time.get_value() * 2)

wave = always_redraw(lambda: axes.get_graph(
    wave_func,
    color=BLUE,
    x_range=[0, 4*PI]
))

# Mark nodes (always zero)
nodes = VGroup(*[
    Dot(axes.c2p(n * PI, 0), color=RED)
    for n in range(5)
])

# Mark antinodes (max amplitude)
antinodes = VGroup(*[
    Dot(axes.c2p((n + 0.5) * PI, 0), color=GREEN)
    for n in range(4)
])

self.add(axes, wave, nodes, antinodes)

# Animate oscillation
self.play(time.animate.set_value(4 * PI), run_time=8, rate_func=linear)
""",
        math_concepts=["standing wave", "nodes", "antinodes", "oscillation"],
        keywords=["wave", "sin", "cos", "oscillation", "standing wave"],
    ),

    AnimationPattern(
        name="traveling_wave",
        category="waves",
        description="""
Animate a wave traveling through space, showing propagation
direction and wavelength.
""",
        code_template="""
# Setup
axes = Axes(x_range=[0, 10], y_range=[-2, 2])

# Time and wave parameters
time = ValueTracker(0)
k = 1  # wave number
omega = 2  # angular frequency

# Traveling wave: sin(kx - omega*t)
def wave_func(x):
    return np.sin(k * x - omega * time.get_value())

wave = always_redraw(lambda: axes.get_graph(
    wave_func,
    color=BLUE,
    x_range=[0, 10]
))

# Label
wave_label = Tex(r"y = \\sin(kx - \\omega t)")
wave_label.to_edge(UP)

# Wavelength marker
wavelength = 2 * PI / k
brace = Brace(
    VGroup(Dot(axes.c2p(0, 0)), Dot(axes.c2p(wavelength, 0))),
    DOWN
)
lambda_label = Tex(r"\\lambda").next_to(brace, DOWN)

self.add(axes, wave, wave_label, brace, lambda_label)

# Animate wave traveling
self.play(time.animate.set_value(4 * PI), run_time=8, rate_func=linear)
""",
        math_concepts=["traveling wave", "wavelength", "propagation", "physics"],
        keywords=["wave", "traveling", "propagation", "sin", "omega"],
    ),

    # === FOURIER PATTERNS ===
    AnimationPattern(
        name="fourier_series_approximation",
        category="fourier",
        description="""
Show a Fourier series approximation of a function, adding
more terms to get closer to the target.
""",
        code_template="""
# Setup
axes = Axes(x_range=[-PI, PI], y_range=[-1.5, 1.5])

# Target function (square wave)
def square_wave(x):
    return 1 if x >= 0 else -1

# Fourier approximations with increasing terms
def fourier_approx(n_terms):
    def f(x):
        result = 0
        for k in range(1, n_terms + 1, 2):  # Odd terms only
            result += (4 / (k * PI)) * np.sin(k * x)
        return result
    return f

# Create approximation graphs
approximations = [
    axes.get_graph(fourier_approx(n), color=BLUE, x_range=[-PI + 0.01, PI - 0.01])
    for n in [1, 3, 5, 9, 15, 31]
]

# Label showing number of terms
n_label = VGroup(Tex("n = "), Integer(1)).arrange(RIGHT)
n_label.to_edge(UP)

self.play(ShowCreation(axes))

# Show first approximation
current = approximations[0]
self.play(ShowCreation(current), Write(n_label))

# Transform through better approximations
for i, (approx, n) in enumerate(zip(approximations[1:], [3, 5, 9, 15, 31])):
    new_label = VGroup(Tex("n = "), Integer(n)).arrange(RIGHT)
    new_label.to_edge(UP)
    self.play(
        Transform(current, approx),
        Transform(n_label, new_label),
        run_time=1.5
    )
    self.wait(0.5)
""",
        math_concepts=["Fourier series", "approximation", "harmonics", "square wave"],
        keywords=["Fourier", "series", "approximation", "sin", "harmonics"],
    ),

    # === GRAPH THEORY PATTERNS ===
    AnimationPattern(
        name="graph_traversal",
        category="graph",
        description="""
Animate traversing a graph (BFS/DFS style), highlighting
nodes and edges as they're visited.
""",
        code_template="""
# Create graph vertices
vertices = {
    "A": np.array([-2, 1, 0]),
    "B": np.array([0, 2, 0]),
    "C": np.array([2, 1, 0]),
    "D": np.array([-1, -1, 0]),
    "E": np.array([1, -1, 0]),
}

# Create nodes
nodes = VGroup(*[
    VGroup(
        Circle(radius=0.3, color=WHITE, fill_opacity=0.5),
        Tex(name)
    ).move_to(pos)
    for name, pos in vertices.items()
])

# Edges
edges_list = [("A", "B"), ("B", "C"), ("A", "D"), ("D", "E"), ("C", "E")]
edges = VGroup(*[
    Line(vertices[a], vertices[b], color=GRAY)
    for a, b in edges_list
])

self.play(ShowCreation(edges), FadeIn(nodes))
self.wait()

# Simulate BFS from A
visited_order = ["A", "B", "D", "C", "E"]
visited_edges = [("A", "B"), ("A", "D"), ("B", "C"), ("D", "E")]

for i, node_name in enumerate(visited_order):
    # Find the node
    node_idx = list(vertices.keys()).index(node_name)
    node = nodes[node_idx]

    # Highlight node
    self.play(
        node[0].animate.set_fill(GREEN, opacity=0.8),
        run_time=0.5
    )

    # Highlight edge used to reach it (if not first)
    if i > 0:
        edge_used = visited_edges[i - 1]
        edge_idx = edges_list.index(edge_used) if edge_used in edges_list else edges_list.index((edge_used[1], edge_used[0]))
        self.play(edges[edge_idx].animate.set_color(GREEN), run_time=0.3)
""",
        math_concepts=["graph", "traversal", "BFS", "DFS", "tree"],
        keywords=["graph", "traversal", "nodes", "edges", "BFS"],
    ),

    # === PHYSICS PATTERNS ===
    AnimationPattern(
        name="pendulum_motion",
        category="physics",
        description="""
Animate a simple pendulum swinging, with the bob tracing
its path and showing velocity/acceleration vectors.
""",
        code_template="""
# Pendulum parameters
pivot = np.array([0, 2, 0])
length = 2.5
theta_max = PI / 4  # 45 degrees max swing

# Angle tracker (oscillating)
time = ValueTracker(0)
omega = np.sqrt(9.8 / length)  # Natural frequency

def get_theta():
    return theta_max * np.cos(omega * time.get_value())

# Bob position
def get_bob_pos():
    theta = get_theta()
    return pivot + length * np.array([np.sin(theta), -np.cos(theta), 0])

# Pendulum components
bob = always_redraw(lambda: Dot(get_bob_pos(), color=BLUE, radius=0.15))
string = always_redraw(lambda: Line(pivot, get_bob_pos(), color=WHITE))
pivot_dot = Dot(pivot, color=GRAY)

# Trace path
trace = TracedPath(
    lambda: get_bob_pos(),
    stroke_color=YELLOW,
    stroke_opacity=0.5,
    stroke_width=2,
)

self.add(pivot_dot, string, bob, trace)

# Animate pendulum swinging
self.play(time.animate.set_value(6 * PI / omega), run_time=10, rate_func=linear)
""",
        math_concepts=["pendulum", "harmonic motion", "oscillation", "physics"],
        keywords=["pendulum", "oscillation", "physics", "TracedPath"],
    ),

    AnimationPattern(
        name="projectile_motion",
        category="physics",
        description="""
Animate projectile motion showing the parabolic path,
with velocity components at each point.
""",
        code_template="""
# Physics parameters
v0 = 5  # Initial velocity
theta = PI / 4  # 45 degrees
g = 9.8

# Time of flight
t_flight = 2 * v0 * np.sin(theta) / g

# Position functions
def x(t): return v0 * np.cos(theta) * t
def y(t): return v0 * np.sin(theta) * t - 0.5 * g * t**2

# Setup
axes = Axes(
    x_range=[0, x(t_flight) + 1],
    y_range=[0, max(v0**2 * np.sin(theta)**2 / (2*g) + 1, 3)],
)

# Time tracker
time = ValueTracker(0)

# Projectile
projectile = always_redraw(lambda: Dot(
    axes.c2p(x(time.get_value()), y(time.get_value())),
    color=RED,
    radius=0.1
))

# Velocity vector
def get_velocity_vec():
    t = time.get_value()
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta) - g * t
    pos = axes.c2p(x(t), y(t))
    vel_scaled = np.array([vx, vy, 0]) * 0.2
    return Arrow(pos, pos + vel_scaled, buff=0, color=GREEN)

velocity = always_redraw(get_velocity_vec)

# Traced path
trace = TracedPath(
    lambda: axes.c2p(x(time.get_value()), y(time.get_value())),
    stroke_color=BLUE,
    stroke_width=3,
)

self.add(axes, trace, projectile, velocity)

# Animate flight
self.play(time.animate.set_value(t_flight), run_time=3, rate_func=linear)
""",
        math_concepts=["projectile", "kinematics", "parabola", "physics"],
        keywords=["projectile", "trajectory", "parabola", "velocity", "TracedPath"],
    ),

    # === NUMBER THEORY PATTERNS ===
    AnimationPattern(
        name="prime_sieve",
        category="number_theory",
        description="""
Animate the Sieve of Eratosthenes, showing primes being
found and composites being crossed out.
""",
        code_template="""
# Create grid of numbers
n = 50
rows, cols = 5, 10

numbers = VGroup()
for i in range(2, n + 1):
    row = (i - 2) // cols
    col = (i - 2) % cols
    num = Integer(i).move_to(
        np.array([col - cols/2 + 0.5, 2 - row, 0]) * 0.8
    )
    numbers.add(num)

self.play(Write(numbers))
self.wait()

# Sieve algorithm
is_prime = [True] * (n + 1)

for p in range(2, int(n**0.5) + 1):
    if is_prime[p]:
        idx = p - 2
        # Highlight current prime
        self.play(
            numbers[idx].animate.set_color(GREEN).scale(1.2),
            run_time=0.3
        )

        # Cross out multiples
        animations = []
        for m in range(p * 2, n + 1, p):
            is_prime[m] = False
            mult_idx = m - 2
            if numbers[mult_idx].get_color() != GRAY:
                animations.append(
                    numbers[mult_idx].animate.set_color(GRAY).set_opacity(0.3)
                )

        if animations:
            self.play(*animations, run_time=0.5)

# Final highlight of all primes
primes_final = [numbers[p - 2] for p in range(2, n + 1) if is_prime[p]]
self.play(*[p.animate.set_color(YELLOW) for p in primes_final])
""",
        math_concepts=["primes", "sieve", "Eratosthenes", "number theory"],
        keywords=["sieve", "prime", "Eratosthenes", "grid"],
    ),

    # === DATA VISUALIZATION PATTERNS ===
    AnimationPattern(
        name="bar_chart_animation",
        category="data",
        description="""
Create an animated bar chart that builds up or updates
to show changing data.
""",
        code_template="""
# Data
categories = ["A", "B", "C", "D", "E"]
values = [3, 7, 2, 8, 5]
max_val = max(values)

# Create bars
bar_width = 0.6
bars = VGroup()
labels = VGroup()

for i, (cat, val) in enumerate(zip(categories, values)):
    bar = Rectangle(
        width=bar_width,
        height=val * 0.4,
        fill_color=BLUE,
        fill_opacity=0.8,
        stroke_width=1
    )
    bar.move_to(np.array([i - len(categories)/2 + 0.5, val * 0.2, 0]))

    label = Tex(cat).next_to(bar, DOWN)
    value_label = Integer(val).next_to(bar, UP)

    bars.add(VGroup(bar, value_label))
    labels.add(label)

# Animate bars growing
for bar in bars:
    bar[0].stretch_to_fit_height(0.01, about_edge=DOWN)
    bar[1].set_opacity(0)

self.play(Write(labels))

# Grow bars one by one
for i, (bar, val) in enumerate(zip(bars, values)):
    self.play(
        bar[0].animate.stretch_to_fit_height(val * 0.4, about_edge=DOWN),
        bar[1].animate.set_opacity(1),
        run_time=0.5
    )

self.wait()

# Update to new values
new_values = [5, 4, 9, 3, 7]
for bar, old_val, new_val in zip(bars, values, new_values):
    new_bar = bar[0].copy()
    new_bar.stretch_to_fit_height(new_val * 0.4, about_edge=DOWN)
    new_label = Integer(new_val).next_to(new_bar, UP)

    self.play(
        Transform(bar[0], new_bar),
        Transform(bar[1], new_label),
        run_time=0.8
    )
""",
        math_concepts=["bar chart", "data visualization", "statistics"],
        keywords=["bar chart", "Rectangle", "data", "visualization"],
    ),

    AnimationPattern(
        name="pie_chart_animation",
        category="data",
        description="""
Create an animated pie chart showing proportions,
with sectors appearing one by one.
""",
        code_template="""
# Data
values = [30, 25, 20, 15, 10]
colors = [RED, BLUE, GREEN, YELLOW, PURPLE]
labels = ["A", "B", "C", "D", "E"]

# Calculate angles
total = sum(values)
angles = [v / total * 2 * PI for v in values]

# Create sectors
sectors = VGroup()
start_angle = PI / 2  # Start from top

for i, (angle, color, label) in enumerate(zip(angles, colors, labels)):
    sector = Sector(
        outer_radius=2,
        angle=angle,
        start_angle=start_angle,
        fill_color=color,
        fill_opacity=0.8,
        stroke_width=2
    )

    # Label position (middle of sector)
    mid_angle = start_angle + angle / 2
    label_pos = 1.2 * np.array([np.cos(mid_angle), np.sin(mid_angle), 0])
    sector_label = Tex(f"{label}: {values[i]}%").move_to(label_pos)

    sectors.add(VGroup(sector, sector_label))
    start_angle += angle

# Animate sectors appearing
for sector in sectors:
    self.play(
        ShowCreation(sector[0]),
        FadeIn(sector[1]),
        run_time=0.7
    )

self.wait()

# Optional: explode a sector
self.play(sectors[0].animate.shift(0.3 * UP))
""",
        math_concepts=["pie chart", "proportions", "data visualization"],
        keywords=["Sector", "pie chart", "proportions", "data"],
    ),

    # === MORE CALCULUS PATTERNS ===
    AnimationPattern(
        name="chain_rule_visualization",
        category="calculus",
        description="""
Visualize the chain rule by showing nested functions and their compositions.
Shows f(g(x)) with both inner and outer derivatives.
""",
        code_template="""
# Setup axes
axes = Axes(x_range=[-2, 4], y_range=[-1, 5])

# Inner function g(x) = x^2
g = lambda x: x**2
g_graph = axes.get_graph(g, color=BLUE, x_range=[0, 2])
g_label = Tex("g(x) = x^2").next_to(g_graph, UP)

# Outer function f(u) = sqrt(u)
f = lambda u: np.sqrt(u) if u >= 0 else 0

# Composition f(g(x)) = |x|
fg = lambda x: f(g(x))
fg_graph = axes.get_graph(fg, color=GREEN, x_range=[0, 2])
fg_label = Tex("f(g(x)) = |x|").next_to(fg_graph, RIGHT)

self.play(ShowCreation(axes))
self.play(ShowCreation(g_graph), Write(g_label))
self.wait()

# Show the composition
self.play(ShowCreation(fg_graph), Write(fg_label))

# Chain rule formula
chain_rule = Tex(r"\\frac{d}{dx}f(g(x)) = f'(g(x)) \\cdot g'(x)")
chain_rule.to_edge(UP)
self.play(Write(chain_rule))
""",
        math_concepts=["chain rule", "composition", "derivative", "calculus"],
        keywords=["chain rule", "composition", "nested", "derivative"],
    ),

    AnimationPattern(
        name="integration_by_parts",
        category="calculus",
        description="""
Visualize integration by parts with the formula and example.
Shows how the integral transforms step by step.
""",
        code_template="""
# Integration by parts formula
formula = Tex(r"\\int u \\, dv = uv - \\int v \\, du")
formula.scale(1.2).to_edge(UP)

# Example: integral of x * e^x
example_title = Tex("Example: ", r"$\\int x e^x \\, dx$")
example_title.next_to(formula, DOWN, buff=0.5)

# Step by step
step1 = Tex(r"u = x, \\quad dv = e^x \\, dx")
step2 = Tex(r"du = dx, \\quad v = e^x")
step3 = Tex(r"= x e^x - \\int e^x \\, dx")
step4 = Tex(r"= x e^x - e^x + C")

steps = VGroup(step1, step2, step3, step4)
steps.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
steps.next_to(example_title, DOWN, buff=0.5)

self.play(Write(formula))
self.wait()
self.play(Write(example_title))

for step in steps:
    self.play(Write(step))
    self.wait(0.5)

# Box the final answer
box = SurroundingRectangle(step4, color=YELLOW)
self.play(ShowCreation(box))
""",
        math_concepts=["integration", "by parts", "calculus", "integral"],
        keywords=["integration", "by parts", "uv", "integral"],
    ),

    # === MORE LINEAR ALGEBRA PATTERNS ===
    AnimationPattern(
        name="eigenvalue_eigenvector",
        category="linear_algebra",
        description="""
Visualize eigenvectors staying on their span while other vectors rotate
during a linear transformation.
""",
        code_template="""
# Setup plane
plane = NumberPlane(x_range=[-4, 4], y_range=[-4, 4])

# Matrix with eigenvalues 3 and 1
matrix = [[2, 1], [1, 2]]
# Eigenvectors: [1,1] with eigenvalue 3, [1,-1] with eigenvalue 1

# Regular vector (will rotate)
regular_vec = Arrow(ORIGIN, [1, 0], buff=0, color=BLUE)
regular_label = Tex("v").next_to(regular_vec.get_end(), UR, buff=0.1)

# Eigenvector (stays on span)
eigen_vec = Arrow(ORIGIN, [1, 1], buff=0, color=GREEN)
eigen_label = Tex(r"\\lambda v").next_to(eigen_vec.get_end(), UR, buff=0.1)

self.play(ShowCreation(plane))
self.play(ShowCreation(regular_vec), Write(regular_label))
self.play(ShowCreation(eigen_vec), Write(eigen_label))
self.wait()

# Apply transformation
self.play(
    plane.animate.apply_matrix(matrix),
    regular_vec.animate.apply_matrix(matrix),
    eigen_vec.animate.apply_matrix(matrix),
    run_time=3
)

# Show eigenvector property
eigen_note = Tex("Eigenvector scaled, not rotated!").to_edge(DOWN)
self.play(Write(eigen_note))
""",
        math_concepts=["eigenvalue", "eigenvector", "linear algebra", "transformation"],
        keywords=["eigenvalue", "eigenvector", "apply_matrix", "span"],
    ),

    AnimationPattern(
        name="determinant_area",
        category="linear_algebra",
        description="""
Visualize the determinant as the scaling factor of area under
a linear transformation.
""",
        code_template="""
# Setup
plane = NumberPlane(x_range=[-3, 3], y_range=[-3, 3])

# Unit square
unit_square = Polygon(
    ORIGIN, RIGHT, RIGHT + UP, UP,
    fill_color=BLUE, fill_opacity=0.5, stroke_color=BLUE
)

# Area label
area_label = Tex("Area = 1").next_to(unit_square, DOWN)

# Matrix to apply (det = 2)
matrix = [[2, 0], [0, 1]]
det_value = 2

self.play(ShowCreation(plane))
self.play(ShowCreation(unit_square), Write(area_label))
self.wait()

# Transform
new_area_label = Tex(f"Area = {det_value}").next_to(
    Polygon(ORIGIN, 2*RIGHT, 2*RIGHT + UP, UP), DOWN
)

self.play(
    plane.animate.apply_matrix(matrix),
    unit_square.animate.apply_matrix(matrix),
    Transform(area_label, new_area_label),
    run_time=2
)

# Show determinant
det_formula = Tex(f"det(A) = {det_value}").to_edge(UP)
self.play(Write(det_formula))
""",
        math_concepts=["determinant", "area", "linear algebra", "scaling"],
        keywords=["determinant", "area", "apply_matrix", "scaling"],
    ),

    # === COMBINATORICS PATTERNS ===
    AnimationPattern(
        name="pascals_triangle",
        category="combinatorics",
        description="""
Build Pascal's triangle row by row, showing binomial coefficients
and the addition pattern.
""",
        code_template="""
# Build Pascal's triangle
n_rows = 6
triangle = VGroup()

for row in range(n_rows):
    row_group = VGroup()
    for col in range(row + 1):
        # Compute binomial coefficient
        from math import comb
        value = comb(row, col)
        num = Integer(value)
        num.move_to(np.array([
            (col - row/2) * 0.8,
            (n_rows/2 - row) * 0.8,
            0
        ]))
        row_group.add(num)
    triangle.add(row_group)

# Animate row by row
for i, row in enumerate(triangle):
    if i == 0:
        self.play(Write(row))
    else:
        # Show arrows from parent numbers
        parent_row = triangle[i-1]
        animations = []
        for j, num in enumerate(row):
            # Each number comes from sum of two above
            if j > 0:
                animations.append(
                    FadeIn(num, shift=DOWN*0.3)
                )
            else:
                animations.append(FadeIn(num, shift=DOWN*0.3))
        self.play(*animations, run_time=0.7)
    self.wait(0.3)

# Highlight a row (binomial expansion)
highlight = SurroundingRectangle(triangle[4], color=YELLOW)
binomial = Tex(r"(a+b)^4 = a^4 + 4a^3b + 6a^2b^2 + 4ab^3 + b^4")
binomial.to_edge(DOWN)
self.play(ShowCreation(highlight), Write(binomial))
""",
        math_concepts=["Pascal triangle", "binomial", "combinatorics", "coefficients"],
        keywords=["Pascal", "triangle", "binomial", "comb"],
    ),

    AnimationPattern(
        name="permutation_cycle",
        category="combinatorics",
        description="""
Visualize a permutation as cycle notation, showing elements
moving to their new positions.
""",
        code_template="""
# Elements
elements = ["A", "B", "C", "D"]
n = len(elements)

# Arrange in circle
radius = 1.5
positions = [
    radius * np.array([np.cos(i * TAU/n + PI/2), np.sin(i * TAU/n + PI/2), 0])
    for i in range(n)
]

# Create labeled dots
dots = VGroup()
labels = VGroup()
for i, (elem, pos) in enumerate(zip(elements, positions)):
    dot = Dot(pos, color=BLUE, radius=0.2)
    label = Tex(elem).move_to(pos)
    dots.add(dot)
    labels.add(label)

self.play(FadeIn(dots), Write(labels))
self.wait()

# Permutation: (A B C D) -> (B C D A) - cycle (1 2 3 4)
# Show arrows
arrows = VGroup()
for i in range(n):
    next_i = (i + 1) % n
    arrow = CurvedArrow(
        positions[i] + 0.3 * normalize(positions[next_i] - positions[i]),
        positions[next_i] - 0.3 * normalize(positions[next_i] - positions[i]),
        color=YELLOW,
        angle=-PI/4
    )
    arrows.add(arrow)

self.play(ShowCreation(arrows))
self.wait()

# Animate the cycle
new_positions = [positions[(i - 1) % n] for i in range(n)]
self.play(*[
    label.animate.move_to(new_pos)
    for label, new_pos in zip(labels, new_positions)
], run_time=2)

# Cycle notation
cycle_notation = Tex("(A \\; B \\; C \\; D)").to_edge(UP)
self.play(Write(cycle_notation))
""",
        math_concepts=["permutation", "cycle", "combinatorics", "group theory"],
        keywords=["permutation", "cycle", "CurvedArrow", "rotation"],
    ),

    # === MORE GEOMETRY PATTERNS ===
    AnimationPattern(
        name="pythagorean_theorem",
        category="geometry",
        description="""
Visualize the Pythagorean theorem with squares on the sides
of a right triangle, showing a² + b² = c².
""",
        code_template="""
# Right triangle
A = np.array([-2, -1, 0])
B = np.array([1, -1, 0])
C = np.array([-2, 2, 0])

triangle = Polygon(A, B, C, color=WHITE, fill_opacity=0.2)

# Side lengths
a = np.linalg.norm(B - C)  # hypotenuse
b = np.linalg.norm(A - C)  # vertical
c_len = np.linalg.norm(B - A)  # horizontal

# Squares on each side
def square_on_side(p1, p2, color):
    side = p2 - p1
    perp = np.array([-side[1], side[0], 0])
    perp = perp / np.linalg.norm(perp) * np.linalg.norm(side)
    return Polygon(p1, p2, p2 + perp, p1 + perp,
                   fill_color=color, fill_opacity=0.5, stroke_color=color)

sq_a = square_on_side(B, C, RED)  # On hypotenuse (outside)
sq_b = square_on_side(C, A, GREEN)  # On vertical (left)
sq_c = square_on_side(A, B, BLUE)  # On horizontal (below)

# Labels
label_a = Tex("c²").move_to(sq_a.get_center())
label_b = Tex("b²").move_to(sq_b.get_center())
label_c = Tex("a²").move_to(sq_c.get_center())

self.play(ShowCreation(triangle))
self.wait()

# Show squares one by one
self.play(ShowCreation(sq_c), Write(label_c))
self.play(ShowCreation(sq_b), Write(label_b))
self.play(ShowCreation(sq_a), Write(label_a))
self.wait()

# Formula
formula = Tex("a² + b² = c²").scale(1.5).to_edge(UP)
self.play(Write(formula))
self.play(Flash(formula, color=YELLOW))
""",
        math_concepts=["Pythagorean theorem", "right triangle", "geometry", "squares"],
        keywords=["Pythagorean", "triangle", "squares", "a² + b² = c²"],
    ),

    AnimationPattern(
        name="similar_triangles",
        category="geometry",
        description="""
Show two similar triangles with corresponding angles and
proportional sides highlighted.
""",
        code_template="""
# Original triangle
A1 = np.array([-3, -1, 0])
B1 = np.array([-1, -1, 0])
C1 = np.array([-2, 1, 0])
tri1 = Polygon(A1, B1, C1, color=BLUE, fill_opacity=0.3)

# Similar triangle (scaled by 1.5)
scale = 1.5
offset = np.array([3, 0, 0])
A2 = offset + scale * (A1 - A1)
B2 = offset + scale * (B1 - A1)
C2 = offset + scale * (C1 - A1)
tri2 = Polygon(A2, B2, C2, color=GREEN, fill_opacity=0.3)

# Labels
labels1 = VGroup(
    Tex("A").next_to(A1, DL, buff=0.1),
    Tex("B").next_to(B1, DR, buff=0.1),
    Tex("C").next_to(C1, UP, buff=0.1),
)
labels2 = VGroup(
    Tex("A'").next_to(A2, DL, buff=0.1),
    Tex("B'").next_to(B2, DR, buff=0.1),
    Tex("C'").next_to(C2, UP, buff=0.1),
)

self.play(ShowCreation(tri1), Write(labels1))
self.play(ShowCreation(tri2), Write(labels2))
self.wait()

# Show angle equality
angle_arcs1 = VGroup(
    Arc(angle=0.4, start_angle=0.3, radius=0.3, color=YELLOW).shift(A1),
    Arc(angle=0.5, start_angle=2.5, radius=0.3, color=RED).shift(B1),
)
angle_arcs2 = VGroup(
    Arc(angle=0.4, start_angle=0.3, radius=0.45, color=YELLOW).shift(A2),
    Arc(angle=0.5, start_angle=2.5, radius=0.45, color=RED).shift(B2),
)

self.play(ShowCreation(angle_arcs1), ShowCreation(angle_arcs2))

# Ratio formula
ratio = Tex(r"\\frac{A'B'}{AB} = \\frac{B'C'}{BC} = \\frac{A'C'}{AC}").to_edge(UP)
self.play(Write(ratio))
""",
        math_concepts=["similar triangles", "proportion", "geometry", "ratio"],
        keywords=["similar", "triangles", "proportion", "scaling"],
    ),

    # === ANIMATION TECHNIQUE PATTERNS ===
    AnimationPattern(
        name="fade_transition",
        category="technique",
        description="""
Smooth fade transition between two scenes or states.
Professional cross-fade effect.
""",
        code_template="""
# First state
state1 = VGroup(
    Square(color=BLUE, fill_opacity=0.5),
    Tex("State 1").scale(0.8)
)

# Second state
state2 = VGroup(
    Circle(color=RED, fill_opacity=0.5),
    Tex("State 2").scale(0.8)
)
state2.move_to(state1)

self.play(FadeIn(state1))
self.wait()

# Cross-fade transition
self.play(
    FadeOut(state1, shift=UP*0.5),
    FadeIn(state2, shift=UP*0.5),
)
self.wait()

# Alternative: use Transform
state3 = VGroup(
    Triangle(color=GREEN, fill_opacity=0.5),
    Tex("State 3").scale(0.8)
)
state3.move_to(state2)

self.play(ReplacementTransform(state2, state3))
""",
        math_concepts=["transition", "fade", "animation"],
        keywords=["FadeIn", "FadeOut", "transition", "cross-fade"],
    ),

    AnimationPattern(
        name="color_wave",
        category="technique",
        description="""
Create a wave of color changes across a group of objects.
Ripple effect animation.
""",
        code_template="""
# Grid of squares
rows, cols = 5, 8
squares = VGroup()
for i in range(rows):
    for j in range(cols):
        sq = Square(side_length=0.5, fill_opacity=0.8, fill_color=BLUE)
        sq.move_to(np.array([j - cols/2 + 0.5, i - rows/2 + 0.5, 0]) * 0.6)
        squares.add(sq)

self.play(FadeIn(squares))
self.wait()

# Color wave from left to right
def color_wave(mob, t):
    for i, sq in enumerate(mob):
        x = sq.get_center()[0]
        # Wave based on position and time
        phase = (x + 3) / 6 - t
        if 0 < phase < 0.3:
            sq.set_fill(YELLOW, opacity=0.9)
        elif phase <= 0:
            sq.set_fill(RED, opacity=0.8)
        else:
            sq.set_fill(BLUE, opacity=0.8)

# Alternative: use LaggedStart
self.play(LaggedStart(*[
    sq.animate.set_fill(YELLOW)
    for sq in squares
], lag_ratio=0.02))

self.play(LaggedStart(*[
    sq.animate.set_fill(BLUE)
    for sq in squares
], lag_ratio=0.02))
""",
        math_concepts=["wave", "color", "ripple", "animation"],
        keywords=["color", "wave", "LaggedStart", "ripple"],
    ),

    AnimationPattern(
        name="typewriter_text",
        category="technique",
        description="""
Typewriter effect for revealing text character by character.
Creates engaging text introductions.
""",
        code_template="""
# Text to type
message = "Hello, Manim!"

# Create individual characters
chars = VGroup(*[Tex(c) for c in message])
chars.arrange(RIGHT, buff=0.05)
chars.move_to(ORIGIN)

# Make all invisible initially
for char in chars:
    char.set_opacity(0)

self.add(chars)

# Typewriter effect
for i, char in enumerate(chars):
    self.play(
        char.animate.set_opacity(1),
        run_time=0.05
    )

self.wait()

# Alternative: AddTextLetterByLetter (if available)
# Or use Write with lag_ratio
text2 = Tex("This is smooth typing")
text2.next_to(chars, DOWN)
self.play(Write(text2, run_time=2, rate_func=linear))
""",
        math_concepts=["text", "typewriter", "reveal", "animation"],
        keywords=["typewriter", "text", "Write", "character"],
    ),

    AnimationPattern(
        name="morphing_shape",
        category="technique",
        description="""
Smoothly morph one shape into another completely different shape.
Demonstrates Transform capabilities.
""",
        code_template="""
# Starting shape
shape1 = Square(side_length=2, color=BLUE, fill_opacity=0.7)

# Intermediate shapes
shape2 = RegularPolygon(n=6, color=GREEN, fill_opacity=0.7)
shape2.scale(1.2)

shape3 = Circle(radius=1, color=RED, fill_opacity=0.7)

shape4 = Star(n=5, color=YELLOW, fill_opacity=0.7)

self.play(ShowCreation(shape1))
self.wait(0.5)

# Morph through shapes
for next_shape in [shape2, shape3, shape4, shape1]:
    self.play(Transform(shape1, next_shape), run_time=1.5)
    self.wait(0.3)

# Final flourish
self.play(shape1.animate.scale(1.5), rate_func=there_and_back)
""",
        math_concepts=["morph", "transform", "shape", "animation"],
        keywords=["Transform", "morph", "shape", "animation"],
    ),

    # === MORE PHYSICS PATTERNS ===
    AnimationPattern(
        name="spring_mass_system",
        category="physics",
        description="""
Animate a spring-mass oscillator with the spring stretching
and compressing as the mass oscillates.
""",
        code_template="""
# Fixed anchor point
anchor = np.array([0, 2, 0])
anchor_dot = Dot(anchor, color=GRAY)

# Spring parameters
rest_length = 1.5
k = 2  # Spring constant
m = 1  # Mass
omega = np.sqrt(k/m)
amplitude = 0.5

# Time tracker
time = ValueTracker(0)

# Mass position
def get_mass_pos():
    t = time.get_value()
    y = rest_length + amplitude * np.cos(omega * t)
    return anchor - np.array([0, y, 0])

# Spring (zigzag line)
def get_spring():
    mass_pos = get_mass_pos()
    n_coils = 10
    points = []
    for i in range(n_coils * 2 + 1):
        t = i / (n_coils * 2)
        y = anchor[1] - t * (anchor[1] - mass_pos[1])
        x = 0.2 * np.sin(i * PI) if 0 < i < n_coils * 2 else 0
        points.append(np.array([x, y, 0]))
    spring = VMobject(color=WHITE)
    spring.set_points_as_corners(points)
    return spring

spring = always_redraw(get_spring)

# Mass (rectangle)
mass = always_redraw(lambda: Square(
    side_length=0.5, fill_color=BLUE, fill_opacity=0.8
).move_to(get_mass_pos()))

self.add(anchor_dot, spring, mass)

# Oscillate
self.play(time.animate.set_value(4 * PI), run_time=8, rate_func=linear)
""",
        math_concepts=["spring", "oscillation", "harmonic motion", "physics"],
        keywords=["spring", "mass", "oscillation", "harmonic"],
    ),

    AnimationPattern(
        name="electric_field_lines",
        category="physics",
        description="""
Visualize electric field lines emanating from point charges.
Shows field direction and strength.
""",
        code_template="""
# Positive charge at origin
charge_pos = ORIGIN
charge = Dot(charge_pos, color=RED, radius=0.2)
charge_label = Tex("+").move_to(charge_pos)

# Field lines radiating outward
n_lines = 8
field_lines = VGroup()
for i in range(n_lines):
    angle = i * TAU / n_lines
    direction = np.array([np.cos(angle), np.sin(angle), 0])
    line = Arrow(
        charge_pos + 0.3 * direction,
        charge_pos + 2 * direction,
        color=YELLOW,
        buff=0
    )
    field_lines.add(line)

self.play(FadeIn(charge), Write(charge_label))
self.play(LaggedStartMap(ShowCreation, field_lines, lag_ratio=0.1))
self.wait()

# Add a negative charge
charge2_pos = np.array([3, 0, 0])
charge2 = Dot(charge2_pos, color=BLUE, radius=0.2)
charge2_label = Tex("-").move_to(charge2_pos)

# Field lines curve between charges
curved_lines = VGroup()
for i in range(5):
    y_offset = (i - 2) * 0.5
    start = charge_pos + np.array([0.3, y_offset, 0])
    end = charge2_pos - np.array([0.3, y_offset, 0])
    mid = (start + end) / 2 + np.array([0, y_offset * 0.5, 0])

    path = VMobject(color=YELLOW)
    path.set_points_smoothly([start, mid, end])
    # Add arrow head
    curved_lines.add(path)

self.play(FadeIn(charge2), Write(charge2_label))
self.play(
    FadeOut(field_lines),
    LaggedStartMap(ShowCreation, curved_lines, lag_ratio=0.1)
)
""",
        math_concepts=["electric field", "charge", "physics", "field lines"],
        keywords=["electric", "field", "charge", "Arrow", "physics"],
    ),

    # === SIGNAL PROCESSING PATTERNS ===
    AnimationPattern(
        name="convolution_animation",
        category="signal",
        description="""
Visualize convolution by sliding one function over another
and showing the overlapping area.
""",
        code_template="""
# Setup
axes = Axes(x_range=[-3, 5], y_range=[-0.5, 2])

# Fixed function f (rectangle pulse)
def f(x):
    return 1 if 0 <= x <= 1 else 0

f_graph = axes.get_graph(f, color=BLUE, x_range=[-1, 2])
f_label = Tex("f(t)").next_to(f_graph, UP)

# Sliding function g (flipped and shifted)
t_tracker = ValueTracker(-1)

def get_g_graph():
    t = t_tracker.get_value()
    def g(x):
        # g(t - x) centered at t
        return 1 if t - 1 <= x <= t else 0
    return axes.get_graph(g, color=RED, x_range=[t-1.5, t+0.5])

g_graph = always_redraw(get_g_graph)

# Shaded overlap region
def get_overlap():
    t = t_tracker.get_value()
    # Compute overlap interval
    left = max(0, t - 1)
    right = min(1, t)
    if left < right:
        return axes.get_area_under_graph(
            f_graph, x_range=[left, right],
            fill_color=GREEN, fill_opacity=0.5
        )
    return VMobject()

overlap = always_redraw(get_overlap)

self.add(axes, f_graph, f_label, g_graph, overlap)

# Slide g across f
self.play(t_tracker.animate.set_value(3), run_time=6, rate_func=linear)
""",
        math_concepts=["convolution", "signal processing", "overlap", "integral"],
        keywords=["convolution", "slide", "overlap", "signal"],
    ),

    # === RECURSION PATTERNS ===
    AnimationPattern(
        name="fractal_tree",
        category="recursion",
        description="""
Animate a recursive fractal tree growing branch by branch.
Shows self-similarity and recursion visually.
""",
        code_template="""
# Recursive tree function
def create_tree(start, angle, length, depth, max_depth):
    if depth > max_depth or length < 0.05:
        return VGroup()

    # Main branch
    end = start + length * np.array([np.sin(angle), np.cos(angle), 0])
    branch = Line(start, end, color=interpolate_color(GREEN, YELLOW, depth/max_depth))

    # Recursively create child branches
    children = VGroup()
    if depth < max_depth:
        # Left branch
        children.add(create_tree(end, angle - PI/5, length * 0.7, depth + 1, max_depth))
        # Right branch
        children.add(create_tree(end, angle + PI/5, length * 0.7, depth + 1, max_depth))

    return VGroup(branch, children)

# Create tree structure
max_depth = 6
tree = create_tree(np.array([0, -2.5, 0]), 0, 1.5, 0, max_depth)

# Animate level by level
def get_branches_at_depth(group, target_depth, current_depth=0):
    branches = []
    for item in group:
        if isinstance(item, Line):
            if current_depth == target_depth:
                branches.append(item)
        elif isinstance(item, VGroup):
            branches.extend(get_branches_at_depth(item, target_depth, current_depth + 1))
    return branches

for d in range(max_depth + 1):
    branches = get_branches_at_depth(tree, d)
    if branches:
        self.play(*[ShowCreation(b) for b in branches], run_time=0.5)
""",
        math_concepts=["fractal", "recursion", "tree", "self-similarity"],
        keywords=["fractal", "tree", "recursion", "branch"],
    ),

    AnimationPattern(
        name="fibonacci_spiral",
        category="recursion",
        description="""
Animate the Fibonacci spiral with squares and the golden spiral.
Shows the connection between Fibonacci numbers and geometry.
""",
        code_template="""
# Fibonacci numbers
fib = [1, 1, 2, 3, 5, 8, 13]

# Build squares
squares = VGroup()
current_pos = ORIGIN
directions = [RIGHT, UP, LEFT, DOWN]  # Spiral direction

for i, size in enumerate(fib):
    sq = Square(side_length=size * 0.3)
    sq.set_stroke(WHITE, width=2)
    sq.set_fill(
        interpolate_color(BLUE, RED, i / len(fib)),
        opacity=0.3
    )

    # Position based on spiral
    if i == 0:
        sq.move_to(ORIGIN)
    elif i == 1:
        sq.next_to(squares[0], RIGHT, buff=0)
    else:
        # Each new square attaches to the growing spiral
        direction = directions[(i - 1) % 4]
        sq.next_to(VGroup(*squares[:i]), direction, buff=0)

    squares.add(sq)

# Animate squares appearing
for sq in squares:
    self.play(ShowCreation(sq), run_time=0.5)

# Draw the golden spiral
spiral_points = []
# (Simplified spiral - actual implementation would use arcs)
for i, sq in enumerate(squares):
    center = sq.get_center()
    spiral_points.append(center)

if len(spiral_points) > 2:
    spiral = VMobject(color=YELLOW, stroke_width=3)
    spiral.set_points_smoothly(spiral_points)
    self.play(ShowCreation(spiral), run_time=2)

# Label
golden = Tex(r"\\phi = \\frac{1 + \\sqrt{5}}{2}").to_edge(UP)
self.play(Write(golden))
""",
        math_concepts=["Fibonacci", "spiral", "golden ratio", "recursion"],
        keywords=["Fibonacci", "spiral", "golden", "sequence"],
    ),

    # === LOGIC/SET THEORY PATTERNS ===
    AnimationPattern(
        name="venn_diagram",
        category="sets",
        description="""
Animate a Venn diagram showing set operations like
union, intersection, and difference.
""",
        code_template="""
# Two overlapping circles
A = Circle(radius=1.5, color=BLUE, fill_opacity=0.3)
A.shift(LEFT * 0.8)
B = Circle(radius=1.5, color=RED, fill_opacity=0.3)
B.shift(RIGHT * 0.8)

# Labels
A_label = Tex("A").move_to(A.get_center() + LEFT * 0.8)
B_label = Tex("B").move_to(B.get_center() + RIGHT * 0.8)

self.play(ShowCreation(A), ShowCreation(B))
self.play(Write(A_label), Write(B_label))
self.wait()

# Highlight intersection A ∩ B
intersection = Intersection(A.copy(), B.copy(), fill_color=PURPLE, fill_opacity=0.7)
intersection_label = Tex(r"A \\cap B").to_edge(UP)

self.play(FadeIn(intersection), Write(intersection_label))
self.wait()
self.play(FadeOut(intersection), FadeOut(intersection_label))

# Highlight union A ∪ B
union = Union(A.copy(), B.copy(), fill_color=GREEN, fill_opacity=0.5)
union_label = Tex(r"A \\cup B").to_edge(UP)

self.play(FadeIn(union), Write(union_label))
self.wait()
self.play(FadeOut(union), FadeOut(union_label))

# Highlight difference A - B
difference = Difference(A.copy(), B.copy(), fill_color=YELLOW, fill_opacity=0.5)
diff_label = Tex(r"A - B").to_edge(UP)

self.play(FadeIn(difference), Write(diff_label))
""",
        math_concepts=["Venn diagram", "sets", "union", "intersection", "difference"],
        keywords=["Venn", "sets", "Intersection", "Union", "Difference"],
    ),

    # === COORDINATE GEOMETRY PATTERNS ===
    AnimationPattern(
        name="parametric_curve",
        category="coordinate",
        description="""
Animate a parametric curve being traced by a moving point,
showing both x(t) and y(t) components.
""",
        code_template="""
# Setup axes
axes = Axes(x_range=[-3, 3], y_range=[-3, 3])

# Parametric equations (Lissajous curve)
def x(t): return 2 * np.sin(3 * t)
def y(t): return 2 * np.sin(2 * t)

# Create curve
t_tracker = ValueTracker(0)

# Tracing dot
dot = always_redraw(lambda: Dot(
    axes.c2p(x(t_tracker.get_value()), y(t_tracker.get_value())),
    color=YELLOW
))

# Traced path
trace = TracedPath(
    lambda: axes.c2p(x(t_tracker.get_value()), y(t_tracker.get_value())),
    stroke_color=BLUE,
    stroke_width=3,
)

# Parametric equations label
equations = VGroup(
    Tex(r"x(t) = 2\\sin(3t)"),
    Tex(r"y(t) = 2\\sin(2t)")
).arrange(DOWN).to_edge(UL)

self.add(axes, trace, dot)
self.play(Write(equations))

# Trace the curve
self.play(t_tracker.animate.set_value(2 * PI), run_time=6, rate_func=linear)
self.wait()

# Continue for second loop
self.play(t_tracker.animate.set_value(4 * PI), run_time=6, rate_func=linear)
""",
        math_concepts=["parametric", "curve", "Lissajous", "coordinates"],
        keywords=["parametric", "TracedPath", "Lissajous", "curve"],
    ),

    AnimationPattern(
        name="polar_rose",
        category="coordinate",
        description="""
Animate a polar rose curve r = cos(k*theta) being drawn.
Beautiful mathematical curve visualization.
""",
        code_template="""
# Polar plane (or just axes)
axes = Axes(x_range=[-3, 3], y_range=[-3, 3])

# Rose parameters
k = 4  # Number of petals
amplitude = 2

# Polar to Cartesian
def polar_to_cart(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta), 0])

# Create the rose curve
theta_tracker = ValueTracker(0)

def get_rose_point():
    theta = theta_tracker.get_value()
    r = amplitude * np.cos(k * theta)
    return axes.c2p(*polar_to_cart(r, theta)[:2])

# Tracing dot and path
dot = always_redraw(lambda: Dot(get_rose_point(), color=RED))
trace = TracedPath(get_rose_point, stroke_color=PINK, stroke_width=3)

# Equation
equation = Tex(f"r = {amplitude}\\\\cos({k}\\\\theta)").to_edge(UP)

self.add(axes, trace, dot)
self.play(Write(equation))

# Draw the rose
self.play(
    theta_tracker.animate.set_value(PI if k % 2 == 1 else 2*PI),
    run_time=6,
    rate_func=linear
)
""",
        math_concepts=["polar", "rose", "curve", "trigonometry"],
        keywords=["polar", "rose", "cos", "TracedPath"],
    ),

    # === MORE CALCULUS PATTERNS ===
    AnimationPattern(
        name="related_rates",
        category="calculus",
        description="Visualize related rates problem with changing quantities.",
        code_template="""
# Ladder sliding down wall problem
wall = Line(3*UP, ORIGIN, color=WHITE)
floor = Line(ORIGIN, 4*RIGHT, color=WHITE)

# Ladder length is constant
ladder_length = 3

# Height tracker (y decreasing)
y_tracker = ValueTracker(2.5)

def get_ladder():
    y = y_tracker.get_value()
    x = np.sqrt(ladder_length**2 - y**2)
    return Line(np.array([0, y, 0]), np.array([x, 0, 0]), color=BLUE, stroke_width=4)

ladder = always_redraw(get_ladder)

# Labels
y_label = always_redraw(lambda: Tex(f"y = {y_tracker.get_value():.1f}").next_to(wall, LEFT))

self.add(wall, floor, ladder, y_label)
self.play(y_tracker.animate.set_value(0.5), run_time=4)
""",
        math_concepts=["related rates", "implicit differentiation", "calculus"],
        keywords=["related rates", "ladder", "changing", "tracker"],
    ),

    AnimationPattern(
        name="mean_value_theorem",
        category="calculus",
        description="Visualize MVT with secant and parallel tangent lines.",
        code_template="""
axes = Axes(x_range=[0, 5], y_range=[0, 4])
func = lambda x: 0.2*x**2 + 0.5
graph = axes.get_graph(func, color=BLUE)

a, b = 1, 4
# Secant line
secant = Line(axes.c2p(a, func(a)), axes.c2p(b, func(b)), color=YELLOW)
slope = (func(b) - func(a)) / (b - a)

# Find c where f'(c) = slope (derivative = 0.4x, so c = slope/0.4)
c = slope / 0.4
tangent_at_c = axes.get_graph(lambda x: slope*(x - c) + func(c), color=GREEN, x_range=[c-1, c+1])

# Points and labels
dots = VGroup(Dot(axes.c2p(a, func(a))), Dot(axes.c2p(b, func(b))), Dot(axes.c2p(c, func(c)), color=RED))

self.play(ShowCreation(axes), ShowCreation(graph))
self.play(ShowCreation(secant), FadeIn(dots[:2]))
self.wait()
self.play(ShowCreation(tangent_at_c), FadeIn(dots[2]))

mvt = Tex(r"f'(c) = \\frac{f(b) - f(a)}{b - a}").to_edge(UP)
self.play(Write(mvt))
""",
        math_concepts=["mean value theorem", "secant", "tangent", "calculus"],
        keywords=["MVT", "mean value", "secant", "tangent"],
    ),

    AnimationPattern(
        name="lhopitals_rule",
        category="calculus",
        description="Visualize L'Hopital's rule for indeterminate forms.",
        code_template="""
# Show 0/0 form and resolution
original = Tex(r"\\lim_{x \\to 0} \\frac{\\sin x}{x}")
equals = Tex("=")
indeterminate = Tex(r"\\frac{0}{0}")
question = Tex("?")

row1 = VGroup(original, equals, indeterminate, question).arrange(RIGHT)
row1.to_edge(UP)

self.play(Write(original))
self.play(Write(equals), Write(indeterminate))
self.play(Write(question))
self.wait()

# Apply L'Hopital
arrow = Tex(r"\\xrightarrow{\\text{L'H}}")
derivative = Tex(r"\\lim_{x \\to 0} \\frac{\\cos x}{1}")
result = Tex("= 1")

row2 = VGroup(arrow, derivative, result).arrange(RIGHT)
row2.next_to(row1, DOWN, buff=0.5)

self.play(Write(arrow))
self.play(Write(derivative))
self.play(Write(result))
self.play(Indicate(result, color=YELLOW))
""",
        math_concepts=["L'Hopital", "limit", "indeterminate", "calculus"],
        keywords=["lhopital", "limit", "indeterminate", "derivative"],
    ),

    AnimationPattern(
        name="taylor_series",
        category="calculus",
        description="Animate Taylor series approximation converging to function.",
        code_template="""
axes = Axes(x_range=[-3, 3], y_range=[-2, 4])

# Target function: e^x
func = lambda x: np.exp(x)
target = axes.get_graph(func, color=WHITE, x_range=[-2, 1.5])

# Taylor approximations of increasing order
def taylor(n):
    def f(x):
        return sum(x**k / np.math.factorial(k) for k in range(n+1))
    return f

approximations = [axes.get_graph(taylor(n), color=BLUE, x_range=[-2, 1.5]) for n in range(6)]

n_label = Tex("n = 0").to_edge(UP)
self.add(axes, target)
current = approximations[0]
self.play(ShowCreation(current), Write(n_label))

for i, approx in enumerate(approximations[1:], 1):
    new_label = Tex(f"n = {i}").to_edge(UP)
    self.play(Transform(current, approx), Transform(n_label, new_label))
    self.wait(0.5)
""",
        math_concepts=["Taylor series", "approximation", "polynomial", "calculus"],
        keywords=["Taylor", "series", "polynomial", "approximation"],
    ),

    # === MORE GEOMETRY PATTERNS ===
    AnimationPattern(
        name="angle_bisector",
        category="geometry",
        description="Construct and animate angle bisector theorem.",
        code_template="""
A, B, C = np.array([-2, -1, 0]), np.array([2, -1, 0]), np.array([0, 2, 0])
triangle = Polygon(A, B, C, color=WHITE)

# Angle at A
AB = B - A
AC = C - A
bisector_dir = AB/np.linalg.norm(AB) + AC/np.linalg.norm(AC)
bisector_dir = bisector_dir / np.linalg.norm(bisector_dir)

# Find where bisector meets BC
t = 5  # extend far enough
bisector_end = A + t * bisector_dir
bisector = DashedLine(A, bisector_end, color=YELLOW)

self.play(ShowCreation(triangle))
self.wait()

# Show angle arcs
arc1 = Arc(radius=0.5, start_angle=np.arctan2(AB[1], AB[0]), angle=0.3).shift(A)
arc2 = Arc(radius=0.5, start_angle=np.arctan2(AC[1], AC[0]) - 0.3, angle=0.3).shift(A)

self.play(ShowCreation(arc1), ShowCreation(arc2))
self.play(ShowCreation(bisector))
""",
        math_concepts=["angle bisector", "triangle", "geometry"],
        keywords=["bisector", "angle", "triangle", "geometry"],
    ),

    AnimationPattern(
        name="parallel_lines_transversal",
        category="geometry",
        description="Show angle relationships with parallel lines and transversal.",
        code_template="""
# Parallel lines
line1 = Line(4*LEFT + UP, 4*RIGHT + UP, color=BLUE)
line2 = Line(4*LEFT + DOWN, 4*RIGHT + DOWN, color=BLUE)

# Transversal
transversal = Line(3*LEFT + 2*UP, 3*RIGHT + 2*DOWN, color=WHITE)

# Intersection points
p1 = np.array([0, 1, 0])
p2 = np.array([0, -1, 0])

# Angle arcs (corresponding angles)
angle1 = Arc(radius=0.4, start_angle=PI, angle=-PI/4, color=YELLOW).shift(p1)
angle2 = Arc(radius=0.4, start_angle=PI, angle=-PI/4, color=YELLOW).shift(p2)

self.play(ShowCreation(line1), ShowCreation(line2))
self.play(ShowCreation(transversal))
self.play(ShowCreation(angle1), ShowCreation(angle2))

label = Tex("Corresponding angles are equal").to_edge(UP)
self.play(Write(label))
""",
        math_concepts=["parallel lines", "transversal", "angles", "geometry"],
        keywords=["parallel", "transversal", "corresponding", "angles"],
    ),

    AnimationPattern(
        name="regular_polygon_construction",
        category="geometry",
        description="Construct regular polygons inscribed in a circle.",
        code_template="""
circle = Circle(radius=2, color=BLUE)
center = Dot(ORIGIN, color=WHITE)

self.play(ShowCreation(circle), FadeIn(center))

# Build polygons from triangle to octagon
for n in range(3, 9):
    angles = [k * TAU / n + PI/2 for k in range(n)]
    vertices = [2 * np.array([np.cos(a), np.sin(a), 0]) for a in angles]
    polygon = Polygon(*vertices, color=YELLOW)

    label = Tex(f"n = {n}").to_edge(UP)

    if n == 3:
        self.play(ShowCreation(polygon), Write(label))
    else:
        new_polygon = Polygon(*vertices, color=YELLOW)
        new_label = Tex(f"n = {n}").to_edge(UP)
        self.play(Transform(polygon, new_polygon), Transform(label, new_label))
    self.wait(0.5)
""",
        math_concepts=["regular polygon", "inscribed", "circle", "geometry"],
        keywords=["polygon", "regular", "inscribed", "circle"],
    ),

    AnimationPattern(
        name="reflection_symmetry",
        category="geometry",
        description="Demonstrate reflection across a line of symmetry.",
        code_template="""
# Original shape
shape = VGroup(
    Polygon([-1, 0, 0], [-0.5, 1, 0], [0, 0.5, 0], color=BLUE, fill_opacity=0.5),
    Dot([-0.5, 0.5, 0], color=YELLOW)
)
shape.shift(LEFT * 2)

# Line of symmetry
symmetry_line = DashedLine(3*UP, 3*DOWN, color=WHITE)

self.play(ShowCreation(shape))
self.play(ShowCreation(symmetry_line))

# Create reflection
reflected = shape.copy()
reflected.flip(RIGHT)
reflected.shift(RIGHT * 4)
reflected.set_color(RED)

self.play(ReplacementTransform(shape.copy(), reflected))

label = Tex("Reflection Symmetry").to_edge(UP)
self.play(Write(label))
""",
        math_concepts=["reflection", "symmetry", "transformation", "geometry"],
        keywords=["reflection", "symmetry", "flip", "mirror"],
    ),

    # === MORE LINEAR ALGEBRA PATTERNS ===
    AnimationPattern(
        name="null_space",
        category="linear_algebra",
        description="Visualize the null space of a matrix transformation.",
        code_template="""
plane = NumberPlane(x_range=[-4, 4], y_range=[-4, 4])

# Matrix that projects onto x-axis (null space is y-axis)
matrix = [[1, 0], [0, 0]]

# Vectors in null space (vertical)
null_vecs = VGroup(*[
    Arrow(ORIGIN, y*UP, color=RED, buff=0)
    for y in [-2, -1, 1, 2]
])

# Regular vector
regular = Arrow(ORIGIN, [2, 1, 0], color=BLUE, buff=0)

self.play(ShowCreation(plane))
self.play(ShowCreation(null_vecs), ShowCreation(regular))

# Apply transformation
self.play(
    plane.animate.apply_matrix(matrix),
    null_vecs.animate.apply_matrix(matrix),
    regular.animate.apply_matrix(matrix),
    run_time=2
)

label = Tex("Null space vectors map to 0").to_edge(UP)
self.play(Write(label))
""",
        math_concepts=["null space", "kernel", "linear algebra", "matrix"],
        keywords=["null space", "kernel", "apply_matrix", "zero"],
    ),

    AnimationPattern(
        name="gram_schmidt",
        category="linear_algebra",
        description="Animate Gram-Schmidt orthogonalization process.",
        code_template="""
plane = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], faded_line_ratio=2)

# Original vectors (not orthogonal)
v1 = np.array([2, 1, 0])
v2 = np.array([1, 2, 0])

vec1 = Arrow(ORIGIN, v1, color=BLUE, buff=0)
vec2 = Arrow(ORIGIN, v2, color=RED, buff=0)

self.play(ShowCreation(plane))
self.play(ShowCreation(vec1), ShowCreation(vec2))

# Orthogonalize: u1 = v1, u2 = v2 - proj(v2 onto u1)
u1 = v1
proj = np.dot(v2, u1) / np.dot(u1, u1) * u1
u2 = v2 - proj

# Show projection
proj_vec = Arrow(ORIGIN, proj, color=YELLOW, buff=0)
self.play(ShowCreation(proj_vec))

# Show orthogonal result
orth_vec = Arrow(ORIGIN, u2, color=GREEN, buff=0)
self.play(Transform(vec2, orth_vec))

label = Tex("Gram-Schmidt Orthogonalization").to_edge(UP)
self.play(Write(label))
""",
        math_concepts=["Gram-Schmidt", "orthogonalization", "linear algebra"],
        keywords=["Gram-Schmidt", "orthogonal", "projection", "basis"],
    ),

    AnimationPattern(
        name="change_of_basis",
        category="linear_algebra",
        description="Visualize vector representation in different bases.",
        code_template="""
plane = NumberPlane(x_range=[-4, 4], y_range=[-4, 4])

# Standard basis
e1 = Arrow(ORIGIN, RIGHT, color=GREEN, buff=0)
e2 = Arrow(ORIGIN, UP, color=RED, buff=0)

# New basis
b1 = Arrow(ORIGIN, np.array([1, 1, 0])/np.sqrt(2), color=YELLOW, buff=0)
b2 = Arrow(ORIGIN, np.array([-1, 1, 0])/np.sqrt(2), color=PURPLE, buff=0)

# Vector to represent
v = np.array([2, 1, 0])
vec = Arrow(ORIGIN, v, color=WHITE, buff=0)

self.play(ShowCreation(plane))
self.play(ShowCreation(e1), ShowCreation(e2))
self.play(ShowCreation(vec))
self.wait()

# Show new basis
self.play(ShowCreation(b1), ShowCreation(b2))

# Decompose in new basis
label1 = Tex("Standard basis: (2, 1)").to_edge(UL)
self.play(Write(label1))
self.wait()

label2 = Tex("New basis: different coords").next_to(label1, DOWN)
self.play(Write(label2))
""",
        math_concepts=["change of basis", "coordinates", "linear algebra"],
        keywords=["basis", "change", "coordinates", "representation"],
    ),

    # === DIFFERENTIAL EQUATIONS ===
    AnimationPattern(
        name="slope_field",
        category="diffeq",
        description="Visualize a slope field for a differential equation.",
        code_template="""
axes = Axes(x_range=[-3, 3], y_range=[-3, 3])

# dy/dx = x - y
def slope(x, y):
    return x - y

# Draw slope field
slopes = VGroup()
for x in np.arange(-2.5, 3, 0.5):
    for y in np.arange(-2.5, 3, 0.5):
        m = slope(x, y)
        angle = np.arctan(m)
        line = Line(ORIGIN, 0.3*RIGHT, color=BLUE, stroke_width=2)
        line.rotate(angle)
        line.move_to(axes.c2p(x, y))
        slopes.add(line)

self.play(ShowCreation(axes))
self.play(LaggedStartMap(ShowCreation, slopes, lag_ratio=0.01), run_time=2)

# Draw a solution curve
t = ValueTracker(-2)
# Solution: y = x - 1 + Ce^(-x), with y(-2) = 0
C = (0 - (-2) + 1) * np.exp(-2)

def sol(x):
    return x - 1 + C * np.exp(-x)

curve = always_redraw(lambda: axes.get_graph(sol, x_range=[-2, t.get_value()], color=YELLOW))
self.add(curve)
self.play(t.animate.set_value(2.5), run_time=3)
""",
        math_concepts=["slope field", "differential equation", "ODE"],
        keywords=["slope field", "diffeq", "ODE", "direction"],
    ),

    AnimationPattern(
        name="phase_plane",
        category="diffeq",
        description="Animate trajectories in a 2D phase plane.",
        code_template="""
axes = Axes(x_range=[-3, 3], y_range=[-3, 3])
axes_labels = axes.get_axis_labels(x_label="x", y_label="y")

# Simple harmonic oscillator: x' = y, y' = -x
# Trajectories are circles

circles = VGroup(*[
    Circle(radius=r, color=BLUE, stroke_opacity=0.5)
    for r in [0.5, 1, 1.5, 2, 2.5]
])

# Equilibrium at origin
eq_point = Dot(ORIGIN, color=RED)

self.play(ShowCreation(axes), Write(axes_labels))
self.play(LaggedStartMap(ShowCreation, circles, lag_ratio=0.2))
self.play(FadeIn(eq_point))

# Animate a point moving along a trajectory
t_tracker = ValueTracker(0)
radius = 1.5
moving_dot = always_redraw(lambda: Dot(
    axes.c2p(radius*np.cos(t_tracker.get_value()), radius*np.sin(t_tracker.get_value())),
    color=YELLOW
))

self.add(moving_dot)
self.play(t_tracker.animate.set_value(2*PI), run_time=4, rate_func=linear)
""",
        math_concepts=["phase plane", "trajectory", "differential equation"],
        keywords=["phase", "trajectory", "orbit", "equilibrium"],
    ),

    # === PROBABILITY PATTERNS ===
    AnimationPattern(
        name="central_limit_theorem",
        category="probability",
        description="Visualize CLT with sum of random variables.",
        code_template="""
axes = Axes(x_range=[-4, 4], y_range=[0, 0.5])

# Start with uniform distribution bars
n_samples = 1
uniform_bars = VGroup()
for x in [-1, 0, 1]:
    bar = Rectangle(width=0.8, height=0.3, fill_color=BLUE, fill_opacity=0.7)
    bar.move_to(axes.c2p(x, 0.15))
    uniform_bars.add(bar)

# Normal distribution for comparison
normal = lambda x: 0.4 * np.exp(-x**2/2)
normal_curve = axes.get_graph(normal, color=YELLOW)

self.play(ShowCreation(axes))
self.play(FadeIn(uniform_bars))
self.wait()

label = Tex("n = 1").to_edge(UP)
self.play(Write(label))

# Transform to normal as n increases
for n in [2, 5, 10, 30]:
    # More bell-shaped distribution
    new_bars = VGroup()
    width = 4 / (n + 1)
    for i, x in enumerate(np.linspace(-2, 2, n + 1)):
        height = normal(x) * 2
        bar = Rectangle(width=width, height=height, fill_color=BLUE, fill_opacity=0.7)
        bar.move_to(axes.c2p(x, height/2))
        new_bars.add(bar)

    new_label = Tex(f"n = {n}").to_edge(UP)
    self.play(Transform(uniform_bars, new_bars), Transform(label, new_label))
    self.wait(0.5)

self.play(ShowCreation(normal_curve))
""",
        math_concepts=["central limit theorem", "normal distribution", "probability"],
        keywords=["CLT", "normal", "sum", "distribution"],
    ),

    AnimationPattern(
        name="bayes_theorem",
        category="probability",
        description="Visualize Bayes' theorem with area diagram.",
        code_template="""
# Prior: P(A)
prior_rect = Rectangle(width=4, height=3, fill_color=BLUE, fill_opacity=0.3)
prior_rect.move_to(LEFT * 2)

# P(B|A) region
likelihood = Rectangle(width=4, height=1.5, fill_color=GREEN, fill_opacity=0.5)
likelihood.move_to(LEFT * 2 + DOWN * 0.75)

# P(A and B)
intersection = Rectangle(width=2, height=1.5, fill_color=YELLOW, fill_opacity=0.7)
intersection.move_to(LEFT * 3 + DOWN * 0.75)

# Labels
a_label = Tex("P(A)").next_to(prior_rect, UP)
b_given_a = Tex("P(B|A)").next_to(likelihood, RIGHT)

self.play(ShowCreation(prior_rect), Write(a_label))
self.wait()
self.play(ShowCreation(likelihood), Write(b_given_a))
self.wait()
self.play(ShowCreation(intersection))

# Bayes formula
formula = Tex(r"P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}")
formula.to_edge(UP)
self.play(Write(formula))
""",
        math_concepts=["Bayes theorem", "conditional probability", "probability"],
        keywords=["Bayes", "conditional", "prior", "posterior"],
    ),

    AnimationPattern(
        name="expected_value",
        category="probability",
        description="Visualize expected value as weighted average.",
        code_template="""
axes = Axes(x_range=[0, 7], y_range=[0, 0.4])

# Discrete probability distribution
values = [1, 2, 3, 4, 5, 6]
probs = [0.1, 0.15, 0.25, 0.25, 0.15, 0.1]

bars = VGroup()
for x, p in zip(values, probs):
    bar = Rectangle(width=0.6, height=p*5, fill_color=BLUE, fill_opacity=0.7)
    bar.move_to(axes.c2p(x, p*2.5))
    bars.add(bar)

# Expected value
E = sum(x * p for x, p in zip(values, probs))
ev_line = DashedLine(axes.c2p(E, 0), axes.c2p(E, 0.4), color=RED)
ev_label = Tex(f"E[X] = {E:.1f}").next_to(ev_line, UP)

self.play(ShowCreation(axes))
self.play(LaggedStartMap(FadeIn, bars, lag_ratio=0.1))
self.wait()
self.play(ShowCreation(ev_line), Write(ev_label))

# Show it as balance point
balance = Triangle(fill_color=RED, fill_opacity=0.8).scale(0.3)
balance.next_to(axes.c2p(E, 0), DOWN)
self.play(FadeIn(balance))
""",
        math_concepts=["expected value", "mean", "probability"],
        keywords=["expected", "mean", "average", "probability"],
    ),

    # === NUMBER THEORY PATTERNS ===
    AnimationPattern(
        name="gcd_euclidean",
        category="number_theory",
        description="Animate Euclidean algorithm for GCD.",
        code_template="""
# GCD(48, 18)
a, b = 48, 18
steps = []
while b:
    steps.append((a, b, a % b))
    a, b = b, a % b
gcd = a

# Display as rectangles being subdivided
current_rect = Rectangle(width=4.8, height=1.8, color=BLUE)
current_rect.to_edge(LEFT)

step_texts = VGroup()
for i, (a, b, r) in enumerate(steps):
    text = Tex(f"{a} = {a//b} \\\\times {b} + {r}")
    step_texts.add(text)
step_texts.arrange(DOWN)
step_texts.to_edge(RIGHT)

self.play(ShowCreation(current_rect))

for i, (a, b, r) in enumerate(steps):
    self.play(Write(step_texts[i]))
    self.wait(0.5)

result = Tex(f"GCD = {gcd}").to_edge(DOWN)
self.play(Write(result))
self.play(Indicate(result, color=YELLOW))
""",
        math_concepts=["GCD", "Euclidean algorithm", "number theory"],
        keywords=["GCD", "Euclidean", "divisor", "algorithm"],
    ),

    AnimationPattern(
        name="modular_arithmetic",
        category="number_theory",
        description="Visualize modular arithmetic on a clock.",
        code_template="""
# Clock circle for mod 12
clock = Circle(radius=2, color=WHITE)
center = Dot(ORIGIN)

# Hour marks
marks = VGroup()
labels = VGroup()
for i in range(12):
    angle = PI/2 - i * TAU/12
    pos = 2 * np.array([np.cos(angle), np.sin(angle), 0])
    mark = Line(0.9*pos, pos, color=WHITE)
    label = Tex(str(i)).scale(0.7).move_to(1.15*pos)
    marks.add(mark)
    labels.add(label)

self.play(ShowCreation(clock), FadeIn(center))
self.play(ShowCreation(marks), Write(labels))

# Show addition: 7 + 8 = 3 (mod 12)
# Arrow from 7 to 7+8=15=3
start_angle = PI/2 - 7 * TAU/12
end_angle = PI/2 - 3 * TAU/12

arc = Arc(radius=1.5, start_angle=start_angle, angle=-8*TAU/12, color=YELLOW)
self.play(ShowCreation(arc), run_time=2)

result = Tex(r"7 + 8 \\equiv 3 \\pmod{12}").to_edge(UP)
self.play(Write(result))
""",
        math_concepts=["modular arithmetic", "congruence", "number theory"],
        keywords=["modular", "mod", "congruence", "clock"],
    ),

    # === TOPOLOGY/MISC PATTERNS ===
    AnimationPattern(
        name="mobius_strip",
        category="topology",
        description="Construct a Mobius strip showing non-orientability.",
        code_template="""
# Create a strip
strip = Rectangle(width=4, height=0.5, fill_color=BLUE, fill_opacity=0.5)

# Show the two ends
left_end = strip.copy().set_fill(RED, opacity=0.7)
left_end.stretch(0.2, 0)
left_end.move_to(strip.get_left())

right_end = strip.copy().set_fill(GREEN, opacity=0.7)
right_end.stretch(0.2, 0)
right_end.move_to(strip.get_right())

self.play(ShowCreation(strip))
self.play(FadeIn(left_end), FadeIn(right_end))

# Label the sides
top_label = Tex("A").next_to(strip, UP)
self.play(Write(top_label))

# Indicate twist
twist_arrow = CurvedArrow(RIGHT*2 + UP, RIGHT*2 + DOWN, color=YELLOW)
twist_label = Tex("Twist").next_to(twist_arrow, RIGHT)

self.play(ShowCreation(twist_arrow), Write(twist_label))

# Result label
mobius = Tex("Möbius Strip - One Side!").to_edge(DOWN)
self.play(Write(mobius))
""",
        math_concepts=["Mobius strip", "topology", "non-orientable"],
        keywords=["Mobius", "strip", "topology", "twist"],
    ),

    AnimationPattern(
        name="fixed_point_iteration",
        category="numerical",
        description="Visualize fixed point iteration convergence.",
        code_template="""
axes = Axes(x_range=[0, 3], y_range=[0, 3])

# y = x line
identity = axes.get_graph(lambda x: x, color=WHITE)

# g(x) = sqrt(x + 1)
g = lambda x: np.sqrt(x + 1)
g_graph = axes.get_graph(g, color=BLUE)

self.play(ShowCreation(axes))
self.play(ShowCreation(identity), ShowCreation(g_graph))

# Fixed point iteration from x0 = 0.5
x = 0.5
path = VGroup()
for _ in range(6):
    y = g(x)
    # Vertical line to g(x)
    v_line = Line(axes.c2p(x, x), axes.c2p(x, y), color=YELLOW)
    # Horizontal line to y=x
    h_line = Line(axes.c2p(x, y), axes.c2p(y, y), color=YELLOW)
    path.add(v_line, h_line)
    x = y

self.play(ShowCreation(path), run_time=3)

# Mark fixed point
fixed = Dot(axes.c2p(1.618, 1.618), color=RED)
label = Tex(r"x^* \\approx 1.618").next_to(fixed, UR)
self.play(FadeIn(fixed), Write(label))
""",
        math_concepts=["fixed point", "iteration", "convergence", "numerical"],
        keywords=["fixed point", "iteration", "cobweb", "convergence"],
    ),

    # === ANIMATION EFFECTS ===
    AnimationPattern(
        name="particle_explosion",
        category="effect",
        description="Particle explosion effect from a center point.",
        code_template="""
center = ORIGIN
n_particles = 30

particles = VGroup(*[
    Dot(center, color=random_color(), radius=0.05)
    for _ in range(n_particles)
])

# Random directions
directions = [
    np.array([np.cos(i*TAU/n_particles + np.random.uniform(-0.2, 0.2)),
              np.sin(i*TAU/n_particles + np.random.uniform(-0.2, 0.2)), 0])
    for i in range(n_particles)
]

self.add(particles)

# Explode
self.play(*[
    p.animate.move_to(center + 3 * d).set_opacity(0)
    for p, d in zip(particles, directions)
], run_time=1.5)
""",
        math_concepts=["animation", "particles", "explosion"],
        keywords=["particles", "explosion", "scatter", "effect"],
    ),

    AnimationPattern(
        name="spotlight_focus",
        category="effect",
        description="Spotlight effect focusing on an object.",
        code_template="""
# Create some background objects
bg = VGroup(*[
    Square(side_length=0.5, fill_opacity=0.3, fill_color=GRAY).move_to(
        np.array([np.random.uniform(-4, 4), np.random.uniform(-2, 2), 0])
    )
    for _ in range(20)
])

# Main object to focus on
main = Star(n=5, fill_color=YELLOW, fill_opacity=0.8).scale(0.8)

self.play(FadeIn(bg), FadeIn(main))

# Dim background
self.play(bg.animate.set_opacity(0.1))

# Grow and highlight main
self.play(
    main.animate.scale(1.5),
    Flash(main, color=YELLOW)
)

# Restore
self.play(bg.animate.set_opacity(0.3), main.animate.scale(1/1.5))
""",
        math_concepts=["spotlight", "focus", "highlight"],
        keywords=["spotlight", "focus", "dim", "highlight"],
    ),

    AnimationPattern(
        name="count_animation",
        category="effect",
        description="Animate counting numbers up or down.",
        code_template="""
# Counter display
counter = Integer(0).scale(3)

self.add(counter)

# Count up to 100
self.play(
    ChangeDecimalToValue(counter, 100),
    run_time=3,
    rate_func=linear
)

self.wait()

# Count down
self.play(
    ChangeDecimalToValue(counter, 0),
    run_time=2
)
""",
        math_concepts=["counting", "animation", "numbers"],
        keywords=["count", "Integer", "decimal", "animation"],
    ),

    AnimationPattern(
        name="path_animation",
        category="effect",
        description="Animate an object following a custom path.",
        code_template="""
# Create a custom path
path = VMobject()
path.set_points_smoothly([
    LEFT * 3,
    LEFT * 2 + UP * 2,
    UP * 2,
    RIGHT * 2 + UP,
    RIGHT * 3,
    RIGHT * 2 + DOWN * 2,
    DOWN * 2,
    LEFT * 2 + DOWN,
    LEFT * 3
])
path.set_color(BLUE)

# Object to animate
dot = Dot(color=YELLOW)
dot.move_to(path.get_start())

# Show path
self.play(ShowCreation(path))

# Animate along path
self.play(MoveAlongPath(dot, path), run_time=4, rate_func=linear)

# Leave a trace
trace = TracedPath(dot.get_center, stroke_color=YELLOW)
self.add(trace)
self.play(MoveAlongPath(dot, path), run_time=4, rate_func=linear)
""",
        math_concepts=["path", "trajectory", "motion"],
        keywords=["MoveAlongPath", "path", "trajectory", "trace"],
    ),

    AnimationPattern(
        name="split_screen",
        category="layout",
        description="Split screen to show comparison side by side.",
        code_template="""
# Dividing line
divider = Line(3*UP, 3*DOWN, color=WHITE)

# Left side content
left_title = Tex("Before").to_edge(UL)
left_content = Square(color=BLUE, fill_opacity=0.5).scale(0.8)
left_content.shift(LEFT * 2)

# Right side content
right_title = Tex("After").to_edge(UR)
right_content = Circle(color=RED, fill_opacity=0.5).scale(0.8)
right_content.shift(RIGHT * 2)

self.play(ShowCreation(divider))
self.play(Write(left_title), Write(right_title))
self.play(ShowCreation(left_content), ShowCreation(right_content))

# Highlight difference
self.play(Indicate(left_content), Indicate(right_content))
""",
        math_concepts=["comparison", "layout", "before-after"],
        keywords=["split", "comparison", "side by side", "layout"],
    ),

    AnimationPattern(
        name="zoom_pan_sequence",
        category="camera",
        description="Zoom and pan camera in sequence.",
        code_template="""
# Create a large scene
grid = VGroup(*[
    VGroup(*[
        Square(side_length=0.8, fill_opacity=0.3).shift(i*RIGHT + j*UP)
        for i in range(-5, 6)
    ])
    for j in range(-3, 4)
])

# Points of interest
poi1 = Dot(np.array([-3, 2, 0]), color=RED, radius=0.2)
poi2 = Dot(np.array([3, -1, 0]), color=BLUE, radius=0.2)

frame = self.camera.frame

self.add(grid, poi1, poi2)

# Start zoomed out
self.play(frame.animate.set_height(10))
self.wait()

# Zoom to first POI
self.play(
    frame.animate.set_height(4).move_to(poi1),
    run_time=2
)
self.wait()

# Pan to second POI
self.play(frame.animate.move_to(poi2), run_time=2)
self.wait()

# Zoom back out
self.play(frame.animate.set_height(10).move_to(ORIGIN), run_time=2)
""",
        math_concepts=["camera", "zoom", "pan", "navigation"],
        keywords=["camera", "zoom", "pan", "frame"],
    ),

    AnimationPattern(
        name="progressive_reveal",
        category="technique",
        description="Reveal content progressively with a sliding mask.",
        code_template="""
# Content to reveal
content = VGroup(
    Tex("Line 1: Important concept"),
    Tex("Line 2: More details"),
    Tex("Line 3: Examples"),
    Tex("Line 4: Summary"),
).arrange(DOWN, aligned_edge=LEFT)

# Initially hidden
content.set_opacity(0)
self.add(content)

# Reveal line by line
for line in content:
    self.play(line.animate.set_opacity(1), run_time=0.5)
    self.wait(0.3)

# Alternatively, use a moving reveal line
reveal_line = Line(4*LEFT, 4*RIGHT, color=YELLOW)
reveal_line.move_to(content.get_top() + UP*0.5)

content.set_opacity(0)
self.add(reveal_line)

def update_opacity(mob):
    for line in content:
        if line.get_center()[1] < reveal_line.get_center()[1]:
            line.set_opacity(1)

content.add_updater(update_opacity)
self.play(reveal_line.animate.move_to(content.get_bottom() + DOWN*0.5), run_time=3)
content.remove_updater(update_opacity)
""",
        math_concepts=["reveal", "progressive", "animation"],
        keywords=["reveal", "mask", "progressive", "fade"],
    ),

    AnimationPattern(
        name="code_typing",
        category="technique",
        description="Simulate code being typed with syntax highlighting.",
        code_template="""
# Code to display
code_text = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)'''

# Create code block (using Text for simplicity)
code = Text(code_text, font="Courier", font_size=24)
code.to_edge(LEFT)

# Type out character by character
displayed = VGroup()
for i, char in enumerate(code_text):
    char_mob = Text(char, font="Courier", font_size=24)
    if char == '\\n':
        continue
    # Position based on original
    char_mob.move_to(code[i].get_center())
    displayed.add(char_mob)

# Show typing
for char in displayed:
    self.play(FadeIn(char), run_time=0.03)

# Cursor blink at end
cursor = Line(UP*0.2, DOWN*0.2, color=WHITE)
cursor.next_to(displayed[-1], RIGHT, buff=0.1)
for _ in range(3):
    self.play(FadeIn(cursor), run_time=0.3)
    self.play(FadeOut(cursor), run_time=0.3)
""",
        math_concepts=["code", "typing", "programming"],
        keywords=["code", "typing", "cursor", "programming"],
    ),

    AnimationPattern(
        name="tree_diagram",
        category="structure",
        description="Animate a tree diagram structure.",
        code_template="""
# Root
root = Dot(UP*2, color=RED)
root_label = Tex("Root").next_to(root, UP)

# Level 1
l1_nodes = VGroup(*[Dot(np.array([x, 0, 0]), color=BLUE) for x in [-2, 0, 2]])

# Level 2
l2_nodes = VGroup(*[
    Dot(np.array([x, -2, 0]), color=GREEN)
    for x in [-3, -1, 1, 3]
])

# Edges
edges = VGroup()
for node in l1_nodes:
    edges.add(Line(root.get_center(), node.get_center(), color=WHITE))

edges.add(Line(l1_nodes[0].get_center(), l2_nodes[0].get_center()))
edges.add(Line(l1_nodes[0].get_center(), l2_nodes[1].get_center()))
edges.add(Line(l1_nodes[2].get_center(), l2_nodes[2].get_center()))
edges.add(Line(l1_nodes[2].get_center(), l2_nodes[3].get_center()))

self.play(FadeIn(root), Write(root_label))
self.play(LaggedStartMap(ShowCreation, edges[:3]), FadeIn(l1_nodes))
self.play(LaggedStartMap(ShowCreation, edges[3:]), FadeIn(l2_nodes))
""",
        math_concepts=["tree", "graph", "hierarchy", "structure"],
        keywords=["tree", "diagram", "nodes", "edges"],
    ),

    AnimationPattern(
        name="flowchart",
        category="structure",
        description="Create and animate a flowchart.",
        code_template="""
# Nodes
start = RoundedRectangle(width=2, height=0.8, color=GREEN, fill_opacity=0.5)
start_text = Tex("Start").move_to(start)
start_group = VGroup(start, start_text).shift(UP*2.5)

decision = Polygon(ORIGIN, RIGHT, DOWN, LEFT, color=YELLOW, fill_opacity=0.5).scale(0.8)
decision.shift(UP*0.5)
decision_text = Tex("?").move_to(decision)
decision_group = VGroup(decision, decision_text)

yes_box = Rectangle(width=1.5, height=0.7, color=BLUE, fill_opacity=0.5).shift(LEFT*2 + DOWN*1.5)
yes_text = Tex("Yes").move_to(yes_box)
yes_group = VGroup(yes_box, yes_text)

no_box = Rectangle(width=1.5, height=0.7, color=RED, fill_opacity=0.5).shift(RIGHT*2 + DOWN*1.5)
no_text = Tex("No").move_to(no_box)
no_group = VGroup(no_box, no_text)

# Arrows
arrows = VGroup(
    Arrow(start.get_bottom(), decision.get_top()),
    Arrow(decision.get_left(), yes_box.get_top()),
    Arrow(decision.get_right(), no_box.get_top()),
)

self.play(FadeIn(start_group))
self.play(ShowCreation(arrows[0]), FadeIn(decision_group))
self.play(ShowCreation(arrows[1]), ShowCreation(arrows[2]))
self.play(FadeIn(yes_group), FadeIn(no_group))
""",
        math_concepts=["flowchart", "decision", "algorithm"],
        keywords=["flowchart", "decision", "arrow", "process"],
    ),

    AnimationPattern(
        name="timeline",
        category="structure",
        description="Create an animated timeline.",
        code_template="""
# Timeline axis
timeline = Line(5*LEFT, 5*RIGHT, color=WHITE)

# Events
events = [
    (-4, "Event 1", BLUE),
    (-1, "Event 2", GREEN),
    (2, "Event 3", RED),
    (4, "Event 4", YELLOW),
]

event_dots = VGroup()
event_labels = VGroup()
event_lines = VGroup()

for x, label, color in events:
    dot = Dot(np.array([x, 0, 0]), color=color)
    line = Line(np.array([x, 0, 0]), np.array([x, 1, 0]), color=color)
    text = Tex(label).scale(0.6).next_to(line, UP)
    event_dots.add(dot)
    event_lines.add(line)
    event_labels.add(text)

self.play(ShowCreation(timeline))

# Animate events appearing in sequence
for dot, line, label in zip(event_dots, event_lines, event_labels):
    self.play(
        FadeIn(dot),
        ShowCreation(line),
        Write(label),
        run_time=0.5
    )
""",
        math_concepts=["timeline", "sequence", "history"],
        keywords=["timeline", "events", "sequence", "chronological"],
    ),

    AnimationPattern(
        name="matrix_multiplication",
        category="linear_algebra",
        description="Visualize matrix multiplication step by step.",
        code_template="""
# Matrices
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
C = Matrix([["?", "?"], ["?", "?"]])

A.shift(LEFT * 3)
B.next_to(A, RIGHT)
equals = Tex("=").next_to(B, RIGHT)
C.next_to(equals, RIGHT)

self.play(Write(A), Write(B), Write(equals), Write(C))
self.wait()

# Highlight row-column multiplication for (0,0) element
row_highlight = SurroundingRectangle(A.get_rows()[0], color=YELLOW)
col_highlight = SurroundingRectangle(B.get_columns()[0], color=YELLOW)

self.play(ShowCreation(row_highlight), ShowCreation(col_highlight))

# Calculate 1*5 + 2*7 = 19
calc = Tex("1·5 + 2·7 = 19").to_edge(DOWN)
self.play(Write(calc))

# Update result matrix
new_C = Matrix([[19, "?"], ["?", "?"]])
new_C.move_to(C)
self.play(Transform(C, new_C))

self.play(FadeOut(row_highlight), FadeOut(col_highlight), FadeOut(calc))
""",
        math_concepts=["matrix multiplication", "linear algebra", "dot product"],
        keywords=["matrix", "multiplication", "row", "column"],
    ),

    AnimationPattern(
        name="neural_network",
        category="ml",
        description="Visualize a simple neural network.",
        code_template="""
# Layer positions
layer_x = [-3, 0, 3]
neurons_per_layer = [3, 4, 2]

# Create neurons
layers = VGroup()
for x, n in zip(layer_x, neurons_per_layer):
    layer = VGroup(*[
        Circle(radius=0.3, color=BLUE, fill_opacity=0.5).shift(np.array([x, (i - n/2 + 0.5) * 1.2, 0]))
        for i in range(n)
    ])
    layers.add(layer)

# Create connections
connections = VGroup()
for l in range(len(layers) - 1):
    for n1 in layers[l]:
        for n2 in layers[l + 1]:
            line = Line(n1.get_center(), n2.get_center(), color=GRAY, stroke_width=1)
            connections.add(line)

self.play(LaggedStartMap(FadeIn, layers, lag_ratio=0.2))
self.play(ShowCreation(connections), run_time=2)

# Animate activation
for layer in layers:
    self.play(*[n.animate.set_fill(YELLOW, opacity=0.8) for n in layer], run_time=0.3)
    self.play(*[n.animate.set_fill(BLUE, opacity=0.5) for n in layer], run_time=0.3)
""",
        math_concepts=["neural network", "machine learning", "layers"],
        keywords=["neural", "network", "layers", "nodes"],
    ),

    # === MORE PATTERNS TO REACH 100+ ===
    AnimationPattern(
        name="gradient_descent",
        category="ml",
        description="Visualize gradient descent on a loss surface.",
        code_template="""
axes = Axes(x_range=[-3, 3], y_range=[0, 10])

# Loss function
loss = lambda x: x**2 + 1
loss_graph = axes.get_graph(loss, color=BLUE)

# Starting point
x = ValueTracker(2.5)
learning_rate = 0.3

dot = always_redraw(lambda: Dot(
    axes.c2p(x.get_value(), loss(x.get_value())), color=YELLOW
))

# Path trace
path = TracedPath(lambda: axes.c2p(x.get_value(), loss(x.get_value())), stroke_color=RED)

self.add(axes, loss_graph, dot, path)

# Gradient descent steps
for _ in range(10):
    current_x = x.get_value()
    gradient = 2 * current_x  # derivative of x^2
    new_x = current_x - learning_rate * gradient
    self.play(x.animate.set_value(new_x), run_time=0.5)

label = Tex("Minimum found!").to_edge(UP)
self.play(Write(label))
""",
        math_concepts=["gradient descent", "optimization", "machine learning"],
        keywords=["gradient", "descent", "optimization", "loss"],
    ),

    AnimationPattern(
        name="regression_line",
        category="ml",
        description="Animate fitting a regression line to data points.",
        code_template="""
axes = Axes(x_range=[0, 10], y_range=[0, 10])

# Data points with some noise
np.random.seed(42)
xs = np.linspace(1, 9, 8)
ys = 0.8 * xs + 1 + np.random.normal(0, 0.5, 8)

dots = VGroup(*[Dot(axes.c2p(x, y), color=BLUE) for x, y in zip(xs, ys)])

# Calculate regression line
m = np.cov(xs, ys)[0, 1] / np.var(xs)
b = np.mean(ys) - m * np.mean(xs)

reg_line = axes.get_graph(lambda x: m*x + b, color=RED)

self.play(ShowCreation(axes))
self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.1))
self.wait()

# Animate line fitting
initial_line = axes.get_graph(lambda x: 5, color=RED)  # horizontal
self.play(ShowCreation(initial_line))
self.play(Transform(initial_line, reg_line), run_time=2)

formula = Tex(f"y = {m:.2f}x + {b:.2f}").to_edge(UP)
self.play(Write(formula))
""",
        math_concepts=["regression", "line fitting", "statistics"],
        keywords=["regression", "line", "fit", "data"],
    ),

    AnimationPattern(
        name="binary_search",
        category="algorithm",
        description="Visualize binary search algorithm.",
        code_template="""
# Sorted array
values = list(range(1, 17))
target = 11

# Create boxes
boxes = VGroup(*[
    VGroup(
        Square(side_length=0.6, fill_color=BLUE, fill_opacity=0.3),
        Integer(v).scale(0.5)
    ).arrange(ORIGIN)
    for v in values
])
boxes.arrange(RIGHT, buff=0.1)

self.play(FadeIn(boxes))

left, right = 0, len(values) - 1
while left <= right:
    mid = (left + right) // 2

    # Highlight search range
    range_rect = SurroundingRectangle(VGroup(*boxes[left:right+1]), color=YELLOW)
    self.play(ShowCreation(range_rect))

    # Check middle
    mid_highlight = boxes[mid][0].copy().set_fill(GREEN, opacity=0.8)
    self.play(FadeIn(mid_highlight))

    if values[mid] == target:
        self.play(Indicate(boxes[mid], color=GREEN))
        break
    elif values[mid] < target:
        left = mid + 1
    else:
        right = mid - 1

    self.play(FadeOut(range_rect), FadeOut(mid_highlight))

result = Tex(f"Found {target} at index {mid}").to_edge(UP)
self.play(Write(result))
""",
        math_concepts=["binary search", "algorithm", "divide and conquer"],
        keywords=["binary", "search", "algorithm", "divide"],
    ),

    AnimationPattern(
        name="sorting_visualization",
        category="algorithm",
        description="Visualize sorting algorithm (bubble sort).",
        code_template="""
# Array to sort
values = [5, 2, 8, 1, 9, 3]
n = len(values)

# Create bars
def create_bars(vals):
    bars = VGroup()
    for i, v in enumerate(vals):
        bar = Rectangle(width=0.5, height=v*0.3, fill_color=BLUE, fill_opacity=0.7)
        bar.move_to(np.array([(i - n/2) * 0.7, v*0.15, 0]))
        bars.add(bar)
    return bars

bars = create_bars(values)
self.play(FadeIn(bars))

# Bubble sort
for i in range(n):
    for j in range(n - i - 1):
        # Highlight comparison
        self.play(
            bars[j].animate.set_fill(RED),
            bars[j+1].animate.set_fill(RED),
            run_time=0.2
        )

        if values[j] > values[j+1]:
            # Swap
            values[j], values[j+1] = values[j+1], values[j]
            self.play(
                bars[j].animate.move_to(bars[j+1].get_center()),
                bars[j+1].animate.move_to(bars[j].get_center()),
                run_time=0.3
            )
            bars[j], bars[j+1] = bars[j+1], bars[j]

        self.play(
            bars[j].animate.set_fill(BLUE),
            bars[j+1].animate.set_fill(BLUE),
            run_time=0.1
        )

# Mark as sorted
self.play(*[b.animate.set_fill(GREEN) for b in bars])
""",
        math_concepts=["sorting", "bubble sort", "algorithm"],
        keywords=["sorting", "bubble", "algorithm", "comparison"],
    ),

    AnimationPattern(
        name="stack_visualization",
        category="data_structure",
        description="Visualize push and pop operations on a stack.",
        code_template="""
# Stack container
container = Rectangle(width=2, height=4, color=WHITE)
container.shift(DOWN * 0.5)
bottom = Line(LEFT, RIGHT, color=WHITE).scale(0.9).next_to(container, DOWN, buff=0)

self.play(ShowCreation(container), ShowCreation(bottom))

stack = []
stack_vis = VGroup()

# Push operations
for val in [3, 7, 1, 5]:
    block = VGroup(
        Rectangle(width=1.8, height=0.6, fill_color=BLUE, fill_opacity=0.7),
        Integer(val)
    ).arrange(ORIGIN)

    y_pos = -1.7 + len(stack) * 0.7
    block.move_to(np.array([0, y_pos, 0]))

    self.play(block.animate.move_to(np.array([0, y_pos, 0])), run_time=0.5)
    stack.append(val)
    stack_vis.add(block)

push_label = Tex("PUSH").to_edge(LEFT)
self.play(Write(push_label))
self.wait()

# Pop operations
pop_label = Tex("POP").to_edge(RIGHT)
self.play(Write(pop_label))

for _ in range(2):
    block = stack_vis[-1]
    self.play(block.animate.shift(RIGHT * 3 + UP), FadeOut(block), run_time=0.5)
    stack_vis.remove(block)
""",
        math_concepts=["stack", "LIFO", "data structure"],
        keywords=["stack", "push", "pop", "LIFO"],
    ),

    AnimationPattern(
        name="linked_list",
        category="data_structure",
        description="Visualize linked list operations.",
        code_template="""
# Node structure
def create_node(value, position):
    box = Rectangle(width=1, height=0.6, fill_color=BLUE, fill_opacity=0.5)
    pointer = Rectangle(width=0.3, height=0.6, fill_color=GREEN, fill_opacity=0.5)
    pointer.next_to(box, RIGHT, buff=0)
    label = Integer(value).move_to(box)
    return VGroup(box, pointer, label).move_to(position)

nodes = VGroup()
positions = [LEFT*3, LEFT, RIGHT, RIGHT*3]
values = [1, 2, 3, 4]

# Create nodes
for val, pos in zip(values, positions):
    node = create_node(val, pos)
    nodes.add(node)

# Create arrows
arrows = VGroup()
for i in range(len(nodes) - 1):
    arrow = Arrow(nodes[i][1].get_center(), nodes[i+1][0].get_left(), buff=0.1)
    arrows.add(arrow)

self.play(LaggedStartMap(FadeIn, nodes, lag_ratio=0.2))
self.play(LaggedStartMap(ShowCreation, arrows, lag_ratio=0.2))

# Insert a new node
new_node = create_node(5, UP*2)
self.play(FadeIn(new_node))
self.play(new_node.animate.move_to((positions[1] + positions[2])/2 + DOWN*0.5))

# Update arrows (simplified)
label = Tex("Insert: 5").to_edge(UP)
self.play(Write(label))
""",
        math_concepts=["linked list", "pointers", "data structure"],
        keywords=["linked", "list", "node", "pointer"],
    ),

    AnimationPattern(
        name="hash_table",
        category="data_structure",
        description="Visualize hash table with collision handling.",
        code_template="""
# Hash table buckets
n_buckets = 5
buckets = VGroup()
bucket_labels = VGroup()

for i in range(n_buckets):
    bucket = Rectangle(width=1.5, height=0.6, color=WHITE)
    bucket.shift(DOWN * i * 0.8)
    label = Integer(i).scale(0.6).next_to(bucket, LEFT)
    buckets.add(bucket)
    bucket_labels.add(label)

buckets.center()
bucket_labels.center().shift(LEFT * 1.5)

self.play(ShowCreation(buckets), Write(bucket_labels))

# Insert values
values = [12, 7, 22, 17]  # 12%5=2, 7%5=2 (collision), 22%5=2, 17%5=2

entries = VGroup()
for val in values:
    hash_val = val % n_buckets

    entry = Tex(str(val)).scale(0.6)
    entry.next_to(buckets[hash_val], RIGHT, buff=0.2 + len([e for e in entries if e.get_center()[1] == buckets[hash_val].get_center()[1]]) * 0.6)

    hash_calc = Tex(f"{val} % {n_buckets} = {hash_val}").to_edge(UP)
    self.play(Write(hash_calc))
    self.play(FadeIn(entry), buckets[hash_val].animate.set_fill(YELLOW, opacity=0.3))
    self.play(FadeOut(hash_calc), buckets[hash_val].animate.set_fill(opacity=0))
    entries.add(entry)
""",
        math_concepts=["hash table", "hashing", "collision"],
        keywords=["hash", "table", "bucket", "collision"],
    ),

    AnimationPattern(
        name="recursion_tree",
        category="algorithm",
        description="Visualize recursion as a call tree.",
        code_template="""
# Fibonacci recursion tree for fib(4)
def create_node(label, pos, color=BLUE):
    circle = Circle(radius=0.3, fill_color=color, fill_opacity=0.5)
    text = Tex(label).scale(0.5)
    return VGroup(circle, text).move_to(pos)

# Tree structure for fib(4)
root = create_node("fib(4)", UP*2)

level1 = VGroup(
    create_node("fib(3)", UP*0.5 + LEFT*2),
    create_node("fib(2)", UP*0.5 + RIGHT*2),
)

level2 = VGroup(
    create_node("fib(2)", DOWN + LEFT*3),
    create_node("fib(1)", DOWN + LEFT),
    create_node("fib(1)", DOWN + RIGHT),
    create_node("fib(0)", DOWN + RIGHT*3),
)

edges = VGroup(
    Line(root.get_bottom(), level1[0].get_top()),
    Line(root.get_bottom(), level1[1].get_top()),
    Line(level1[0].get_bottom(), level2[0].get_top()),
    Line(level1[0].get_bottom(), level2[1].get_top()),
    Line(level1[1].get_bottom(), level2[2].get_top()),
    Line(level1[1].get_bottom(), level2[3].get_top()),
)

self.play(FadeIn(root))
self.play(ShowCreation(edges[:2]), FadeIn(level1))
self.play(ShowCreation(edges[2:]), FadeIn(level2))

# Highlight base cases
base_cases = VGroup(level2[1], level2[2], level2[3])
self.play(*[n[0].animate.set_fill(GREEN) for n in base_cases])
""",
        math_concepts=["recursion", "tree", "fibonacci", "call stack"],
        keywords=["recursion", "tree", "call", "fibonacci"],
    ),

    AnimationPattern(
        name="cross_product",
        category="vectors",
        description="Visualize cross product of two 3D vectors.",
        code_template="""
# 3D scene setup
axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3])

# Two vectors
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
c = np.cross(a, b)  # [0, 0, 1]

vec_a = Arrow3D(ORIGIN, a, color=RED)
vec_b = Arrow3D(ORIGIN, b, color=BLUE)
vec_c = Arrow3D(ORIGIN, c, color=GREEN)

# Labels
a_label = Tex("a").next_to(vec_a.get_end(), RIGHT)
b_label = Tex("b").next_to(vec_b.get_end(), UP)
c_label = Tex("a × b").next_to(vec_c.get_end(), OUT)

self.set_camera_orientation(phi=60*DEGREES, theta=45*DEGREES)
self.play(ShowCreation(axes))
self.play(ShowCreation(vec_a), Write(a_label))
self.play(ShowCreation(vec_b), Write(b_label))
self.wait()

# Show cross product
self.play(ShowCreation(vec_c), Write(c_label))

# Show it's perpendicular
self.begin_ambient_camera_rotation(rate=0.2)
self.wait(5)
""",
        math_concepts=["cross product", "3D vectors", "perpendicular"],
        keywords=["cross", "product", "3D", "perpendicular"],
    ),

    AnimationPattern(
        name="unit_circle_trig",
        category="trigonometry",
        description="Animate unit circle with sin/cos values.",
        code_template="""
# Unit circle
circle = Circle(radius=2, color=WHITE)
axes = Axes(x_range=[-2.5, 2.5], y_range=[-2.5, 2.5])

# Angle tracker
theta = ValueTracker(0)

# Point on circle
point = always_redraw(lambda: Dot(
    2 * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0]),
    color=YELLOW
))

# Radius line
radius = always_redraw(lambda: Line(
    ORIGIN,
    2 * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0]),
    color=GREEN
))

# Projections
cos_line = always_redraw(lambda: Line(
    2 * np.array([np.cos(theta.get_value()), 0, 0]),
    2 * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0]),
    color=BLUE
))
sin_line = always_redraw(lambda: Line(
    ORIGIN,
    2 * np.array([np.cos(theta.get_value()), 0, 0]),
    color=RED
))

# Labels
cos_label = always_redraw(lambda: Tex(f"cos = {np.cos(theta.get_value()):.2f}").to_edge(UL))
sin_label = always_redraw(lambda: Tex(f"sin = {np.sin(theta.get_value()):.2f}").next_to(cos_label, DOWN))

self.add(axes, circle, radius, point, cos_line, sin_line, cos_label, sin_label)

# Rotate around
self.play(theta.animate.set_value(2*PI), run_time=6, rate_func=linear)
""",
        math_concepts=["unit circle", "trigonometry", "sin", "cos"],
        keywords=["unit circle", "trig", "sin", "cos"],
    ),

    AnimationPattern(
        name="power_series",
        category="calculus",
        description="Visualize power series convergence radius.",
        code_template="""
axes = Axes(x_range=[-3, 3], y_range=[-2, 4])

# 1/(1-x) = 1 + x + x^2 + x^3 + ...
target = axes.get_graph(lambda x: 1/(1-x) if abs(x) < 0.99 else None, color=WHITE, x_range=[-2, 0.95])

# Partial sums
def partial_sum(n):
    def f(x):
        return sum(x**k for k in range(n+1))
    return f

colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
approximations = [axes.get_graph(partial_sum(n), color=colors[i % len(colors)], x_range=[-1.5, 0.9]) for i, n in enumerate([1, 2, 3, 5, 10])]

self.play(ShowCreation(axes))
self.play(ShowCreation(target))

# Show radius of convergence
radius_line = DashedLine(axes.c2p(-1, -2), axes.c2p(-1, 4), color=RED)
radius_label = Tex("|x| < 1").next_to(radius_line, LEFT)

self.play(ShowCreation(radius_line), Write(radius_label))

current = approximations[0]
self.play(ShowCreation(current))

for approx in approximations[1:]:
    self.play(Transform(current, approx))
    self.wait(0.3)
""",
        math_concepts=["power series", "convergence", "radius", "calculus"],
        keywords=["power series", "convergence", "radius", "sum"],
    ),

    # =============================================================================
    # ADVANCED 3B1B PATTERNS - Reactive/Declarative Style
    # =============================================================================

    AnimationPattern(
        name="value_tracker_reactive",
        category="updater",
        description="""
The signature 3b1b pattern for smooth, continuous animations. Use ValueTracker
to control a parameter, then add updaters to mobjects that react to changes.
This creates buttery-smooth animations where everything updates together.

Key insight: Define the END STATE declaratively, let updaters handle the animation.
""",
        code_template="""
# ValueTracker controls a parameter (e.g., time, x-position, slider value)
t_tracker = ValueTracker(0)
get_t = t_tracker.get_value

# Create mobjects that depend on the tracker
dot = Dot(color=YELLOW)
dot.add_updater(lambda m: m.move_to(axes.c2p(get_t(), func(get_t()))))

label = DecimalNumber(0, num_decimal_places=2)
label.add_updater(lambda m: m.set_value(get_t()))
label.add_updater(lambda m: m.next_to(dot, UP))

# The graph traces the path
trace = TracedPath(dot.get_center, stroke_color=BLUE, stroke_width=2)

self.add(dot, label, trace)

# Animate by changing the tracker - everything updates automatically!
self.play(t_tracker.animate.set_value(5), run_time=4, rate_func=linear)
""",
        math_concepts=["calculus", "continuous", "parametric", "dynamics"],
        keywords=["ValueTracker", "updater", "reactive", "smooth", "continuous", "TracedPath"],
    ),

    AnimationPattern(
        name="reactive_area_integral",
        category="updater",
        description="""
Show area under a curve that updates reactively as parameters change.
Use .become() inside an updater to recreate the area each frame.
This is how 3b1b shows integrals that change with sliders or time.
""",
        code_template="""
# Parameter tracker (e.g., upper bound of integral, or function parameter)
s_tracker = ValueTracker(1)
get_s = s_tracker.get_value

# Graph that reacts to s
def func(t):
    return np.exp(-get_s() * t)

graph = axes.get_graph(lambda t: 1)  # Initial
axes.bind_graph_to_func(graph, func)  # Graph auto-updates with func!

# Area that recreates itself each frame
area = axes.get_area_under_graph(graph, x_range=[0, 3])
def update_area(area):
    area.become(axes.get_area_under_graph(graph, x_range=[0, 3]))
area.add_updater(update_area)

self.add(graph, area)

# Animate s changing - graph AND area update together smoothly
self.play(s_tracker.animate.set_value(3), run_time=4)
self.play(s_tracker.animate.set_value(0.5), run_time=3)
""",
        math_concepts=["integral", "area", "calculus", "continuous"],
        keywords=["become", "updater", "area", "integral", "reactive", "bind_graph_to_func"],
    ),

    AnimationPattern(
        name="camera_reorient_zoom",
        category="camera",
        description="""
3b1b uses camera movements to guide attention. The frame object can be
animated to zoom, pan, and reorient smoothly. Use time_span for
choreographed camera moves during longer animations.
""",
        code_template="""
# Get reference to the camera frame
frame = self.frame  # or self.camera.frame in older versions

# Zoom in on a specific point
self.play(
    frame.animate.set_height(4).move_to(point_of_interest),
    run_time=2
)

# Pan across while something else animates
self.play(
    some_animation,
    frame.animate.shift(3 * RIGHT),
    run_time=3
)

# 3D reorientation (for ThreeDScene)
self.play(
    frame.animate.reorient(20, 70, 0),  # theta, phi, gamma angles
    run_time=2
)

# Delayed camera move using time_span (starts at t=2, ends at t=5)
self.play(
    long_animation,  # runs full duration
    frame.animate.reorient(0, 0, 0, (5, 3, 0), 12).set_anim_args(time_span=(2, 5)),
    run_time=6
)

# Reset to default view
self.play(frame.animate.to_default_state())
""",
        math_concepts=["visualization", "presentation", "3d"],
        keywords=["frame", "camera", "zoom", "reorient", "time_span", "pan"],
    ),

    AnimationPattern(
        name="transform_matching_tex",
        category="transform",
        description="""
Intelligently morph between equations by matching corresponding tex parts.
Much smoother than Transform for equations - parts that match stay in place,
parts that differ smoothly morph. Essential for equation derivations.
""",
        code_template="""
# Define equations with consistent tex structure
eq1 = Tex(R"\\int_0^\\infty", R"e^{-st}", R"dt")
eq2 = Tex(R"\\int_0^\\infty", R"e^{-st}", R"dt", R"=", R"\\frac{1}{s}")

# TransformMatchingTex matches by tex content automatically
self.play(TransformMatchingTex(eq1, eq2))

# For more control, use TransformMatchingShapes with key_map
eq3 = Tex(R"x^2 + 2x + 1")
eq4 = Tex(R"(x+1)^2")
self.play(TransformMatchingShapes(eq3, eq4))

# Color-coded transformation
t2c = {R"{s}": YELLOW, R"e": BLUE}  # tex-to-color map
eq_colored = Tex(R"e^{-{s}t}", t2c=t2c)
self.play(TransformMatchingTex(eq1.copy(), eq_colored))
""",
        math_concepts=["algebra", "equation", "derivation", "proof"],
        keywords=["TransformMatchingTex", "TransformMatchingShapes", "equation", "morph", "t2c"],
    ),

    AnimationPattern(
        name="interactive_slider",
        category="updater",
        description="""
Add an interactive slider that controls a parameter. The slider displays
the current value and can be animated. Great for exploring parameter spaces.
""",
        code_template="""
# Create a value tracker for the parameter
s_tracker = ValueTracker(1)

# Create slider UI element
s_slider = Slider(
    s_tracker,
    x_range=(0, 5, 0.1),  # min, max, step
    var_name="s",  # displays as "s = 1.00"
)
s_slider.scale(1.5)
s_slider.to_edge(UP)
s_slider.fix_in_frame()  # Stays fixed even when camera moves

self.add(s_slider)

# Animate the slider
self.play(s_tracker.animate.set_value(4), run_time=3)
self.play(s_tracker.animate.set_value(0.5), run_time=2)

# Other mobjects react to s_tracker via updaters
graph.add_updater(lambda m: m.become(
    axes.get_graph(lambda x: np.sin(s_tracker.get_value() * x))
))
""",
        math_concepts=["parameter", "interactive", "exploration"],
        keywords=["Slider", "ValueTracker", "fix_in_frame", "interactive", "parameter"],
    ),

    AnimationPattern(
        name="decimal_number_counter",
        category="updater",
        description="""
Animated number display that counts up/down smoothly. Use DecimalNumber
with updaters for live-updating values. Essential for showing calculations.
""",
        code_template="""
# Create a tracker for the value
value_tracker = ValueTracker(0)

# DecimalNumber that updates automatically
decimal = DecimalNumber(0, num_decimal_places=2, font_size=72)
decimal.add_updater(lambda m: m.set_value(value_tracker.get_value()))

# Position it (can also add position updater)
decimal.to_edge(UP)
decimal.fix_in_frame()  # Optional: stays fixed during camera moves

self.add(decimal)

# Animate counting up
self.play(value_tracker.animate.set_value(100), run_time=4, rate_func=linear)

# For integers, use num_decimal_places=0
integer_display = Integer(0)
integer_display.add_updater(lambda m: m.set_value(int(value_tracker.get_value())))

# Make changeable numbers in equations
equation = Tex(R"\\int_0^{0.01} 1 \\, dt = 0.01", font_size=72)
decimals = equation.make_number_changeable("0.01", replace_all=True)
for dec in decimals:
    dec.add_updater(lambda m: m.set_value(value_tracker.get_value()))
""",
        math_concepts=["counting", "calculation", "display"],
        keywords=["DecimalNumber", "Integer", "make_number_changeable", "counter", "updater"],
    ),

    AnimationPattern(
        name="fix_in_frame_ui",
        category="camera",
        description="""
Keep UI elements (labels, equations, sliders) fixed in place even when
the camera moves or zooms. Essential for educational animations where
you need persistent annotations.
""",
        code_template="""
# Create UI elements
title = Text("Integration Demo").to_edge(UP)
equation = Tex(R"\\int f(x) dx").to_corner(UR)
slider = Slider(tracker, x_range=(0, 5)).to_edge(DOWN)

# Fix them in the frame (won't move with camera)
title.fix_in_frame()
equation.fix_in_frame()
slider.fix_in_frame()

self.add(title, equation, slider)

# Now camera can move freely without affecting UI
self.play(
    self.frame.animate.scale(0.5).move_to(detail_point),
    run_time=2
)
# title, equation, slider stay in their screen positions!

# To unfix:
title.unfix_from_frame()
""",
        math_concepts=["visualization", "presentation", "ui"],
        keywords=["fix_in_frame", "unfix_from_frame", "UI", "camera", "label"],
    ),

    AnimationPattern(
        name="lagged_start_professional",
        category="sequence",
        description="""
Professional staggered animations with precise timing control.
Use LaggedStart with lag_ratio for cascading effects, and time_span
for choreographed multi-part animations.
""",
        code_template="""
# Basic LaggedStart - each starts before previous finishes
items = VGroup(*[Square() for _ in range(5)]).arrange(RIGHT)
self.play(LaggedStart(*[FadeIn(item, shift=UP) for item in items], lag_ratio=0.2))

# LaggedStartMap - cleaner syntax for same animation on each item
self.play(LaggedStartMap(FadeIn, items, lag_ratio=0.15))

# Cascading with different animations
self.play(LaggedStart(
    ShowCreation(line1),
    Write(label1),
    ShowCreation(line2),
    Write(label2),
    lag_ratio=0.3
))

# Time spans for precise choreography in long animations
self.play(
    animation1,  # plays full duration
    animation2.set_anim_args(time_span=(0, 2)),    # plays t=0 to t=2
    animation3.set_anim_args(time_span=(1.5, 4)),  # plays t=1.5 to t=4
    animation4.set_anim_args(time_span=(3, 5)),    # plays t=3 to t=5
    run_time=5
)
""",
        math_concepts=["animation", "timing", "sequence"],
        keywords=["LaggedStart", "LaggedStartMap", "lag_ratio", "time_span", "choreography"],
    ),

    AnimationPattern(
        name="always_redraw_dynamic",
        category="updater",
        description="""
For complex objects that need complete reconstruction each frame,
use always_redraw. Simpler than manual updaters for derived geometry.
""",
        code_template="""
# Dot that moves along a path
t_tracker = ValueTracker(0)
dot = Dot(color=YELLOW)
dot.add_updater(lambda m: m.move_to(axes.c2p(t_tracker.get_value(), 0)))

# Tangent line that's always redrawn based on dot position
tangent = always_redraw(lambda: axes.get_secant_slope_group(
    x=t_tracker.get_value(),
    graph=graph,
    dx=0.01,
    secant_line_length=3,
))

# Vertical line from x-axis to curve
v_line = always_redraw(lambda: Line(
    axes.c2p(t_tracker.get_value(), 0),
    axes.c2p(t_tracker.get_value(), func(t_tracker.get_value())),
    color=YELLOW
))

# Area that changes with upper bound
area = always_redraw(lambda: axes.get_area_under_graph(
    graph,
    x_range=[0, t_tracker.get_value()],
    fill_opacity=0.5
))

self.add(dot, tangent, v_line, area)
self.play(t_tracker.animate.set_value(4), run_time=5)
""",
        math_concepts=["calculus", "geometry", "dynamic"],
        keywords=["always_redraw", "dynamic", "tangent", "derivative", "area"],
    ),

    AnimationPattern(
        name="traced_path_trajectory",
        category="updater",
        description="""
Draw the path that a moving object traces. Perfect for showing
trajectories, parametric curves being drawn, or phase space orbits.
""",
        code_template="""
# Moving dot controlled by tracker
t_tracker = ValueTracker(0)
dot = Dot(color=YELLOW)
dot.add_updater(lambda m: m.move_to(
    axes.c2p(np.cos(t_tracker.get_value()), np.sin(t_tracker.get_value()))
))

# TracedPath follows the dot and draws its trail
trace = TracedPath(
    dot.get_center,
    stroke_color=BLUE,
    stroke_width=3,
    stroke_opacity=[0, 1],  # Fades in (gradient)
)

# Or with fading trail
fading_trace = TracedPath(
    dot.get_center,
    stroke_color=BLUE,
    stroke_width=2,
    dissipating_time=0.5,  # Trail fades after 0.5 seconds
)

self.add(dot, trace)
self.play(t_tracker.animate.set_value(2 * PI), run_time=4, rate_func=linear)
""",
        math_concepts=["parametric", "trajectory", "phase space", "orbit"],
        keywords=["TracedPath", "trajectory", "trail", "path", "parametric"],
    ),

    AnimationPattern(
        name="transform_from_copy",
        category="transform",
        description="""
Create a copy of one object and transform it into another.
Useful for showing derivations where the original stays in place.
""",
        code_template="""
# Original equation stays in place
original = Tex(R"e^{i\\pi}").to_edge(UP)
result = Tex(R"= -1").next_to(original, RIGHT)

self.play(Write(original))

# Transform a COPY into the result (original stays)
self.play(TransformFromCopy(original, result))

# Multiple copies spreading out
source = Tex(R"f(x)")
targets = VGroup(*[
    Tex(text).move_to(pos)
    for text, pos in [("f'(x)", UP), ("\\int f", DOWN), ("f^{-1}", RIGHT)]
])

self.play(LaggedStart(*[
    TransformFromCopy(source, target)
    for target in targets
], lag_ratio=0.3))
""",
        math_concepts=["transformation", "derivation", "copy"],
        keywords=["TransformFromCopy", "copy", "derivation", "spread"],
    ),

    AnimationPattern(
        name="indicate_and_flash",
        category="highlight",
        description="""
Draw attention to specific elements with indication animations.
3b1b uses these to guide viewer focus during explanations.
""",
        code_template="""
# Indicate - scales up and changes color temporarily
self.play(Indicate(equation, color=YELLOW, scale_factor=1.2))

# FlashAround - draws attention with animated border
self.play(FlashAround(important_region, color=YELLOW, stroke_width=4))

# Circumscribe - draws a shape around the object
self.play(Circumscribe(equation, color=RED, shape=Rectangle))

# ShowPassingFlash - light travels along a path
self.play(ShowPassingFlash(
    curve.copy().set_stroke(YELLOW, 5),
    time_width=0.3,
    run_time=2
))

# Wiggle - shakes object for emphasis
self.play(Wiggle(wrong_answer, scale_value=1.1, rotation_angle=0.05))

# ApplyWave - wave passes through object
self.play(ApplyWave(text, amplitude=0.3))

# Flash - quick flash at a point
self.play(Flash(point, color=WHITE, flash_radius=0.5))
""",
        math_concepts=["visualization", "emphasis", "attention"],
        keywords=["Indicate", "FlashAround", "Circumscribe", "ShowPassingFlash", "highlight"],
    ),

    AnimationPattern(
        name="number_line_with_tracker",
        category="updater",
        description="""
Animated number line with a moving indicator. Classic 3b1b style
for showing values changing on a scale.
""",
        code_template="""
# Create number line
number_line = NumberLine(x_range=[-3, 3, 1], length=10, include_numbers=True)

# Value tracker
x_tracker = ValueTracker(0)

# Dot on the number line
dot = Dot(color=YELLOW)
dot.add_updater(lambda m: m.move_to(number_line.n2p(x_tracker.get_value())))

# Label showing value
label = DecimalNumber(0, num_decimal_places=2)
label.add_updater(lambda m: m.set_value(x_tracker.get_value()))
label.add_updater(lambda m: m.next_to(dot, UP))

# Triangle pointer
pointer = Triangle(fill_opacity=1, fill_color=YELLOW).scale(0.2)
pointer.add_updater(lambda m: m.next_to(dot, DOWN, buff=0))

self.add(number_line, dot, label, pointer)
self.play(x_tracker.animate.set_value(2), run_time=2)
self.play(x_tracker.animate.set_value(-1.5), run_time=2)
""",
        math_concepts=["number line", "value", "scale"],
        keywords=["NumberLine", "ValueTracker", "n2p", "pointer", "indicator"],
    ),

    AnimationPattern(
        name="riemann_sum_reactive",
        category="updater",
        description="""
ADVANCED Riemann sum with reactive rectangles. The number of rectangles
updates smoothly as a parameter changes. This is the professional 3b1b style.
""",
        code_template="""
# Tracker for number of rectangles (can be continuous!)
n_tracker = ValueTracker(4)

# Reactive rectangles using always_redraw
rects = always_redraw(lambda: axes.get_riemann_rectangles(
    graph,
    x_range=[a, b],
    dx=(b - a) / max(1, n_tracker.get_value()),
    colors=[BLUE, GREEN],
    fill_opacity=0.7,
))

# Label showing current n
n_label = always_redraw(lambda: Tex(
    f"n = {int(n_tracker.get_value())}"
).to_corner(UR))

# Sum approximation
sum_value = always_redraw(lambda: DecimalNumber(
    sum(func(a + i * (b-a)/n_tracker.get_value()) * (b-a)/n_tracker.get_value()
        for i in range(int(n_tracker.get_value()))),
    num_decimal_places=3
).to_corner(DR))

self.add(rects, n_label, sum_value)

# Smooth animation from 4 to 64 rectangles
self.play(n_tracker.animate.set_value(64), run_time=6, rate_func=linear)

# Can also use discrete steps
for n in [4, 8, 16, 32]:
    self.play(n_tracker.animate.set_value(n), run_time=1)
    self.wait(0.5)
""",
        math_concepts=["integral", "riemann sum", "calculus", "approximation"],
        keywords=["always_redraw", "get_riemann_rectangles", "reactive", "n_tracker"],
    ),

    # === INTRO/OUTRO PATTERNS ===
    AnimationPattern(
        name="opening_quote",
        category="intro",
        description="""
Display an opening quote with attribution, fading in with a cinematic feel.
Classic 3b1b video intro style.
""",
        code_template="""
# Opening quote scene
quote = Tex(r'"Mathematics is the language of nature."')
quote.to_edge(UP, buff=1)

author = Tex("-- Richard Feynman", color=YELLOW)
author.next_to(quote, DOWN, buff=0.5)

self.play(FadeIn(quote, run_time=3, rate_func=linear))
self.wait(0.5)
self.play(Write(author, run_time=2))
self.wait(2)
self.play(FadeOut(quote), FadeOut(author))
""",
        math_concepts=["introduction", "quote", "attribution"],
        keywords=["opening", "quote", "intro", "FadeIn", "cinematic"],
    ),

    AnimationPattern(
        name="end_screen_credits",
        category="outro",
        description="""
Create an end screen with scrolling patron credits.
Includes thanks message and support links.
""",
        code_template="""
# End screen with scrolling names
title = Text("Thanks to Patrons", font_size=48)
title.to_edge(UP)

# Name list (example)
names = ["Patron 1", "Patron 2", "Patron 3"]
name_mobjects = VGroup(*[Text(name, font_size=24) for name in names])
name_mobjects.arrange(DOWN, buff=0.2)
name_mobjects.next_to(title, DOWN, buff=1)

self.play(Write(title))
self.wait(0.5)

# Scroll names
self.play(
    name_mobjects.animate.shift(UP * len(names) * 0.4),
    run_time=len(names) * 0.3,
    rate_func=linear
)
""",
        math_concepts=["credits", "scrolling", "outro"],
        keywords=["end screen", "credits", "patron", "scroll", "outro"],
    ),

    AnimationPattern(
        name="video_banner",
        category="banner",
        description="""
Create a video banner/thumbnail with title and characters.
High-resolution branding for video previews.
""",
        code_template="""
# Banner with title and pi creatures
# Note: Use InteractiveScene or high-res camera

title = Text("Video Title", font_size=72)
title.to_edge(UP, buff=0.5)

# Subtitle or date
subtitle = Text("Episode 1", font_size=36, color=GRAY)
subtitle.next_to(title, DOWN)

# Background elements
grid = NumberPlane(
    x_range=[-10, 10], y_range=[-6, 6],
    background_line_style={"stroke_opacity": 0.3}
)

self.add(grid)
self.play(Write(title), FadeIn(subtitle, shift=UP))
# Add pi creatures or other branding elements
""",
        math_concepts=["branding", "thumbnail", "title"],
        keywords=["banner", "thumbnail", "title", "branding"],
    ),

    # === CHARACTER PATTERNS ===
    AnimationPattern(
        name="pi_creature_intro",
        category="character",
        description="""
Introduce a pi creature character with a greeting animation.
The creature waves, blinks, and displays a speech bubble.
Requires PiCreature from 3b1b's custom library.
""",
        code_template="""
# Pi creature greeting (requires PiCreature from 3b1b library)
# from custom.characters.pi_creature import PiCreature

pi = PiCreature(color=BLUE)
pi.to_edge(DOWN)

# Wave animation with speech bubble
self.play(PiCreatureSays(
    pi,
    "Hello!",
    bubble_type="speech",
    target_mode="happy"
))
self.wait()

# Thinking pose
self.play(pi.change_mode("pondering"))
self.wait()

# Return to normal
self.play(pi.change_mode("plain"))
""",
        math_concepts=["character", "mascot", "introduction"],
        keywords=["pi creature", "character", "wave", "speech bubble", "greeting"],
    ),

    AnimationPattern(
        name="pi_creature_explanation",
        category="character",
        description="""
Pi creature explaining a concept with gestures and expressions.
Points at mathematical objects and changes mood.
Requires PiCreature from 3b1b's custom library.
""",
        code_template="""
# Pi creature explaining (requires PiCreature)
pi = PiCreature(color=BLUE)
pi.to_corner(DL)

# Math object to explain
equation = Tex("E = mc^2").to_edge(UP)

# Look at the equation
pi.look_at(equation)
self.play(
    ShowCreation(equation),
    pi.change_mode("thinking")
)
self.wait()

# Point at specific part
self.play(
    pi.change_mode("speaking"),
    Indicate(equation[2])  # Point at m
)

# Express understanding
self.play(pi.change_mode("hooray"))
""",
        math_concepts=["explanation", "teaching", "character"],
        keywords=["pi creature", "explain", "look_at", "point", "expression"],
    ),

    AnimationPattern(
        name="teacher_students_scene",
        category="character",
        description="""
A teacher pi creature with multiple student pi creatures.
Classic 3b1b educational scene layout.
""",
        code_template="""
# TeacherStudentsScene layout (from 3b1b custom)
# Teacher on the right, students on the left

teacher = PiCreature(color=GREY_BROWN)
teacher.to_edge(RIGHT)
teacher.flip()

students = VGroup(*[
    PiCreature(color=color).scale(0.6)
    for color in [BLUE_D, BLUE_E, BLUE_C]
])
students.arrange(RIGHT, buff=0.5)
students.to_edge(LEFT)

# All look at teacher
for student in students:
    student.look_at(teacher)

self.add(teacher, students)

# Teacher speaks
self.play(teacher.change_mode("speaking"))
self.wait()

# Students react
self.play(*[
    s.change_mode("thinking") for s in students
])
""",
        math_concepts=["teaching", "education", "classroom"],
        keywords=["teacher", "students", "pi creature", "classroom", "education"],
    ),
]


def get_patterns_by_category(category: str) -> list[AnimationPattern]:
    """Get all patterns in a category."""
    return [p for p in ANIMATION_PATTERNS if p.category == category]


def get_patterns_for_concepts(concepts: list[str]) -> list[AnimationPattern]:
    """Get patterns matching given math concepts."""
    concepts_lower = [c.lower() for c in concepts]
    matching = []
    for pattern in ANIMATION_PATTERNS:
        pattern_concepts = [c.lower() for c in pattern.math_concepts]
        if any(c in pattern_concepts or any(c in pc for pc in pattern_concepts)
               for c in concepts_lower):
            matching.append(pattern)
    return matching


def get_patterns_for_keywords(keywords: list[str]) -> list[AnimationPattern]:
    """Get patterns matching given code keywords."""
    keywords_lower = [k.lower() for k in keywords]
    matching = []
    for pattern in ANIMATION_PATTERNS:
        pattern_keywords = [k.lower() for k in pattern.keywords]
        if any(k in pattern_keywords or any(k in pk for pk in pattern_keywords)
               for k in keywords_lower):
            matching.append(pattern)
    return matching


def detect_pattern_from_prompt(prompt: str) -> list[AnimationPattern]:
    """Detect which patterns are relevant for a given prompt."""
    prompt_lower = prompt.lower()

    matching = []

    # Check for specific pattern triggers
    pattern_triggers = {
        # Progression patterns
        "riemann_sum_convergence": ["riemann", "rectangles", "approximate area", "integral approximation", "sum converge"],
        "series_partial_sums": ["series", "partial sum", "convergence", "infinite sum"],

        # Transform patterns (ADVANCED 3B1B STYLE)
        "transform_matching_tex": ["equation morph", "equation transform", "algebraic", "rearrange", "simplify", "derivation"],
        "transform_from_copy": ["copy and transform", "derivation", "original stays", "spread out"],
        "morph_graph_to_graph": ["morph", "transform graph", "change function"],

        # Updater patterns (ADVANCED 3B1B STYLE - prioritize these!)
        "value_tracker_reactive": ["continuous", "smooth", "parameter", "slider", "dynamic", "animate", "change", "varying", "moving"],
        "reactive_area_integral": ["area under", "integral", "area changes", "accumulate", "shade"],
        "always_redraw_dynamic": ["redraw", "dynamic object", "updating", "depends on", "follows"],
        "traced_path_trajectory": ["trajectory", "trace", "path", "trail", "orbit", "follows path"],
        "decimal_number_counter": ["counter", "counting", "display value", "show number", "numerical"],
        "number_line_with_tracker": ["number line", "scale", "indicator", "value on line"],
        "riemann_sum_reactive": ["riemann", "rectangles converge", "dx", "n rectangles"],

        # Sequence patterns (ADVANCED 3B1B STYLE)
        "lagged_start_professional": ["one by one", "staggered", "sequence", "reveal", "cascade", "choreograph"],
        "build_up_construction": ["step by step", "construct", "build up"],
        "interactive_slider": ["slider", "interactive", "explore parameter", "adjust"],

        # Camera patterns (ADVANCED 3B1B STYLE)
        "camera_reorient_zoom": ["zoom", "close up", "detail", "magnify", "pan", "camera", "focus on"],
        "fix_in_frame_ui": ["ui", "fixed label", "persistent", "annotation stays"],

        # Highlight patterns (ADVANCED 3B1B STYLE)
        "indicate_and_flash": ["highlight", "flash", "emphasize", "attention", "indicate", "circumscribe", "wiggle"],

        # Calculus patterns
        "tangent_line_animation": ["tangent", "derivative", "slope at point", "instantaneous"],
        "limit_visualization": ["limit", "approach", "continuous", "discontinuous"],
        "area_accumulation": ["accumulate", "integral", "ftc", "fundamental theorem"],
        "derivative_definition": ["secant", "limit definition", "h approaches"],

        # Probability patterns
        "probability_distribution": ["probability", "distribution", "gaussian", "normal", "histogram"],
        "random_walk": ["random walk", "brownian", "stochastic", "coin flip"],

        # Complex number patterns
        "complex_plane_multiplication": ["complex number", "complex plane", "multiply complex"],
        "euler_formula": ["euler", "e^i", "unit circle", "complex exponential"],

        # Vector patterns
        "vector_addition": ["vector addition", "head to tail", "parallelogram"],
        "dot_product_projection": ["dot product", "projection", "inner product"],

        # Equation patterns
        "equation_derivation": ["derivation", "proof", "algebraic steps", "expand"],
        "text_highlight": ["highlight text", "annotate", "color equation"],

        # 3D patterns
        "surface_plot_3d": ["surface", "3d plot", "z = f(x,y)", "multivariable"],
        "vector_field_3d": ["vector field", "3d field", "flow"],

        # Wave patterns
        "standing_wave": ["standing wave", "nodes", "antinodes", "resonance"],
        "traveling_wave": ["traveling wave", "propagation", "wavelength"],

        # Fourier patterns
        "fourier_series_approximation": ["fourier", "harmonic", "square wave", "approximation"],

        # Graph theory patterns
        "graph_traversal": ["graph", "bfs", "dfs", "traversal", "tree"],

        # Physics patterns
        "pendulum_motion": ["pendulum", "swing", "harmonic motion"],
        "projectile_motion": ["projectile", "trajectory", "throw", "launch"],

        # Number theory patterns
        "prime_sieve": ["prime", "sieve", "eratosthenes", "factor"],

        # Geometry patterns
        "triangle_construction": ["triangle", "vertices", "construct triangle"],
        "circle_theorem": ["circle theorem", "inscribed angle", "central angle"],
        "rotating_shape": ["rotate", "rotation", "symmetry"],

        # Linear algebra patterns
        "matrix_transformation": ["matrix", "linear transformation", "transform plane", "eigenvector", "linear algebra"],

        # Data visualization patterns
        "bar_chart_animation": ["bar chart", "histogram", "bars", "data"],
        "pie_chart_animation": ["pie chart", "sectors", "proportions"],

        # More calculus patterns
        "chain_rule_visualization": ["chain rule", "composition", "nested function"],
        "integration_by_parts": ["integration by parts", "uv", "parts"],

        # More linear algebra patterns
        "eigenvalue_eigenvector": ["eigenvalue", "eigenvector", "eigen"],
        "determinant_area": ["determinant", "area scaling", "det"],

        # Combinatorics patterns
        "pascals_triangle": ["pascal", "binomial", "triangle", "coefficients"],
        "permutation_cycle": ["permutation", "cycle", "group"],

        # More geometry patterns
        "pythagorean_theorem": ["pythagorean", "right triangle", "a² + b²"],
        "similar_triangles": ["similar triangles", "proportion", "scaling"],

        # Animation technique patterns
        "fade_transition": ["fade", "transition", "cross-fade"],
        "color_wave": ["color wave", "ripple", "wave effect"],
        "typewriter_text": ["typewriter", "text reveal", "letter by letter"],
        "morphing_shape": ["morph", "shape transform", "metamorphosis"],

        # More physics patterns
        "spring_mass_system": ["spring", "oscillator", "spring-mass"],
        "electric_field_lines": ["electric field", "field lines", "charge"],

        # Signal processing patterns
        "convolution_animation": ["convolution", "signal", "overlap"],

        # Recursion patterns
        "fractal_tree": ["fractal", "tree", "branching", "recursive"],
        "fibonacci_spiral": ["fibonacci", "spiral", "golden ratio"],

        # Set theory patterns
        "venn_diagram": ["venn", "sets", "union", "intersection"],

        # Coordinate geometry patterns
        "parametric_curve": ["parametric", "lissajous", "trace"],
        "polar_rose": ["polar", "rose", "r = cos"],

        # More calculus patterns
        "related_rates": ["related rates", "changing", "ladder"],
        "mean_value_theorem": ["mean value", "mvt", "secant tangent"],
        "lhopitals_rule": ["lhopital", "indeterminate", "0/0"],
        "taylor_series": ["taylor", "series expansion", "polynomial"],
        "power_series": ["power series", "radius convergence"],

        # More geometry patterns
        "angle_bisector": ["bisector", "angle bisector"],
        "parallel_lines_transversal": ["parallel", "transversal", "corresponding"],
        "regular_polygon_construction": ["regular polygon", "inscribed"],
        "reflection_symmetry": ["reflection", "symmetry", "mirror"],

        # More linear algebra patterns
        "null_space": ["null space", "kernel", "nullity"],
        "gram_schmidt": ["gram schmidt", "orthogonalization"],
        "change_of_basis": ["change of basis", "coordinates"],

        # Differential equations patterns
        "slope_field": ["slope field", "direction field", "ODE"],
        "phase_plane": ["phase plane", "trajectory", "equilibrium"],

        # More probability patterns
        "central_limit_theorem": ["central limit", "clt", "normal"],
        "bayes_theorem": ["bayes", "conditional", "posterior"],
        "expected_value": ["expected value", "mean", "average"],

        # More number theory patterns
        "gcd_euclidean": ["gcd", "euclidean", "greatest common"],
        "modular_arithmetic": ["modular", "mod", "clock"],

        # Topology patterns
        "mobius_strip": ["mobius", "strip", "non-orientable"],

        # Numerical patterns
        "fixed_point_iteration": ["fixed point", "iteration", "cobweb"],

        # Effect patterns
        "particle_explosion": ["explosion", "particles", "scatter"],
        "spotlight_focus": ["spotlight", "focus", "dim"],
        "count_animation": ["count", "counter", "number"],
        "path_animation": ["path", "follow", "MoveAlongPath"],

        # Layout patterns
        "split_screen": ["split screen", "comparison", "side by side"],
        "zoom_pan_sequence": ["zoom pan", "camera sequence"],
        "progressive_reveal": ["reveal", "progressive", "mask"],
        "code_typing": ["code", "typing", "programming"],

        # Structure patterns
        "tree_diagram": ["tree diagram", "hierarchy"],
        "flowchart": ["flowchart", "decision", "process"],
        "timeline": ["timeline", "events", "chronological"],

        # More linear algebra patterns
        "matrix_multiplication": ["matrix multiplication", "row column"],

        # ML patterns
        "neural_network": ["neural network", "layers", "nodes"],
        "gradient_descent": ["gradient descent", "optimization"],
        "regression_line": ["regression", "line fit", "least squares"],

        # Algorithm patterns
        "binary_search": ["binary search", "divide conquer"],
        "sorting_visualization": ["sorting", "bubble sort", "comparison"],
        "recursion_tree": ["recursion", "call tree", "fibonacci tree"],

        # Data structure patterns
        "stack_visualization": ["stack", "push pop", "lifo"],
        "linked_list": ["linked list", "nodes pointers"],
        "hash_table": ["hash table", "hashing", "bucket"],

        # More vector patterns
        "cross_product": ["cross product", "perpendicular", "3d vectors"],

        # Trigonometry patterns
        "unit_circle_trig": ["unit circle", "trig", "sin cos circle"],
    }

    for pattern in ANIMATION_PATTERNS:
        triggers = pattern_triggers.get(pattern.name, pattern.keywords)
        if any(trigger in prompt_lower for trigger in triggers):
            matching.append(pattern)

    return matching



def get_all_patterns() -> list[AnimationPattern]:
    """Get all defined patterns."""
    return ANIMATION_PATTERNS
