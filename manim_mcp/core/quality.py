"""Animation quality scoring based on 3Blue1Brown patterns.

This module scores generated Manim code against quality criteria
derived from analyzing Grant Sanderson's actual animation code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class QualityScore:
    """Quality assessment of generated Manim code."""

    # Animation patterns (0-10 each)
    pacing_score: int = 0           # Varied wait() calls
    transform_score: int = 0        # Uses Transform/FadeTransform
    reveal_score: int = 0           # Uses LaggedStart patterns
    color_score: int = 0            # Semantic color usage
    structure_score: int = 0        # Has proper scene arc

    # Detected patterns
    patterns_found: list[str] = field(default_factory=list)
    issues_found: list[str] = field(default_factory=list)

    # Metrics
    estimated_duration: float = 0.0
    animation_count: int = 0
    wait_calls: list[float] = field(default_factory=list)

    @property
    def total_score(self) -> int:
        """Total quality score (0-50)."""
        return (
            self.pacing_score +
            self.transform_score +
            self.reveal_score +
            self.color_score +
            self.structure_score
        )

    @property
    def grade(self) -> str:
        """Letter grade based on total score."""
        score = self.total_score
        if score >= 45:
            return "A"
        elif score >= 40:
            return "B"
        elif score >= 30:
            return "C"
        elif score >= 20:
            return "D"
        return "F"

    @property
    def is_3b1b_quality(self) -> bool:
        """Whether this meets 3Blue1Brown quality standards."""
        return self.total_score >= 35


class AnimationQualityScorer:
    """Scores generated Manim code for 3Blue1Brown-style quality.

    Based on patterns extracted from 3b1b/videos repository.
    """

    # 3b1b animation patterns to look for
    LAGGED_PATTERNS = [
        r"LaggedStart",
        r"LaggedStartMap",
        r"lag_ratio\s*=",
    ]

    TRANSFORM_PATTERNS = [
        r"Transform\s*\(",
        r"FadeTransform\s*\(",
        r"TransformFromCopy\s*\(",
        r"TransformMatchingTex\s*\(",
        r"ReplacementTransform\s*\(",
        r"MoveToTarget\s*\(",
    ]

    EMPHASIS_PATTERNS = [
        r"Indicate\s*\(",
        r"FlashAround\s*\(",
        r"Circumscribe\s*\(",
        r"SurroundingRectangle\s*\(",
    ]

    SEMANTIC_COLORS = {
        "BLUE": "input/primary",
        "TEAL": "supporting",
        "GREEN": "transformation",
        "YELLOW": "result/highlight",
        "GOLD": "result/highlight",
        "RED": "constraint/warning",
        "GREY": "scaffolding",
    }

    def score(self, code: str) -> QualityScore:
        """Score the generated code against 3b1b quality criteria.

        Args:
            code: Generated Manim Python code

        Returns:
            QualityScore with detailed breakdown
        """
        result = QualityScore()

        # Score pacing (varied wait calls)
        result.pacing_score, result.wait_calls = self._score_pacing(code)

        # Score transform usage
        result.transform_score = self._score_transforms(code, result)

        # Score reveal patterns
        result.reveal_score = self._score_reveals(code, result)

        # Score color semantics
        result.color_score = self._score_colors(code, result)

        # Score scene structure
        result.structure_score = self._score_structure(code, result)

        # Estimate duration
        result.estimated_duration = self._estimate_duration(code, result.wait_calls)

        # Count animations
        result.animation_count = len(re.findall(r"self\.play\s*\(", code))

        return result

    def _score_pacing(self, code: str) -> tuple[int, list[float]]:
        """Score pacing based on wait() call variety."""
        # Extract all wait() calls with their durations
        wait_pattern = r"self\.wait\s*\(\s*(\d*\.?\d*)\s*\)"
        matches = re.findall(wait_pattern, code)

        # Also count bare self.wait() calls (default duration)
        bare_waits = len(re.findall(r"self\.wait\s*\(\s*\)", code))

        durations = []
        for match in matches:
            if match:
                try:
                    durations.append(float(match))
                except ValueError:
                    durations.append(1.0)  # Default

        # Add default durations for bare waits
        durations.extend([1.0] * bare_waits)

        if not durations:
            return 2, []  # No waits = poor pacing

        # Check for variety
        unique_durations = set(durations)

        if len(unique_durations) >= 3:
            # Excellent variety: 0.5, 1.0, 2.0+ patterns
            score = 10
        elif len(unique_durations) == 2:
            score = 7
        elif len(durations) >= 3:
            # Multiple waits but same duration
            score = 5
        else:
            score = 3

        # Bonus for ending with longer wait (let it breathe)
        if durations and durations[-1] >= 1.5:
            score = min(10, score + 1)

        return score, durations

    def _score_transforms(self, code: str, result: QualityScore) -> int:
        """Score use of transform animations."""
        score = 0
        found = []

        for pattern in self.TRANSFORM_PATTERNS:
            if re.search(pattern, code):
                score += 2
                found.append(pattern.replace(r"\s*\(", "()"))

        # Cap at 10
        score = min(10, score)

        if found:
            result.patterns_found.append(f"Transforms: {', '.join(found)}")

        # Check for anti-pattern: FadeOut then FadeIn (should be Transform)
        if re.search(r"FadeOut.*\n.*FadeIn", code):
            result.issues_found.append("Consider using Transform instead of FadeOut→FadeIn")
            score = max(0, score - 2)

        return score

    def _score_reveals(self, code: str, result: QualityScore) -> int:
        """Score use of staggered reveal patterns."""
        score = 0

        for pattern in self.LAGGED_PATTERNS:
            if re.search(pattern, code):
                score += 3
                result.patterns_found.append("Uses LaggedStart patterns")
                break

        # Check for emphasis patterns
        for pattern in self.EMPHASIS_PATTERNS:
            if re.search(pattern, code):
                score += 2
                result.patterns_found.append("Uses emphasis animations")
                break

        # Check for run_time variations
        run_times = re.findall(r"run_time\s*=\s*(\d*\.?\d+)", code)
        if run_times:
            try:
                rt_values = [float(rt) for rt in run_times]
                if any(rt >= 1.5 for rt in rt_values):
                    score += 2
                    result.patterns_found.append("Uses longer run_times for emphasis")
            except ValueError:
                pass

        return min(10, score)

    def _score_colors(self, code: str, result: QualityScore) -> int:
        """Score semantic color usage."""
        colors_used = []
        for color in self.SEMANTIC_COLORS:
            if color in code:
                colors_used.append(color)

        if not colors_used:
            return 3  # Default/no colors

        # Check for semantic usage patterns
        score = min(10, len(colors_used) * 2)

        # Bonus for result highlighting pattern
        if "YELLOW" in code or "GOLD" in code:
            if re.search(r"(Indicate|FlashAround|set_color.*YELLOW)", code):
                score = min(10, score + 2)
                result.patterns_found.append("Highlights results with YELLOW/GOLD")

        return score

    def _score_structure(self, code: str, result: QualityScore) -> int:
        """Score scene structure (establish→build→insight→resolve)."""
        score = 0

        # Check for title/setup at start
        lines = code.split('\n')
        construct_started = False
        has_early_title = False

        for i, line in enumerate(lines[:30]):  # Check first 30 lines
            if "def construct" in line:
                construct_started = True
            if construct_started:
                if re.search(r"(title|Title|header|Header)", line):
                    has_early_title = True
                    break
                if re.search(r'Text\s*\([^)]*font_size\s*=\s*3[6-9]|4[0-8]', line):
                    has_early_title = True  # Large text = likely title
                    break

        if has_early_title:
            score += 3
            result.patterns_found.append("Has title/setup phase")

        # Check for cleanup between sections
        if re.search(r"FadeOut\s*\(\s*\*\s*self\.mobjects", code):
            score += 2
            result.patterns_found.append("Clears screen between sections")

        # Check for final breathing room
        wait_pattern = r"self\.wait\s*\(\s*(\d*\.?\d*)\s*\)"
        all_waits = list(re.finditer(wait_pattern, code))
        if all_waits:
            last_wait = all_waits[-1]
            try:
                duration = float(last_wait.group(1)) if last_wait.group(1) else 1.0
                if duration >= 1.5:
                    score += 2
                    result.patterns_found.append("Ends with breathing room")
            except ValueError:
                pass

        # Check for phase comments or structure
        phase_indicators = [
            r"#\s*(Phase|Step|Section|Part)\s*\d",
            r"#\s*(Setup|Build|Reveal|Conclusion|Resolve)",
            r"#\s*(Establish|Introduction|Main|Insight)",
        ]
        for pattern in phase_indicators:
            if re.search(pattern, code, re.IGNORECASE):
                score += 1
                result.patterns_found.append("Has structured phase comments")
                break

        return min(10, score)

    def _estimate_duration(self, code: str, wait_calls: list[float]) -> float:
        """Estimate animation duration in seconds."""
        # Sum of explicit waits
        wait_total = sum(wait_calls) if wait_calls else 0.0

        # Estimate animation time from play() calls
        play_count = len(re.findall(r"self\.play\s*\(", code))

        # Extract run_time values
        run_times = re.findall(r"run_time\s*=\s*(\d*\.?\d+)", code)
        explicit_runtime = sum(float(rt) for rt in run_times if rt)

        # Estimate: each play() takes ~1s default, minus explicit run_times
        implicit_plays = max(0, play_count - len(run_times))
        implicit_runtime = implicit_plays * 1.0

        return wait_total + explicit_runtime + implicit_runtime


def score_animation_code(code: str) -> QualityScore:
    """Convenience function to score animation code.

    Args:
        code: Generated Manim Python code

    Returns:
        QualityScore with detailed breakdown
    """
    scorer = AnimationQualityScorer()
    return scorer.score(code)
