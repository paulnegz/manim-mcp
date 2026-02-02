"""Layout Validator: 3b1b-style region management to avoid collisions.

When UR is used, next text goes to DL (opposite corner).
Also auto-positions shapes that would overlap at origin.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EDGE_BUFF = 0.3
CORNER_BUFF = 0.5

# Opposite corners for collision avoidance
ALTERNATIVES = {
    'UP': 'DOWN', 'DOWN': 'UP',
    'UL': 'DR', 'UR': 'DL', 'DL': 'UR', 'DR': 'UL',
}


@dataclass
class RegionTracker:
    """Tracks used screen regions."""
    used: set[str] = field(default_factory=set)

    def use(self, region: str) -> str:
        """Mark region as used. Returns alternative if already used."""
        if region not in self.used:
            self.used.add(region)
            return region

        # Already used - try alternative
        alt = ALTERNATIVES.get(region)
        if alt and alt not in self.used:
            self.used.add(alt)
            return alt

        # Both used - return original anyway
        self.used.add(region)
        return region


def _add_buff_to_edge(line: str) -> tuple[str, bool]:
    """Add buff to to_edge() if missing. Returns (new_line, changed)."""
    match = re.search(r'\.to_edge\s*\(\s*(UP|DOWN|LEFT|RIGHT)\s*\)', line)
    if not match:
        return line, False

    direction = match.group(1)
    new_line = re.sub(
        r'\.to_edge\s*\(\s*(UP|DOWN|LEFT|RIGHT)\s*\)',
        f'.to_edge({direction}, buff={EDGE_BUFF})',
        line
    )
    return new_line, new_line != line


def _add_buff_to_corner(line: str) -> tuple[str, bool]:
    """Add buff to to_corner() if missing. Returns (new_line, changed)."""
    match = re.search(r'\.to_corner\s*\(\s*(UL|UR|DL|DR)\s*\)', line)
    if not match:
        return line, False

    corner = match.group(1)
    new_line = re.sub(
        r'\.to_corner\s*\(\s*(UL|UR|DL|DR)\s*\)',
        f'.to_corner({corner}, buff={CORNER_BUFF})',
        line
    )
    return new_line, new_line != line


def _detect_region(line: str) -> str | None:
    """Detect which region this line targets."""
    patterns = [
        (r'to_edge\s*\(\s*(UP|DOWN|LEFT|RIGHT)', 1),
        (r'to_corner\s*\(\s*(UL|UR|DL|DR)', 1),
    ]
    for pattern, group in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(group)
    return None


def _relocate_region(line: str, old: str, new: str) -> str:
    """Replace region in line."""
    if old in ['UL', 'UR', 'DL', 'DR']:
        return line.replace(f'to_corner({old}', f'to_corner({new}')
    return line.replace(f'to_edge({old}', f'to_edge({new}')


def validate_and_fix_layout(code: str) -> str:
    """Fix layout issues: add buff, relocate collisions."""
    if not code:
        return code

    fixes = []
    tracker = RegionTracker()
    result_lines = []

    for i, line in enumerate(code.split('\n'), 1):
        # Step 1: Add buff if missing
        line, changed = _add_buff_to_edge(line)
        if changed:
            fixes.append(f'L{i}: added buff to to_edge')

        line, changed = _add_buff_to_corner(line)
        if changed:
            fixes.append(f'L{i}: added buff to to_corner')

        # Step 2: Handle region collision
        region = _detect_region(line)
        if region:
            new_region = tracker.use(region)
            if new_region != region:
                line = _relocate_region(line, region, new_region)
                fixes.append(f'L{i}: moved {region}→{new_region}')

        result_lines.append(line)

    code = '\n'.join(result_lines)

    # Step 3: Add safe_position helper if needed
    if 'always_redraw' in code and 'move_to' in code and 'def safe_position' not in code:
        match = re.search(r'(class \w+\([^)]+\):\s*\n)', code)
        if match:
            helper = '''
    @staticmethod
    def safe_position(pos):
        import numpy as np
        return np.array([np.clip(pos[0], -6.5, 6.5), np.clip(pos[1], -3.5, 3.5), 0])

'''
            code = code[:match.end()] + helper + code[match.end():]
            fixes.append('added safe_position helper')

    if fixes:
        logger.info("[LAYOUT] %d fixes: %s", len(fixes), '; '.join(fixes[:3]))

    return code


# Shape classes that need positioning when multiple exist
SHAPE_CLASSES = {
    "Rectangle", "Square", "Circle", "Ellipse", "Triangle", "Polygon",
    "Line", "Arrow", "DashedLine", "Arc", "Dot", "Annulus", "Sector",
    "RoundedRectangle", "RegularPolygon", "Star", "Cross",
}

# Position methods that indicate a shape has been positioned
POSITION_METHODS = {
    "to_edge", "to_corner", "move_to", "shift", "next_to", "align_to",
    "set_x", "set_y", "set_z", "center", "to_center",
}

# Default positions to assign to unpositioned shapes (spread across screen)
DEFAULT_POSITIONS = [
    '.to_corner(DL, buff=1)',
    '.to_corner(DR, buff=1)',
    '.to_corner(UL, buff=1)',
    '.to_corner(UR, buff=1)',
    '.to_edge(LEFT, buff=1)',
    '.to_edge(RIGHT, buff=1)',
    '.to_edge(DOWN, buff=1)',
    '.to_edge(UP, buff=1)',
]


def auto_position_shapes(code: str) -> str:
    """Auto-position shapes that would overlap at origin.

    Finds shape mobjects created without positioning methods and
    injects positioning to spread them across the screen.

    Args:
        code: Python code to fix

    Returns:
        Fixed code with positioning added
    """
    if not code:
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Track shape variables: name -> (line_num, class_name, has_position)
    shape_vars: dict[str, tuple[int, str, bool]] = {}
    lines = code.split('\n')

    class ShapeAnalyzer(ast.NodeVisitor):
        """Analyze shape assignments and position method calls."""

        def visit_Assign(self, node: ast.Assign) -> None:
            if isinstance(node.value, ast.Call):
                func = node.value.func
                class_name = None

                if isinstance(func, ast.Name) and func.id in SHAPE_CLASSES:
                    class_name = func.id
                elif isinstance(func, ast.Attribute) and func.attr in SHAPE_CLASSES:
                    class_name = func.attr

                if class_name:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            shape_vars[target.id] = (node.lineno, class_name, False)

            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr

                if method_name in POSITION_METHODS:
                    current = node.func.value
                    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
                        current = current.func.value

                    if isinstance(current, ast.Name) and current.id in shape_vars:
                        line_num, class_name, _ = shape_vars[current.id]
                        shape_vars[current.id] = (line_num, class_name, True)

            self.generic_visit(node)

    analyzer = ShapeAnalyzer()
    analyzer.visit(tree)

    # Find unpositioned shapes
    unpositioned = [(name, line_num, cls) for name, (line_num, cls, has_pos)
                    in shape_vars.items() if not has_pos]

    # Only fix if 2+ shapes are unpositioned
    if len(unpositioned) < 2:
        return code

    # Sort by line number to maintain order
    unpositioned.sort(key=lambda x: x[1])

    # Track which positions we've used
    position_idx = 0
    fixes = []

    for var_name, line_num, class_name in unpositioned:
        if position_idx >= len(DEFAULT_POSITIONS):
            position_idx = 0  # Wrap around if too many shapes

        position = DEFAULT_POSITIONS[position_idx]
        position_idx += 1

        # Find the line and add positioning after the assignment
        # Insert a new line after the shape creation
        line_idx = line_num - 1  # 0-indexed
        if line_idx < len(lines):
            # Get indentation from original line
            original_line = lines[line_idx]
            indent = len(original_line) - len(original_line.lstrip())
            indent_str = ' ' * indent

            # Insert positioning line after the shape creation
            position_line = f"{indent_str}{var_name}{position}"
            lines.insert(line_idx + 1, position_line)

            # Adjust subsequent line numbers since we inserted a line
            for i, (name, ln, cls) in enumerate(unpositioned):
                if ln > line_num:
                    unpositioned[i] = (name, ln + 1, cls)

            fixes.append(f"{var_name}→{position.split('(')[0][1:]}")

    if fixes:
        logger.info("[LAYOUT] Auto-positioned %d shapes: %s", len(fixes), '; '.join(fixes[:4]))

    return '\n'.join(lines)
