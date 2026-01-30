"""Layout Validator: 3b1b-style region management to avoid collisions.

When UR is used, next text goes to DL (opposite corner).
"""

from __future__ import annotations

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
