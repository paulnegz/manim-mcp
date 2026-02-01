You are filling a code section in a Manim animation.

CONTEXT (code before this section):
```python
{prefix}
```

SECTION HEADER:
{header_comment}

YOUR TASK:
Generate ONLY the color definitions that will be used in this animation.

RULES:
1. Use descriptive variable names (e.g., primary_color, accent_color, background_highlight)
2. Assign Manim color constants (BLUE, RED, GREEN, YELLOW, WHITE, etc.)
3. Can use color variants (BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E)
4. DO NOT shadow Manim built-ins (don't write BLUE = BLUE)
5. Use ALL_CAPS for color variable names
6. One color definition per line
7. Only define colors that will actually be used

EXAMPLE OUTPUT:
PRIMARY_COLOR = BLUE_D
ACCENT_COLOR = YELLOW
HIGHLIGHT_COLOR = GREEN_B

Generate the color definitions:
