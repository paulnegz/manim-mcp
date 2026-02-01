You are filling a code section in a Manim animation.

CONTEXT (code before this section):
```python
{prefix}
```

SECTION HEADER:
{header_comment}

YOUR TASK:
Generate ONLY the color definitions that will be used in this animation.
Define colors as variables like: PRIMARY_COLOR = BLUE

RULES:
- Output ONLY Python code, no explanations
- Define 2-4 color variables using Manim color constants (BLUE, RED, GREEN, YELLOW, etc.)
- Use descriptive names like PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR
- Do NOT include any self.play(), self.wait(), or animation code
- Do NOT include comments (the section header is already present)

CONTEXT (code after this section):
```python
{suffix}
```

Generate the color definitions:
