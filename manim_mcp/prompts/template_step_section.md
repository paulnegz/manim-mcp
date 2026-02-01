You are filling a code section in a Manim animation using Fill-in-the-Middle generation.

CONTEXT (code before this section):
```python
{prefix}
```

SECTION HEADER:
{header_comment}

NARRATION TO VISUALIZE:
"{narration}"

YOUR TASK:
Generate ONLY the Manim code to visualize this narration step.
The code will be inserted between the section header and the self.wait(2) call.
{rag_context}
RULES:
- Output ONLY Python code, no explanations or markdown
- Use self.play() for animations
- Create or modify Mobjects as needed for this step
- You can reference colors like PRIMARY_COLOR, SECONDARY_COLOR defined earlier
- Do NOT add self.wait() - it's already in the template
- Do NOT include the section header comment - it's already present
- Keep the code focused on this single narration step
- Use variables from previous steps if they're in the prefix context

CONTEXT (code after this section):
```python
{suffix}
```

Generate the code for this step:
