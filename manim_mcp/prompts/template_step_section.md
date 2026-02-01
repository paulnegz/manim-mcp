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

RULES:
1. Create objects needed for this step
2. Use self.play() for animations
3. Use self.wait() for pauses (typically 0.5-2 seconds)
4. Reference colors defined in the # --- COLORS --- section
5. Position objects explicitly (.move_to(), .next_to(), .shift())
6. Add comments describing what's happening
7. Keep code clean and readable
8. Match the narration content exactly

AVAILABLE ANIMATIONS:
- Create, Write, FadeIn, FadeOut
- Transform, ReplacementTransform
- GrowArrow, ShowCreation
- Indicate, Flash, Circumscribe
- MoveToTarget, ApplyMethod

DO NOT:
- Define new colors (use existing ones)
- Create the class definition
- Include imports
- Add markdown fences

Generate the code for this step:
