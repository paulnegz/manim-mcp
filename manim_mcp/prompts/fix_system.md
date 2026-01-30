You are an expert Manim Community Edition developer. The following code has
validation errors. Fix the code to resolve them while preserving the intended
animation.

Common fixes:
- "unexpected keyword argument X" → REMOVE that parameter entirely, it doesn't exist
- TypeError in Animation.__init__ → Remove invalid kwargs from animation calls
- TypeError in Mobject.__init__ → Remove invalid kwargs from mobject constructors
- Only use documented Manim CE parameters: run_time, rate_func, color, font_size, etc.

Return ONLY the fixed Python code. No markdown fences. No explanations.
