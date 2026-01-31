You are an expert ManimGL (3b1b/manim) developer. The following code has
validation errors. Fix the code to resolve them while preserving the intended
animation.

IMPORTANT: This is ManimGL, NOT Manim Community Edition. They have different APIs.

## Syntax Errors (CRITICAL - fix these first!)

**Unterminated string literal:**
- Check for MISMATCHED QUOTES: string started with " but ends with ' (or vice versa)
- Check for APOSTROPHES in English text: "Newton's Law" is fine, but 'Newton's Law' breaks!
  - ALWAYS use double quotes for text containing apostrophes: Text("Newton's First Law")
- Check for missing closing quote at end of string
- If error says "line X, column Y", look at EXACTLY that position

**Common quote mistakes to fix:**
- ❌ Text('Newton's Law')  →  ✅ Text("Newton's Law")
- ❌ TexText('Snell's Law')  →  ✅ TexText("Snell's Law")
- ❌ Text("Hello')  →  ✅ Text("Hello")
- ❌ Tex(r'\frac{1}{2}')  →  ✅ Tex(r"\frac{1}{2}") if content has apostrophes

## API Errors

- "unexpected keyword argument X" → REMOVE that parameter entirely, it doesn't exist in ManimGL
- "has no attribute 'set_X'" → Method doesn't exist. Use constructor params instead (e.g., Text("hi", font_size=24) not text.set_font_size(24))
- "has no attribute 'set_font_size'" → ManimGL Text doesn't have this. Use font_size= in constructor
- "has no attribute 'set_stroke_opacity'" → Use .set_stroke(opacity=X) instead
- "has no attribute 'set_fill_opacity'" → Use .set_fill(opacity=X) instead
- TypeError in Animation.__init__ → Remove invalid kwargs from animation calls
- TypeError in Mobject.__init__ → Remove invalid kwargs from mobject constructors
- ManimGL uses: run_time, rate_func, color, font_size (in constructor), stroke_width, fill_opacity
- ManimGL does NOT have: set_font_size(), set_stroke_opacity(), set_fill_opacity(), tips=, x_length=, y_length=, include_numbers=

Return ONLY the fixed Python code. No markdown fences. No explanations.
