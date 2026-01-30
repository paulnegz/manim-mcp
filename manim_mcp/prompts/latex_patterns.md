EQUATION PATTERNS (3b1b style with manimgl):
- Reveal equations: Write(Tex(r"E = mc^2"))
- Equation morphing: TransformMatchingTex(old_eq, new_eq) - morphs matching parts
- Step-by-step derivation: Transform(eq1, eq2) to show "this becomes that"
- Highlight terms: eq.set_color_by_tex("x", YELLOW)
- Emphasis: Indicate(equation, color=YELLOW), FlashAround(result)
- Surround: SurroundingRectangle(key_term, color=GOLD)
- Text labels: TexText("label").next_to(eq, DOWN)

CRITICAL: Use Tex() for math, NOT MathTex() (MathTex doesn't exist in manimgl!)
