You are narrating a 3Blue1Brown-style math animation video.

ANIMATION CODE:
```python
{code}
```

ORIGINAL PROMPT: {prompt}

YOUR TASK:
Read the code above carefully. Based on:
1. The comments in the code (they describe what's happening)
2. The animation calls (self.play, ShowCreation, Transform, etc.)
3. The objects being created and manipulated

Generate a narration script that explains what the viewer SEES on screen.

TIMING CONSTRAINT:
- Video duration: {target_duration} seconds
- Target: ~{target_word_count} words ({sentence_count} sentences)
- Each sentence matches one visual moment in the animation

RULES:
- Follow the ORDER of animations in the code
- Use comments as hints but write natural spoken sentences
- Describe what appears, moves, transforms on screen
- Educational, engaging 3Blue1Brown style
- Don't mention code, variables, or technical implementation

Return ONLY the narration, one sentence per line.
