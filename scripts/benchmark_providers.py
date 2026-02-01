#!/usr/bin/env python3
"""
Benchmark DeepSeek vs Gemini for Manim code generation.

Compares providers across multiple metrics:
- Performance (generation time, retries)
- Code quality (syntax, Manim API, render success)
- RAG utilization
- Edge case handling

Usage:
    python scripts/benchmark_providers.py                    # Full benchmark
    python scripts/benchmark_providers.py --quick            # Quick test (3 prompts)
    python scripts/benchmark_providers.py --provider gemini  # Single provider test
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ANSI color codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"  # No Color

# Configuration
QUALITY = "low"

# Test prompts organized by difficulty and edge cases
TEST_PROMPTS = {
    "simple": [
        "Create a blue circle that moves from left to right",
        "Draw a red square and rotate it 90 degrees",
        "Show text 'Hello World' that fades in",
    ],
    "medium": [
        "Show the Pythagorean theorem with a right triangle and squares on each side",
        "Animate a sine wave being drawn, then transform it into a cosine wave",
        "Create a number line from -5 to 5 with a dot that moves along it",
    ],
    "complex": [
        "Demonstrate matrix multiplication with two 2x2 matrices showing step-by-step calculation",
        "Animate the derivative of x^2 showing the limit definition with shrinking delta",
        "Create a 3D rotating cube with labeled vertices A-H",
    ],
    "latex_heavy": [
        "Show the quadratic formula derivation: ax^2 + bx + c = 0 to x = (-b +/- sqrt(b^2-4ac)) / 2a",
        "Animate the chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x) with a concrete example",
    ],
    "timing_critical": [
        "Create 3 circles that appear sequentially, then all move to center simultaneously",
        "Animate a bouncing ball with realistic acceleration and deceleration",
    ],
    "long_animation": [
        "Explain the concept of integration as area under curve with multiple examples",
    ],
}

QUICK_PROMPTS = {
    "simple": ["Create a blue circle that moves from left to right"],
    "medium": ["Show the Pythagorean theorem with a right triangle and squares on each side"],
    "complex": ["Animate the derivative of x^2 showing the limit definition"],
}


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    provider: str
    category: str
    prompt: str
    success: bool
    generation_time_ms: float
    retry_count: int
    code_length: int = 0
    lines_of_code: int = 0
    ast_complexity: int = 0
    syntax_valid: bool = False
    manim_valid: bool = False
    render_success: bool = False
    rag_context_used: bool = False
    error: Optional[str] = None
    render_id: Optional[str] = None


@dataclass
class ProviderStats:
    """Aggregated statistics for a provider."""
    provider: str
    total_tests: int = 0
    successes: int = 0
    first_attempt_success: int = 0
    total_time_ms: float = 0
    total_retries: int = 0
    total_lines: int = 0
    total_ast_complexity: int = 0
    syntax_valid_count: int = 0
    manim_valid_count: int = 0
    render_success_count: int = 0
    rag_used_count: int = 0
    category_results: dict = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult):
        self.total_tests += 1
        if result.success:
            self.successes += 1
            if result.retry_count == 0:
                self.first_attempt_success += 1
        self.total_time_ms += result.generation_time_ms
        self.total_retries += result.retry_count
        self.total_lines += result.lines_of_code
        self.total_ast_complexity += result.ast_complexity
        if result.syntax_valid:
            self.syntax_valid_count += 1
        if result.manim_valid:
            self.manim_valid_count += 1
        if result.render_success:
            self.render_success_count += 1
        if result.rag_context_used:
            self.rag_used_count += 1

        # Track by category
        cat = result.category
        if cat not in self.category_results:
            self.category_results[cat] = {"total": 0, "success": 0}
        self.category_results[cat]["total"] += 1
        if result.success:
            self.category_results[cat]["success"] += 1


def get_project_dir() -> Path:
    """Get the project directory."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def find_manim_container() -> Optional[str]:
    """Find the running manim-mcp container."""
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    containers = result.stdout.strip().split("\n")
    # Look for container with manim-mcp in name (but not chromadb or minio)
    for container in containers:
        if "manim-mcp" in container and "chromadb" not in container and "minio" not in container:
            return container
    return None


def check_container_running(container: str) -> bool:
    """Check if the Docker container is running."""
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return container in result.stdout.split("\n")


def analyze_code_quality(code: str) -> tuple[int, int, bool]:
    """Analyze code for quality metrics.

    Returns: (lines_of_code, ast_complexity, syntax_valid)
    """
    lines = len([l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")])

    try:
        tree = ast.parse(code)
        # Count complexity: classes, functions, loops, conditionals
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += 2
            elif isinstance(node, (ast.For, ast.While, ast.If, ast.With)):
                complexity += 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
        return lines, complexity, True
    except SyntaxError:
        return lines, 0, False


def run_benchmark(
    provider: str,
    category: str,
    prompt: str,
    test_num: int,
    total_tests: int,
    container: str,
) -> BenchmarkResult:
    """Run a single benchmark test."""
    print(f"\n{CYAN}[{test_num}/{total_tests}]{NC} {BOLD}{provider.upper()}{NC} - {category}")
    print(f"  Prompt: {prompt[:60]}...")

    start_time = time.time()

    # Build the docker exec command
    cmd = [
        "docker", "exec",
        "-e", f"MANIM_MCP_LLM_PROVIDER={provider}",
        "-e", "MANIM_MCP_RAG_ENABLED=true",
        container,
        "manim-mcp", "gen", prompt,
        "--mode", "simple",
        "--quality", QUALITY,
    ]

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    exit_code = result.returncode

    generation_time_ms = (time.time() - start_time) * 1000

    # Parse output for metrics
    retry_count = len(re.findall(r"retry|attempt|retrying", output, re.I))
    rag_used = "RAG" in output or "context" in output.lower()

    # Extract render ID
    render_id = None
    render_match = re.search(r"Render ID[:\s]+(\S+)", output)
    if render_match:
        render_id = render_match.group(1)

    # Extract generated code if available
    code = ""
    code_match = re.search(r"```python\n(.*?)```", output, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    # Analyze code quality
    lines_of_code, ast_complexity, syntax_valid = analyze_code_quality(code) if code else (0, 0, False)

    # Check if it's a Manim-valid scene
    manim_valid = "class" in code and "Scene" in code if code else False

    # Check render success
    render_success = exit_code == 0 and ("mp4" in output or "video" in output.lower() or render_id is not None)

    success = exit_code == 0

    if success:
        print(f"  {GREEN}✓ SUCCESS{NC} - {generation_time_ms:.0f}ms, {retry_count} retries")
    else:
        error_match = re.search(r"(Error|Exception):\s*(.+?)(?:\n|$)", output)
        error = error_match.group(2)[:100] if error_match else "Unknown error"
        print(f"  {RED}✗ FAILED{NC} - {error[:50]}...")

    return BenchmarkResult(
        provider=provider,
        category=category,
        prompt=prompt,
        success=success,
        generation_time_ms=generation_time_ms,
        retry_count=retry_count,
        code_length=len(code),
        lines_of_code=lines_of_code,
        ast_complexity=ast_complexity,
        syntax_valid=syntax_valid,
        manim_valid=manim_valid,
        render_success=render_success,
        rag_context_used=rag_used,
        error=output if not success else None,
        render_id=render_id,
    )


def print_comparison_report(
    stats: dict[str, ProviderStats],
    results: list[BenchmarkResult],
):
    """Print a comprehensive comparison report."""
    providers = list(stats.keys())
    if len(providers) < 2:
        providers = providers + ["(N/A)"] * (2 - len(providers))

    p1, p2 = providers[0], providers[1] if len(providers) > 1 else "(N/A)"
    s1 = stats.get(p1, ProviderStats(provider=p1))
    s2 = stats.get(p2, ProviderStats(provider=p2)) if p2 != "(N/A)" else ProviderStats(provider=p2)

    def pct(count: int, total: int) -> str:
        return f"{100*count/total:.1f}%" if total > 0 else "N/A"

    def avg(total: float, count: int) -> str:
        return f"{total/count:.0f}" if count > 0 else "N/A"

    def winner(v1, v2, higher_better=True) -> str:
        if v1 == v2 or v2 == 0 or v1 == 0:
            return "Tie"
        if higher_better:
            return p1.title() if v1 > v2 else p2.title()
        return p1.title() if v1 < v2 else p2.title()

    print()
    print(f"{BOLD}{'═' * 80}{NC}")
    print(f"{BOLD}║{'DeepSeek vs Gemini Benchmark Report (RAG Enabled)':^78}║{NC}")
    print(f"{BOLD}{'═' * 80}{NC}")

    # Performance section
    print(f"\n{YELLOW}═══ PERFORMANCE ═══════════════════════════════════════════════════════════════{NC}")
    print(f"{'Metric':<25} {p1.title():<15} {p2.title():<15} Winner")
    print("-" * 70)

    avg_time_1 = s1.total_time_ms / s1.total_tests if s1.total_tests else 0
    avg_time_2 = s2.total_time_ms / s2.total_tests if s2.total_tests else 0
    print(f"{'Avg Generation Time':<25} {avg_time_1:,.0f} ms{'':<7} {avg_time_2:,.0f} ms{'':<7} {winner(avg_time_1, avg_time_2, False)}")

    first_success_1 = 100 * s1.first_attempt_success / s1.total_tests if s1.total_tests else 0
    first_success_2 = 100 * s2.first_attempt_success / s2.total_tests if s2.total_tests else 0
    print(f"{'First-Attempt Success':<25} {first_success_1:.1f}%{'':<10} {first_success_2:.1f}%{'':<10} {winner(first_success_1, first_success_2)}")

    avg_retries_1 = s1.total_retries / s1.total_tests if s1.total_tests else 0
    avg_retries_2 = s2.total_retries / s2.total_tests if s2.total_tests else 0
    print(f"{'Avg Retries Needed':<25} {avg_retries_1:.2f}{'':<12} {avg_retries_2:.2f}{'':<12} {winner(avg_retries_1, avg_retries_2, False)}")

    # Code quality section
    print(f"\n{YELLOW}═══ CODE QUALITY ══════════════════════════════════════════════════════════════{NC}")
    print(f"{'Metric':<25} {p1.title():<15} {p2.title():<15} Winner")
    print("-" * 70)

    print(f"{'Syntax Valid':<25} {pct(s1.syntax_valid_count, s1.total_tests):<15} {pct(s2.syntax_valid_count, s2.total_tests):<15} {winner(s1.syntax_valid_count, s2.syntax_valid_count)}")
    print(f"{'Manim API Valid':<25} {pct(s1.manim_valid_count, s1.total_tests):<15} {pct(s2.manim_valid_count, s2.total_tests):<15} {winner(s1.manim_valid_count, s2.manim_valid_count)}")
    print(f"{'Renders Successfully':<25} {pct(s1.render_success_count, s1.total_tests):<15} {pct(s2.render_success_count, s2.total_tests):<15} {winner(s1.render_success_count, s2.render_success_count)}")

    avg_lines_1 = s1.total_lines / s1.successes if s1.successes else 0
    avg_lines_2 = s2.total_lines / s2.successes if s2.successes else 0
    print(f"{'Avg Lines of Code':<25} {avg_lines_1:.0f}{'':<14} {avg_lines_2:.0f}{'':<14} {winner(avg_lines_1, avg_lines_2, False)} (concise)")

    avg_complexity_1 = s1.total_ast_complexity / s1.successes if s1.successes else 0
    avg_complexity_2 = s2.total_ast_complexity / s2.successes if s2.successes else 0
    print(f"{'AST Complexity (avg)':<25} {avg_complexity_1:.1f}{'':<14} {avg_complexity_2:.1f}{'':<14} {winner(avg_complexity_1, avg_complexity_2, False)} (simpler)")

    # RAG utilization section
    print(f"\n{YELLOW}═══ RAG UTILIZATION ═══════════════════════════════════════════════════════════{NC}")
    print(f"{'Metric':<25} {p1.title():<15} {p2.title():<15} Winner")
    print("-" * 70)

    print(f"{'RAG Context Used':<25} {pct(s1.rag_used_count, s1.total_tests):<15} {pct(s2.rag_used_count, s2.total_tests):<15} {winner(s1.rag_used_count, s2.rag_used_count)}")

    # Edge cases section
    print(f"\n{YELLOW}═══ EDGE CASES (Success Rate) ═════════════════════════════════════════════════{NC}")
    print(f"{'Category':<25} {p1.title():<15} {p2.title():<15} Winner")
    print("-" * 70)

    all_categories = set(s1.category_results.keys()) | set(s2.category_results.keys())
    for cat in sorted(all_categories):
        r1 = s1.category_results.get(cat, {"total": 0, "success": 0})
        r2 = s2.category_results.get(cat, {"total": 0, "success": 0})
        rate1 = 100 * r1["success"] / r1["total"] if r1["total"] else 0
        rate2 = 100 * r2["success"] / r2["total"] if r2["total"] else 0
        label = f"{cat} ({r1['total']} prompts)"
        print(f"{label:<25} {rate1:.0f}%{'':<14} {rate2:.0f}%{'':<14} {winner(rate1, rate2)}")

    # Recommendation
    print(f"\n{YELLOW}═══ RECOMMENDATION ════════════════════════════════════════════════════════════{NC}")

    # Calculate overall scores
    score1 = s1.successes + s1.first_attempt_success + s1.render_success_count - s1.total_retries
    score2 = s2.successes + s2.first_attempt_success + s2.render_success_count - s2.total_retries

    if score1 > score2:
        primary, fallback = p1.title(), p2.title()
    else:
        primary, fallback = p2.title(), p1.title()

    print(f"  Primary: {GREEN}{primary}{NC} (higher success rate, better reliability)")
    print(f"  Fallback: {BLUE}{fallback}{NC} (alternative option)")

    # Summary stats
    print(f"\n{BOLD}SUMMARY{NC}")
    print(f"  {p1.title()}: {s1.successes}/{s1.total_tests} success, avg {s1.total_time_ms/s1.total_tests if s1.total_tests else 0:,.0f}ms, {s1.total_retries} total retries")
    print(f"  {p2.title()}: {s2.successes}/{s2.total_tests} success, avg {s2.total_time_ms/s2.total_tests if s2.total_tests else 0:,.0f}ms, {s2.total_retries} total retries")


def save_results_json(results: list[BenchmarkResult], output_path: Path):
    """Save results to JSON file for further analysis."""
    data = []
    for r in results:
        data.append({
            "provider": r.provider,
            "category": r.category,
            "prompt": r.prompt,
            "success": r.success,
            "generation_time_ms": r.generation_time_ms,
            "retry_count": r.retry_count,
            "code_length": r.code_length,
            "lines_of_code": r.lines_of_code,
            "ast_complexity": r.ast_complexity,
            "syntax_valid": r.syntax_valid,
            "manim_valid": r.manim_valid,
            "render_success": r.render_success,
            "rag_context_used": r.rag_context_used,
            "render_id": r.render_id,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek vs Gemini for Manim code generation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with 3 prompts only",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "deepseek", "both"],
        default="both",
        help="Which provider(s) to test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    project_dir = get_project_dir()
    output_dir = project_dir / "assets" / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the manim container
    container = find_manim_container()
    if not container:
        print(f"{RED}Error: No manim-mcp container found running{NC}")
        print("Start it with: docker compose up -d")
        sys.exit(1)
    print(f"Using container: {container}")

    # Select prompts
    prompts = QUICK_PROMPTS if args.quick else TEST_PROMPTS

    # Select providers
    providers = []
    if args.provider in ["gemini", "both"]:
        providers.append("gemini")
    if args.provider in ["deepseek", "both"]:
        providers.append("deepseek")

    # Count total tests
    total_prompts = sum(len(p) for p in prompts.values())
    total_tests = total_prompts * len(providers)

    print()
    print(f"{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  DeepSeek vs Gemini Benchmark{NC}")
    print(f"{BOLD}{'=' * 60}{NC}")
    print(f"Providers: {', '.join(providers)}")
    print(f"Categories: {', '.join(prompts.keys())}")
    print(f"Total prompts: {total_prompts}")
    print(f"Total tests: {total_tests}")
    print(f"RAG: Enabled")
    print(f"Quality: {QUALITY}")

    results: list[BenchmarkResult] = []
    stats: dict[str, ProviderStats] = {}

    test_num = 0
    for provider in providers:
        stats[provider] = ProviderStats(provider=provider)

        for category, category_prompts in prompts.items():
            for prompt in category_prompts:
                test_num += 1
                result = run_benchmark(
                    provider=provider,
                    category=category,
                    prompt=prompt,
                    test_num=test_num,
                    total_tests=total_tests,
                    container=container,
                )
                results.append(result)
                stats[provider].add_result(result)

    # Print comparison report
    print_comparison_report(stats, results)

    # Save results
    output_path = args.output or (output_dir / f"benchmark_{int(time.time())}.json")
    save_results_json(results, output_path)

    # Return exit code based on success rate
    total_success = sum(s.successes for s in stats.values())
    total_tests = sum(s.total_tests for s in stats.values())
    success_rate = total_success / total_tests if total_tests else 0

    sys.exit(0 if success_rate >= 0.5 else 1)


if __name__ == "__main__":
    main()
