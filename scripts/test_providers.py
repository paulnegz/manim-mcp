#!/usr/bin/env python3
"""Test script for manim-mcp provider combinations.

Tests all combinations of: LLM (Gemini/Claude) x Mode (Simple/Advanced) x RAG (On/Off)

Usage:
    python scripts/test_providers.py              # Run all 8 tests
    python scripts/test_providers.py --no-audio   # Run without audio (faster, no TTS quota)
    python scripts/test_providers.py --quick      # Run only simple mode tests (4 tests)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ANSI color codes
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    NC = "\033[0m"  # No Color


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    render_id: str | None = None
    error: str | None = None


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run a command and return exit code and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
        return result.returncode, result.stdout + result.stderr
    except Exception as e:
        return 1, str(e)


def check_container_running(container: str) -> bool:
    """Check if Docker container is running."""
    code, output = run_command(["docker", "ps", "--format", "{{.Names}}"])
    return container in output


def run_test(
    test_num: int,
    test_name: str,
    provider: str,
    mode: str,
    rag_enabled: bool,
    output_file: str,
    container: str,
    prompt: str,
    quality: str,
    audio_flag: str,
    output_dir: Path,
) -> TestResult:
    """Run a single test and return the result."""
    print()
    print("=" * 40)
    print(f"{Colors.YELLOW}Test {test_num}: {test_name}{Colors.NC}")
    print(f"  Provider: {provider} | Mode: {mode} | RAG: {rag_enabled} | Audio: {audio_flag or 'none'}")
    print("=" * 40)

    start_time = time.time()

    # Build docker exec command
    cmd = [
        "docker", "exec",
        "-e", f"MANIM_MCP_LLM_PROVIDER={provider}",
        "-e", f"MANIM_MCP_RAG_ENABLED={'true' if rag_enabled else 'false'}",
        container,
        "manim-mcp", "gen", prompt,
        "--mode", mode,
        "--quality", quality,
    ]
    if audio_flag:
        cmd.append(audio_flag)

    print(f"Running: {' '.join(cmd)}")

    exit_code, output = run_command(cmd)
    duration = time.time() - start_time

    if exit_code == 0:
        print(f"{Colors.GREEN}✓ PASS{Colors.NC} ({duration:.1f}s)")

        # Extract render ID
        render_id = None
        for line in output.split("\n"):
            if "Render ID" in line:
                render_id = line.split()[-1]
                break

        if render_id:
            print(f"  Render ID: {render_id}")

            # Find and copy video file
            find_cmd = [
                "docker", "exec", container,
                "find", "/tmp", "-path", f"*{render_id}*", "-name", "*.mp4"
            ]
            _, find_output = run_command(find_cmd)
            video_paths = [p for p in find_output.strip().split("\n") if p]

            if video_paths:
                video_path = video_paths[0]
                local_path = output_dir / output_file
                copy_cmd = ["docker", "cp", f"{container}:{video_path}", str(local_path)]
                run_command(copy_cmd)

                if local_path.exists():
                    size = local_path.stat().st_size
                    size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024 * 1024 else f"{size / 1024:.1f}KB"
                    print(f"  Saved: {local_path} ({size_str})")

        return TestResult(name=test_name, passed=True, duration=duration, render_id=render_id)
    else:
        print(f"{Colors.RED}✗ FAIL{Colors.NC} ({duration:.1f}s)")

        # Extract error message
        error_lines = []
        for line in output.split("\n"):
            if any(x in line for x in ["Error", "Exception", "✗"]):
                error_lines.append(line)
                if len(error_lines) >= 3:
                    break

        error = "\n".join(error_lines) if error_lines else None
        if error:
            print(f"  Error: {error}")

        return TestResult(name=test_name, passed=False, duration=duration, error=error)


def main():
    parser = argparse.ArgumentParser(description="Test manim-mcp provider combinations")
    parser.add_argument("--no-audio", action="store_true", help="Run without audio (faster)")
    parser.add_argument("--quick", action="store_true", help="Run only simple mode tests (4 tests)")
    args = parser.parse_args()

    # Configuration
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / "assets" / "test_videos"
    prompt = "Explain calculus from first principle"
    quality = "low"
    container = "manim-mcp-manim-mcp-1"
    audio_flag = "" if args.no_audio else "--audio"
    run_advanced = not args.quick

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check container is running
    if not check_container_running(container):
        print(f"{Colors.RED}Error: Container {container} is not running{Colors.NC}")
        print("Start it with: docker compose up -d")
        sys.exit(1)

    print()
    print("=" * 44)
    print("  manim-mcp Provider Test Suite")
    print("=" * 44)
    print(f"Prompt: {prompt}")
    print(f"Quality: {quality}")
    print(f"Audio: {audio_flag or 'disabled'}")
    print(f"Output: {output_dir}")
    print()

    results: list[TestResult] = []

    # Define test configurations
    tests = [
        (1, "Simple_Gemini_RAG", "gemini", "simple", True, "test1_simple_gemini_rag.mp4"),
        (2, "Advanced_Gemini_RAG", "gemini", "advanced", True, "test2_advanced_gemini_rag.mp4"),
        (3, "Simple_Claude_RAG", "claude", "simple", True, "test3_simple_claude_rag.mp4"),
        (4, "Advanced_Claude_RAG", "claude", "advanced", True, "test4_advanced_claude_rag.mp4"),
        (5, "Simple_Gemini_NoRAG", "gemini", "simple", False, "test5_simple_gemini_norag.mp4"),
        (6, "Advanced_Gemini_NoRAG", "gemini", "advanced", False, "test6_advanced_gemini_norag.mp4"),
        (7, "Simple_Claude_NoRAG", "claude", "simple", False, "test7_simple_claude_norag.mp4"),
        (8, "Advanced_Claude_NoRAG", "claude", "advanced", False, "test8_advanced_claude_norag.mp4"),
    ]

    for test_num, test_name, provider, mode, rag_enabled, output_file in tests:
        # Skip advanced tests if --quick
        if mode == "advanced" and not run_advanced:
            continue

        result = run_test(
            test_num=test_num,
            test_name=test_name,
            provider=provider,
            mode=mode,
            rag_enabled=rag_enabled,
            output_file=output_file,
            container=container,
            prompt=prompt,
            quality=quality,
            audio_flag=audio_flag,
            output_dir=output_dir,
        )
        results.append(result)

    # Summary
    pass_count = sum(1 for r in results if r.passed)
    fail_count = sum(1 for r in results if not r.passed)

    print()
    print("=" * 44)
    print("  Test Summary")
    print("=" * 44)
    print(f"Passed: {Colors.GREEN}{pass_count}{Colors.NC}")
    print(f"Failed: {Colors.RED}{fail_count}{Colors.NC}")
    print()
    print("Results:")

    for result in sorted(results, key=lambda r: r.name):
        if result.passed:
            print(f"  {Colors.GREEN}✓{Colors.NC} {result.name}")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} {result.name}")

    print()
    print(f"Videos saved to: {output_dir}")

    # List videos
    videos = list(output_dir.glob("*.mp4"))
    if videos:
        for video in sorted(videos):
            size = video.stat().st_size
            size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024 * 1024 else f"{size / 1024:.1f}KB"
            print(f"  {video.name} ({size_str})")
    else:
        print("  (no videos)")

    # Exit with failure if any tests failed
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
