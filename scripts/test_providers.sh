#!/bin/bash
# Test script for manim-mcp provider combinations
# Tests all combinations of: LLM (Gemini/Claude) x Mode (Simple/Advanced) x RAG (On/Off)
#
# Usage:
#   ./scripts/test_providers.sh              # Run all 8 tests
#   ./scripts/test_providers.sh --no-audio   # Run without audio (faster, no TTS quota)
#   ./scripts/test_providers.sh --quick      # Run only simple mode tests (4 tests)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/assets/test_videos"
PROMPT="Explain calculus from first principle"
QUALITY="low"
CONTAINER="manim-mcp-manim-mcp-1"

# Parse arguments
AUDIO_FLAG="--audio"
RUN_ADVANCED=true
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-audio)
            AUDIO_FLAG=""
            shift
            ;;
        --quick)
            RUN_ADVANCED=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-audio] [--quick]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Results tracking
declare -A RESULTS
PASS_COUNT=0
FAIL_COUNT=0

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "$CONTAINER"; then
    echo -e "${RED}Error: Container $CONTAINER is not running${NC}"
    echo "Start it with: docker compose up -d"
    exit 1
fi

run_test() {
    local test_num=$1
    local test_name=$2
    local provider=$3
    local mode=$4
    local rag_enabled=$5
    local output_file=$6

    echo ""
    echo "========================================"
    echo -e "${YELLOW}Test $test_num: $test_name${NC}"
    echo "  Provider: $provider | Mode: $mode | RAG: $rag_enabled | Audio: ${AUDIO_FLAG:-none}"
    echo "========================================"

    local start_time=$(date +%s)

    # Run the test
    local cmd="docker exec"
    cmd+=" -e MANIM_MCP_LLM_PROVIDER=$provider"
    cmd+=" -e MANIM_MCP_RAG_ENABLED=$rag_enabled"
    cmd+=" $CONTAINER"
    cmd+=" manim-mcp gen \"$PROMPT\" --mode $mode --quality $QUALITY $AUDIO_FLAG"

    echo "Running: $cmd"

    # Capture output and exit code
    local output
    local exit_code
    output=$(eval "$cmd" 2>&1) || exit_code=$?
    exit_code=${exit_code:-0}

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC} (${duration}s)"
        RESULTS["$test_name"]="PASS"
        ((PASS_COUNT++))

        # Extract render ID and copy video
        local render_id=$(echo "$output" | grep "Render ID" | awk '{print $NF}')
        if [ -n "$render_id" ]; then
            echo "  Render ID: $render_id"

            # Find and copy the video file
            local video_path=$(docker exec "$CONTAINER" find /tmp -name "*_with_audio.mp4" -o -name "*.mp4" 2>/dev/null | grep "$render_id" | head -1 || true)
            if [ -z "$video_path" ]; then
                video_path=$(docker exec "$CONTAINER" find /tmp -path "*$render_id*" -name "*.mp4" 2>/dev/null | head -1 || true)
            fi

            if [ -n "$video_path" ]; then
                docker cp "$CONTAINER:$video_path" "$OUTPUT_DIR/$output_file" 2>/dev/null || true
                if [ -f "$OUTPUT_DIR/$output_file" ]; then
                    local size=$(ls -lh "$OUTPUT_DIR/$output_file" | awk '{print $5}')
                    echo "  Saved: $OUTPUT_DIR/$output_file ($size)"
                fi
            fi
        fi
    else
        echo -e "${RED}✗ FAIL${NC} (${duration}s)"
        RESULTS["$test_name"]="FAIL"
        ((FAIL_COUNT++))

        # Show error summary
        local error_line=$(echo "$output" | grep -E "(Error|Exception|✗)" | head -3)
        if [ -n "$error_line" ]; then
            echo "  Error: $error_line"
        fi
    fi
}

echo ""
echo "============================================"
echo "  manim-mcp Provider Test Suite"
echo "============================================"
echo "Prompt: $PROMPT"
echo "Quality: $QUALITY"
echo "Audio: ${AUDIO_FLAG:-disabled}"
echo "Output: $OUTPUT_DIR"
echo ""

# Test 1: Simple Gemini + RAG
run_test 1 "Simple_Gemini_RAG" "gemini" "simple" "true" "test1_simple_gemini_rag.mp4"

# Test 2: Advanced Gemini + RAG
if [ "$RUN_ADVANCED" = true ]; then
    run_test 2 "Advanced_Gemini_RAG" "gemini" "advanced" "true" "test2_advanced_gemini_rag.mp4"
fi

# Test 3: Simple Claude + RAG
run_test 3 "Simple_Claude_RAG" "claude" "simple" "true" "test3_simple_claude_rag.mp4"

# Test 4: Advanced Claude + RAG
if [ "$RUN_ADVANCED" = true ]; then
    run_test 4 "Advanced_Claude_RAG" "claude" "advanced" "true" "test4_advanced_claude_rag.mp4"
fi

# Test 5: Simple Gemini + No RAG
run_test 5 "Simple_Gemini_NoRAG" "gemini" "simple" "false" "test5_simple_gemini_norag.mp4"

# Test 6: Advanced Gemini + No RAG
if [ "$RUN_ADVANCED" = true ]; then
    run_test 6 "Advanced_Gemini_NoRAG" "gemini" "advanced" "false" "test6_advanced_gemini_norag.mp4"
fi

# Test 7: Simple Claude + No RAG
run_test 7 "Simple_Claude_NoRAG" "claude" "simple" "false" "test7_simple_claude_norag.mp4"

# Test 8: Advanced Claude + No RAG
if [ "$RUN_ADVANCED" = true ]; then
    run_test 8 "Advanced_Claude_NoRAG" "claude" "advanced" "false" "test8_advanced_claude_norag.mp4"
fi

# Summary
echo ""
echo "============================================"
echo "  Test Summary"
echo "============================================"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"
echo ""
echo "Results:"
for test_name in "${!RESULTS[@]}"; do
    result="${RESULTS[$test_name]}"
    if [ "$result" = "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} $test_name"
    else
        echo -e "  ${RED}✗${NC} $test_name"
    fi
done | sort

echo ""
echo "Videos saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "  (no videos)"

# Exit with failure if any tests failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
