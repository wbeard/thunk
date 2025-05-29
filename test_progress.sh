#!/bin/bash
# Quick test script for structured logging

echo "🧪 Deep Research Agent Structured Logging Testing"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "src/thunk/__init__.py" ]; then
    echo "❌ Please run this from the project root directory"
    exit 1
fi

# Function to run test with error handling
run_test() {
    echo "Running: $1"
    echo "Press Ctrl+C to stop early, or wait for completion"
    echo "----------------------------------------------"
    if ! uv run "$@"; then
        echo "❌ Test failed or was interrupted"
    fi
    echo ""
}

case "${1:-help}" in
    "quick")
        echo "🚀 Quick Structured Logger Test"
        run_test test_ui_only.py
        ;;
    
    "sim")
        echo "🎭 Full Research Simulation (5 minutes)"
        run_test test_progress_simulation.py "Latest quantum computing breakthroughs" --duration 5.0
        ;;
    
    "sim-fast")
        echo "⚡ Fast Research Simulation (30 seconds)"
        run_test test_progress_simulation.py "AI quantum applications" --duration 0.5
        ;;
    
    "help"|*)
        echo "Available test commands:"
        echo ""
        echo "Logger Test (instant):"
        echo "  ./test_progress.sh quick      - Test structured logging interface"
        echo ""
        echo "Research Simulation Tests:"
        echo "  ./test_progress.sh sim        - Full 5-minute simulation"
        echo "  ./test_progress.sh sim-fast   - 30-second simulation"
        echo ""
        echo "Examples:"
        echo "  ./test_progress.sh quick      # Test the structured logging quickly"
        echo "  ./test_progress.sh sim-fast   # Quick end-to-end simulation"
        echo ""
        echo "💡 Pro tip: Use 'quick' for rapid logging format testing"
        ;;
esac