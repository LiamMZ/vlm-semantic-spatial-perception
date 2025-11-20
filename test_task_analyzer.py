#!/usr/bin/env python3
"""Test script to verify LLMTaskAnalyzer uses new GenAI interface."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.planning.llm_task_analyzer import LLMTaskAnalyzer

load_dotenv()

def test_task_analyzer():
    print("\n" + "="*70)
    print("TESTING LLMTASKANALYZER WITH NEW GENAI SDK")
    print("="*70)
    print()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ WARNING: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        print("Testing initialization without API key...")
        api_key = None

    try:
        analyzer = LLMTaskAnalyzer(api_key=api_key)
        print("✓ LLMTaskAnalyzer initialized successfully!")
        print(f"  • Using new SDK: {analyzer.use_new_sdk}")
        print(f"  • Model: {analyzer.model_name}")
        print()
        print("="*70)
        print("TEST PASSED")
        print("="*70)
        return True

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("="*70)
        print("TEST FAILED")
        print("="*70)
        return False

if __name__ == "__main__":
    success = test_task_analyzer()
    sys.exit(0 if success else 1)
