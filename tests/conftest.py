"""
Shared test configuration and fixtures.
"""

import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_api: requires live LLM API access")


@pytest.fixture
def genai_api_key():
    """
    Fetch the GenAI API key from environment. Skip tests needing API if absent.
    Supports GENAI_API_KEY, GOOGLE_API_KEY, and GEMINI_API_KEY for compatibility.
    """
    key = (
        os.environ.get("GENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not key:
        pytest.skip("GENAI_API_KEY/GOOGLE_API_KEY/GEMINI_API_KEY not set; skipping API-dependent test.")
    return key
