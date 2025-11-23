"""VLM Semantic Spatial Perception System"""

import os
from pathlib import Path

__version__ = "0.1.0"

# Load environment variables from .env file in project root
try:
    from dotenv import load_dotenv

    # Find project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    dotenv_path = project_root / ".env"

    if dotenv_path.exists():
        load_dotenv(dotenv_path)
except ImportError:
    # python-dotenv not installed, skip loading
    pass
