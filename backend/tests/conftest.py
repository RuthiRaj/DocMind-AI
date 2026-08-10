"""
Shared pytest configuration for DocMind AI backend tests.
"""

import os
import sys
from pathlib import Path

# Ensure backend package imports resolve and settings validation succeeds in CI/local test runs.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("GROQ_API_KEY", "test-groq-key-for-unit-tests")
