"""Shared pytest fixtures."""

import sys
from pathlib import Path

# Make `import core`, `import agents`, etc. work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
