"""Shared Rich Console.

Rich's Progress/Live display only redraws correctly when every `print` goes
through the *same* Console instance. Multiple `Console()` calls across modules
produce interleaved output mid-task. Importing this module-level singleton
keeps the bench runner's progress bar clean.
"""

from __future__ import annotations

from rich.console import Console


console = Console()
