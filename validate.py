#!/usr/bin/env python3
"""Compatibility wrapper for scripts/legacy/validate.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "scripts" / "legacy" / "validate.py"
    runpy.run_path(str(script_path), run_name="__main__")
