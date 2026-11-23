#!/usr/bin/env python3
"""
MOSAIC-v2 Root Launcher Forwarder
Allows running 'python scripts/launcher.py' directly from repository root.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
backend_launcher = ROOT / "backend" / "scripts" / "launcher.py"

if __name__ == "__main__":
    import runpy
    sys.argv[0] = str(backend_launcher)
    runpy.run_path(str(backend_launcher), run_name="__main__")
