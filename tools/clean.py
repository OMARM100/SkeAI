"""Remove generated Python bytecode and SkeAI native extension artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
removed = 0

for path in ROOT.rglob("__pycache__"):
    if path.is_dir() and ".git" not in path.parts:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1

for path in (ROOT / "src" / "skeai").glob("_cpp*"):
    if path.is_file() and path.suffix in {".so", ".pyd", ".dylib"}:
        path.unlink()
        removed += 1

print(f"clean: removed {removed} generated item(s)")
