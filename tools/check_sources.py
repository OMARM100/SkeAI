"""Inventory and syntax-check supported SkeAI source files.

This deliberately checks the source tree, not only the C++ extension:
- every Python source is byte-compiled;
- C/C++ and assembly files are discovered and reported;
- the native C++ extension build remains the authoritative compile/link check.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {
    ".py", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
    ".asm", ".s", ".S",
}
IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "build",
    "dist", "site-packages",
}


def discover() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files, key=lambda p: str(p.relative_to(ROOT)))


def main() -> int:
    files = discover()
    if not files:
        print("ERROR: no source files found.", file=sys.stderr)
        return 1

    by_suffix: dict[str, int] = {}
    for path in files:
        by_suffix[path.suffix] = by_suffix.get(path.suffix, 0) + 1

    print("SkeAI source inventory")
    print("======================")
    for suffix, count in sorted(by_suffix.items()):
        print(f"{suffix}: {count}")
    print(f"Total source files: {len(files)}")
    print()

    python_files = [p for p in files if p.suffix == ".py"]
    for path in python_files:
        try:
            py_compile.compile(
                str(path),
                cfile=None,
                doraise=True,
                optimize=0,
            )
        except py_compile.PyCompileError as exc:
            print(f"FAIL Python syntax: {path.relative_to(ROOT)}: {exc.msg}")
            return 1
        print(f"PASS Python syntax: {path.relative_to(ROOT)}")

    native_files = [
        p for p in files
        if p.suffix.lower() in {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".asm", ".s"}
    ]
    if native_files:
        print()
        print("Native source files discovered (compiled by their build target):")
        for path in native_files:
            print(f"FOUND {path.relative_to(ROOT)}")

    print()
    print("PASS source discovery and Python syntax checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
