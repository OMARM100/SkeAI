"""Build SkeAI's C++ engine extension."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path


def find_compiler() -> str:
    configured = os.environ.get("CXX")
    if configured:
        return configured

    for candidate in ("c++", "clang++", "g++"):
        if shutil.which(candidate):
            return candidate

    raise RuntimeError(
        "No C++ compiler found. Set the CXX environment variable."
    )


def find_python_library() -> tuple[str, str]:
    libdir = sysconfig.get_config_var("LIBDIR")
    library = sysconfig.get_config_var("LDLIBRARY")

    if not libdir or not library:
        raise RuntimeError(
            "Python library location is unavailable."
        )

    library_path = Path(libdir) / library

    if not library_path.exists():
        raise RuntimeError(
            f"Python library was not found: {library_path}"
        )

    if not library.startswith("lib") or not library.endswith(
        (".so", ".a", ".dylib")
    ):
        raise RuntimeError(
            f"Unsupported Python library name: {library}"
        )

    library_name = library[3:]
    for suffix in (".so", ".a", ".dylib"):
        if library_name.endswith(suffix):
            library_name = library_name[:-len(suffix)]
            break

    return str(libdir), library_name


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "cpp" / "skeai_engine.cpp"
    package_dir = root / "src" / "skeai"

    if not source.exists():
        raise FileNotFoundError(
            f"C++ engine source not found: {source}"
        )

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError(
            "Python EXT_SUFFIX is unavailable."
        )

    for existing in package_dir.glob("_cpp*.so"):
        existing.unlink()

    output = package_dir / f"_cpp{extension_suffix}"

    include_dir = sysconfig.get_config_var("INCLUDEPY")
    if not include_dir:
        raise RuntimeError(
            "Python include directory is unavailable."
        )

    python_libdir, python_library = find_python_library()

    command = [
        find_compiler(),
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        f"-I{include_dir}",
        f"-L{python_libdir}",
        f"-Wl,-rpath,{python_libdir}",
        f"-l{python_library}",
    ]

    command.extend(
        shlex.split(
            sysconfig.get_config_var("CXXFLAGS") or ""
        )
    )
    command.extend(
        shlex.split(
            os.environ.get("CXXFLAGS", "")
        )
    )

    command.extend(
        [
            str(source),
            "-o",
            str(output),
        ]
    )

    print("Building SkeAI C++ engine:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
    )

    print(f"Linked Python runtime: {python_libdir}/lib{python_library}.so")
    print(f"Built: {output}")


if __name__ == "__main__":
    main()
