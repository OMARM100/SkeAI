"""Build SkeAI's optional native C numerical backend."""

from __future__ import annotations

import os
import shlex
import subprocess
import sysconfig
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "native" / "skeai_native.c"
    package_dir = root / "src" / "skeai"
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")

    if not extension_suffix:
        raise RuntimeError("Python EXT_SUFFIX is unavailable.")

    output = package_dir / f"_native{extension_suffix}"
    output.unlink(missing_ok=True)

    ldshared = sysconfig.get_config_var("LDSHARED")
    if not ldshared:
        ldshared = f"{os.environ.get('CC', 'cc')} -shared"

    include_dir = sysconfig.get_config_var("INCLUDEPY")
    if not include_dir:
        raise RuntimeError("Python include directory is unavailable.")

    command = shlex.split(ldshared)
    command.extend(shlex.split(os.environ.get("CFLAGS", "")))
    command.extend(["-O3", "-fPIC", f"-I{include_dir}", str(source), "-o", str(output)])

    print("Building SkeAI native backend:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"Built: {output}")


if __name__ == "__main__":
    main()
