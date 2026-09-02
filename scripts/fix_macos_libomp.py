"""Make LightGBM importable on macOS without Homebrew.

THE PROBLEM
-----------
LightGBM's macOS wheel is compiled with OpenMP for multithreaded training, so
`lib_lightgbm.dylib` has a load-time dependency on `@rpath/libomp.dylib`. Apple's
clang does not ship an OpenMP runtime, and the only search paths baked into the
wheel are Homebrew's and MacPorts':

    /opt/homebrew/opt/libomp/lib/
    /opt/local/lib/libomp/

On a machine with neither package manager, both are absent and `import lightgbm`
dies with an OSError before any of our code runs.

The documented upstream fix is `brew install libomp`. This script is the equivalent
for a machine without Homebrew, so the project does not require installing a system
package manager just to import one library.

THE FIX
-------
scikit-learn ships its own vendored `libomp.dylib` (in `sklearn/.dylibs/`) because it
has the same requirement. We copy that runtime next to `lib_lightgbm.dylib` and add
`@loader_path` — "the directory I was loaded from" — to LightGBM's runtime search
path. LightGBM then finds OpenMP beside itself.

Copying rather than pointing an rpath at sklearn's directory is deliberate: it keeps
LightGBM self-contained, so upgrading or removing scikit-learn later cannot silently
break model training.

WHY IT IS A SCRIPT, NOT A MANUAL COMMAND
----------------------------------------
It patches a file inside site-packages, so it does NOT survive `pip install
--force-reinstall lightgbm` or a rebuilt venv. Making it an idempotent, re-runnable
script means recovery is one command, and `make setup` can call it automatically.

Safe to run repeatedly: it exits early if LightGBM already imports.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Paths that a Homebrew or MacPorts install would provide, checked first so that a
# machine which later installs libomp properly uses the system copy.
SYSTEM_LIBOMP_CANDIDATES = [
    Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
    Path("/usr/local/opt/libomp/lib/libomp.dylib"),
    Path("/opt/local/lib/libomp/libomp.dylib"),
]


def lightgbm_imports() -> bool:
    """Try importing LightGBM in a SUBPROCESS.

    A subprocess is essential: dyld resolves a shared library's dependencies once,
    when it is first loaded into a process. If we tried and failed to import
    lightgbm in *this* process, a later retry after the fix would still see the
    cached failure. A fresh interpreter gives an honest answer every time.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import lightgbm"],
        capture_output=True,
    )
    return result.returncode == 0


def find_libomp() -> Path | None:
    """Locate an OpenMP runtime: a system copy if present, else sklearn's vendored one."""
    for candidate in SYSTEM_LIBOMP_CANDIDATES:
        if candidate.exists():
            return candidate

    try:
        import sklearn
    except ImportError:
        return None

    vendored = Path(sklearn.__file__).parent / ".dylibs" / "libomp.dylib"
    return vendored if vendored.exists() else None


def lightgbm_lib_dir() -> Path | None:
    """Directory holding lib_lightgbm.dylib.

    Imported via importlib rather than `import lightgbm` because importing the
    package is exactly what currently fails — but locating its files on disk does
    not require executing it.
    """
    import importlib.util

    spec = importlib.util.find_spec("lightgbm")
    if spec is None or not spec.origin:
        return None
    lib_dir = Path(spec.origin).parent / "lib"
    return lib_dir if lib_dir.is_dir() else None


def existing_rpaths(dylib: Path) -> list[str]:
    """Read the runtime search paths already baked into a dylib via `otool -l`.

    Used to keep this script idempotent — adding the same rpath twice is harmless
    at runtime but pollutes the binary, and re-running should be a clean no-op.
    """
    out = subprocess.run(
        ["otool", "-l", str(dylib)], capture_output=True, text=True
    ).stdout
    paths: list[str] = []
    lines = out.splitlines()
    for i, line in enumerate(lines):
        # An LC_RPATH load command prints its value on a line a couple below it,
        # formatted as: "path <value> (offset N)"
        if "LC_RPATH" in line:
            for follow in lines[i : i + 4]:
                stripped = follow.strip()
                if stripped.startswith("path "):
                    paths.append(stripped.split(" ", 1)[1].split(" (offset")[0])
                    break
    return paths


def main() -> int:
    if sys.platform != "darwin":
        print("Not macOS — nothing to do.")
        return 0

    if lightgbm_imports():
        print("LightGBM already imports cleanly — no patch needed.")
        return 0

    print("LightGBM fails to import; attempting the OpenMP fix.\n")

    lib_dir = lightgbm_lib_dir()
    if lib_dir is None:
        print("FAIL: could not locate the lightgbm package. Is it installed?")
        return 1

    target_dylib = lib_dir / "lib_lightgbm.dylib"
    if not target_dylib.exists():
        print(f"FAIL: {target_dylib} not found.")
        return 1

    source_libomp = find_libomp()
    if source_libomp is None:
        print(
            "FAIL: no libomp.dylib found on this machine.\n"
            "      Install scikit-learn (which vendors one), or `brew install libomp`."
        )
        return 1
    print(f"  found OpenMP runtime: {source_libomp}")

    destination = lib_dir / "libomp.dylib"
    if not destination.exists():
        shutil.copy2(source_libomp, destination)
        print(f"  copied -> {destination}")
    else:
        print(f"  already present: {destination}")

    # @loader_path resolves to the directory containing lib_lightgbm.dylib at load
    # time. Using it rather than an absolute path keeps the venv relocatable.
    if "@loader_path" in existing_rpaths(target_dylib):
        print("  @loader_path rpath already present")
    else:
        proc = subprocess.run(
            ["install_name_tool", "-add_rpath", "@loader_path", str(target_dylib)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"FAIL: install_name_tool failed:\n{proc.stderr}")
            return 1
        print("  added @loader_path to lib_lightgbm.dylib rpaths")

    if lightgbm_imports():
        print("\nPASS — LightGBM now imports.")
        return 0

    print("\nFAIL — LightGBM still does not import. Run `import lightgbm` for detail.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
