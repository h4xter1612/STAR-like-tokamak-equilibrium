"""
utils.py

Utility helpers to check (and optionally install) all Python dependencies
required for the STAR-like spherical tokamak project.

Usage (from the project root):

    python utils.py

This will:
  * Check imports for all required packages.
  * Attempt to install any missing package via: python -m pip install <name>.
  * Print a summary report at the end.

You can also import and call `ensure_environment()` from other scripts:

    from utils import ensure_environment
    ensure_environment(auto_install=True)

NOTE:
  - This only handles Python packages via pip.
  - External tools such as `ffmpeg` (needed for Matplotlib animations
    when saving .mp4) must be installed at the system level.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Dict

try:
    # Python 3.8+
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    # Fallback for very old Python versions
    import importlib_metadata  # type: ignore


# ----------------------------------------------------------------------
# Configuration: modules and corresponding pip packages
# ----------------------------------------------------------------------

@dataclass
class Dependency:
    module_name: str   # name used in "import <module_name>"
    pip_name: str      # name used in "pip install <pip_name>"
    min_version: Optional[str] = None  # e.g. "1.22.0" (optional)


REQUIRED_DEPENDENCIES = [
    # Core scientific stack
    Dependency("numpy",      "numpy"),
    Dependency("matplotlib", "matplotlib"),
    Dependency("scipy",      "scipy"),

    # Progress / UI helpers
    Dependency("tqdm",       "tqdm"),

    # FreeGS / FreeGSNKE ecosystem
    # NOTE: if these fail to install from pip, you may need to install them
    # manually from their Git repository or local source.
    Dependency("freegs4e",   "freegs4e"),
    Dependency("freegsnke",  "freegsnke"),
]


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _get_distribution_version(dist_name: str) -> Optional[str]:
    """
    Try to get the installed distribution's version from importlib.metadata.
    Returns None if the distribution is not found.
    """
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _import_module(module_name: str) -> bool:
    """
    Try to import a module by name.

    Returns True if the import succeeds, False otherwise.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _install_with_pip(pip_name: str) -> bool:
    """
    Try to install a package using pip:

        python -m pip install <pip_name>

    Returns True if the command exits with code 0, False otherwise.
    """
    print(f"  -> Installing via pip: {pip_name!r}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name]
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"     [ERROR] pip failed for {pip_name!r} (exit code {exc.returncode}).")
        return False
    except FileNotFoundError:
        print("     [ERROR] Could not execute pip; make sure pip is available.")
        return False


def _check_single_dependency(
    dep: Dependency,
    auto_install: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Check a single dependency. Optionally attempt to install it if missing.

    Returns a small dict with keys:
      - 'module': module_name
      - 'pip': pip_name
      - 'ok': 'yes' / 'no'
      - 'version': detected version (if any)
      - 'note': additional text (e.g. error message or hints)
    """
    module_name = dep.module_name
    pip_name = dep.pip_name

    print(f"\nChecking dependency: '{module_name}' (pip: '{pip_name}')")

    # First, try importing the module
    if _import_module(module_name):
        version = _get_distribution_version(pip_name) or "unknown"
        print(f"  -> OK, module '{module_name}' is importable, version = {version}")
        note = ""
        ok = "yes"
    else:
        print(f"  -> Module '{module_name}' not found.")
        ok = "no"
        version = None
        note = "missing"

        if auto_install:
            print("  -> Attempting to install it...")
            if _install_with_pip(pip_name) and _import_module(module_name):
                version = _get_distribution_version(pip_name) or "unknown"
                ok = "yes"
                note = "installed via pip"
                print(f"  -> Successfully installed '{pip_name}', version = {version}")
            else:
                note = (
                    "could not be installed automatically – "
                    "please install it manually and re-run utils.py"
                )
                print(f"  -> WARNING: {note}")

    # Minimal version check (if requested)
    if ok == "yes" and dep.min_version is not None and version is not None:
        from packaging.version import Version  # only used if installed
        if Version(version) < Version(dep.min_version):
            note = f"version {version} < required {dep.min_version}"
            print(f"  -> WARNING: {note}")
            ok = "no"

    return {
        "module": module_name,
        "pip": pip_name,
        "ok": ok,
        "version": version,
        "note": note,
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def ensure_environment(auto_install: bool = True) -> None:
    """
    Check (and optionally install) all required dependencies for this project.

    Parameters
    ----------
    auto_install : bool, optional
        If True (default), attempt to automatically install missing
        packages using pip. If False, only perform checks.

    This function prints a summary report at the end.
    """
    print("============================================================")
    print("  STAR-like tokamak environment check")
    print("  Python executable :", sys.executable)
    print("  Python version    :", sys.version.split()[0])
    print("============================================================")

    summary = []
    for dep in REQUIRED_DEPENDENCIES:
        info = _check_single_dependency(dep, auto_install=auto_install)
        summary.append(info)

    # Final summary
    print("\n============================================================")
    print("Summary of dependencies:")
    print("------------------------------------------------------------")
    for info in summary:
        status = "OK " if info["ok"] == "yes" else "MISSING"
        ver = info["version"] or "n/a"
        note = info["note"] or ""
        print(
            f"{status:8s} "
            f"module={info['module']:<12s} "
            f"(pip={info['pip']:<12s}) "
            f"version={ver:<12s} "
            f"{note}"
        )
    print("------------------------------------------------------------")

    missing = [s for s in summary if s["ok"] != "yes"]
    if missing:
        print(
            "\nOne or more dependencies are still missing or have "
            "unsatisfied version constraints."
        )
        print("Please install them manually (e.g. with pip) and re-run utils.py.")
    else:
        print("\nAll required dependencies appear to be available. ✅")
    print("============================================================")


# ----------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------

def _parse_cli_args():
    """
    Very small CLI: currently only supports a --no-install flag.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Check and (optionally) install Python dependencies "
                    "for the STAR-like tokamak project."
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Only check dependencies; do not attempt to install missing ones.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    ensure_environment(auto_install=not args.no_install)

