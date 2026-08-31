#!/usr/bin/env python3
"""Convenience entry point for the public prototype live controller."""

from pathlib import Path
import sys


# The simulator package uses a package-relative import layout.  Add both the
# checkout parent and the source distribution so this file works when invoked
# directly from any directory, without requiring PYTHONPATH.
_simulator_root = Path(__file__).resolve().parent
_checkout_root = _simulator_root.parent
for _path in (_checkout_root, _checkout_root / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load_controller_main():
    """Return the controller entry point.

    Keep this as a normal package import.  PyInstaller's archive importer
    resolves the package-relative imports correctly; manufacturing a stub
    ``physical_lab`` package works from source but prevents that importer from
    finding the controller inside a one-file executable.
    """

    from simulator.physical_lab.prototype_controller import main

    return main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_load_controller_main()())
