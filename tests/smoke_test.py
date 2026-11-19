"""Smoke test run against every built wheel (see cibuildwheel's test-command).

Checks that the compiled module is importable from the package root on every
platform, that no stray build artifacts were packaged, and that a tiny 2D
interpolation runs end to end.
"""

import os
import sys

import numpy as np
from py_ams_point_interpolant.ams_point_interpolant import (
    ams_point_interpolant_2d,
    get_parallel_types,
)

import py_ams_point_interpolant


def check_package_layout():
    pkg_dir = os.path.dirname(py_ams_point_interpolant.__file__)
    entries = os.listdir(pkg_dir)
    bad = [
        e
        for e in entries
        if e in ("Release", "Debug") or e.startswith("point_interpolant")
    ]
    if bad:
        sys.exit(f"Unexpected entries in {pkg_dir}: {bad}")
    modules = [
        e
        for e in entries
        if e.startswith("ams_point_interpolant") and e.endswith((".so", ".pyd"))
    ]
    if len(modules) != 1:
        sys.exit(f"Expected exactly one compiled module in {pkg_dir}, got {modules}")


def check_interpolation():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(200, 2))
    vals = pts[:, 0] ** 2 + pts[:, 1] ** 2
    for exact in (False, True):
        interp = ams_point_interpolant_2d(
            pts, vals, np.array([]), np.array([]), depth=6, exact_interpolation=exact
        )
        out = interp.eval(pts[:10])
        if not np.all(np.isfinite(out)):
            sys.exit(f"Non-finite result (exact={exact}): {out}")
        err = np.abs(out - vals[:10]).max()
        if err > 0.05:
            sys.exit(f"Interpolation error too large (exact={exact}): {err}")


if __name__ == "__main__":
    print("py_ams_point_interpolant", py_ams_point_interpolant.__version__)
    print("parallel types:", get_parallel_types())
    check_package_layout()
    check_interpolation()
    print("Smoke test OK")
