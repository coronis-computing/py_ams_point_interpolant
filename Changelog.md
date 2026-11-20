# Changelog

All notable changes to this project will be documented in this file.

## v0.1.1

* Fixed the import of the compiled module on Windows. The extension was declared with the wrong name (`point_interpolant` instead of `ams_point_interpolant`) and, with MSVC, the actual module was placed in a `Release/` subfolder of the package, so `from py_ams_point_interpolant.ams_point_interpolant import ...` failed. The wheels now contain a single `ams_point_interpolant` module at the package root on all platforms.
* Added support for Python 3.13 and 3.14: updated pybind11 to v3.0.1 and wheels are now built for CPython 3.10-3.14 on Linux, Windows and macOS (arm64).
* Fixed `exact_interpolation=True`, which crashed (segmentation fault) or returned a zero solution.
* Fixed passing gradients without values (the dimension of the problem was read from the number of points instead of the number of coordinates).
* The main functions and classes (`ams_point_interpolant_2d`, `ams_point_interpolant_3d`, `Interpolant2D`, `Interpolant3D`, `get_parallel_types`) can now be imported directly from `py_ams_point_interpolant`, which also exposes `__version__`. If the compiled module cannot be loaded, an `ImportError` with an informative message is raised.
* Every built wheel is now tested in CI (`tests/smoke_test.py`): it checks the package layout and runs a small 2D interpolation, both approximate and exact.
* Packaging fixes: removed the console scripts pointing to non-existent modules, and the source distribution now includes the readme, license, changelog, demos and tests.

## v0.1.0

* First release: Python bindings (pybind11) for the adaptive multigrid point interpolant of [PoissonRecon](https://github.com/mkazhdan/PoissonRecon), for 2D and 3D scattered data with values and/or gradients.
* Wheels for CPython 3.10-3.12 on Linux, Windows and macOS.
