# Python Adaptive Multigrid Solver Point Interpolant (py_ams_point_interpolant)

Python interface for the PointInterpolant executable of the Adaptive Multigrid Solvers (a.k.a. PoissonRecon) project.

## Installation

### Requirements

- Python >= 3.10
- CMake >= 3.20
- NumPy >= 1.19.0
- pybind11 >= 2.11.1 (will be automatically downloaded by CMake)
- [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) (will be automatically downloaded by CMake)
- OpenMP (optional, for parallel computation)

### From PyPI

```
pip install py_ams_point_interpolant
```

### From sources

This project requires Python >= 3.10 and CMake to build (cmake should automatically download pybind and PoissonRecon sources, also required). To build and install it using `pip`:

```
cd <path_to>/py_ams_point_interpolant
pip install .
```

## Demos

You have examples of usage in the `demos` folder for 2D and 3D interpolation. To run them from the folder containing this readme:

```
python demo/demo_2d.py
python demo/demo_3d.py
```
