"""Python interface for the Adaptive Multigrid Solvers' PointInterpolant."""

__author__ = "Ricard Campos"
__email__ = "ricard.campos@coronis.es"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("py_ams_point_interpolant")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "unknown"

try:
    from .ams_point_interpolant import (  # noqa: F401
        Interpolant2D,
        Interpolant3D,
        ams_point_interpolant_2d,
        ams_point_interpolant_3d,
        get_parallel_types,
    )
except ImportError as e:
    raise ImportError(
        "Could not import the compiled 'ams_point_interpolant' extension module. "
        "If you are installing from source, make sure the C++ extension was built "
        "(pip install .). Original error: " + str(e)
    ) from e
