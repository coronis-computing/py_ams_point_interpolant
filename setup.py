"""
Setup script for py_ams_point_interpolant
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    HAS_PYBIND11 = True
except ImportError:
    HAS_PYBIND11 = False


class CMakeBuild(build_ext):
    def run(self):
        # Check if CMake is available
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Get paths
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            # f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += [
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_args += ["--", "-j4"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


class CMakeExtension(Extension):
    """Extension that uses CMake to build"""

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


# Read README
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as f:
        return f.read()


setup(
    name="py_ams_point_interpolant",
    packages=find_packages(),
    ext_modules=[CMakeExtension("py_ams_point_interpolant.point_interpolant")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={
        "py_ams_point_interpolant": ["*.so", "*.pyd", "*.dll"],
    },
    zip_safe=False,
)
