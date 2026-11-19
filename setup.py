"""
Setup script for py_ams_point_interpolant
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import shutil
import glob as glob_module
import platform


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
        ext_fullpath = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.abspath(os.path.dirname(ext_fullpath))
        cfg = "Debug" if self.debug else "Release"

        if sys.platform == "win32":
            build_args = ["--config", cfg, "--", "/m"]
        else:
            nproc = os.cpu_count()
            build_args = ["--config", cfg, "--", f"-j{nproc}"]

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            # Multi-config generators (e.g., MSVC) append the configuration name
            # (Release/, Debug/) to the output directory unless the per-config
            # variables are set explicitly.
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG={ext_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        if sys.platform == "darwin":
            archs = os.environ.get("CIBW_ARCHS", platform.machine())
            deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")
            cmake_args += [
                f"-DCMAKE_OSX_ARCHITECTURES={archs}",
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}",
            ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

        # Move .so/.pyd
        patterns = [
            os.path.join(ext_dir, "ams_point_interpolant*.so"),
            os.path.join(ext_dir, "ams_point_interpolant*.pyd"),
            os.path.join(ext_dir, "Release", "ams_point_interpolant*.so"),
            os.path.join(ext_dir, "Release", "ams_point_interpolant*.pyd"),
            os.path.join(ext_dir, "Debug", "ams_point_interpolant*.so"),
            os.path.join(ext_dir, "Debug", "ams_point_interpolant*.pyd"),
        ]

        built_libs = []
        for pattern in patterns:
            built_libs.extend(glob_module.glob(pattern))

        if not built_libs:
            raise RuntimeError("CMake build succeeded but no .so/.pyd file was found")

        # Make sure the module ends up exactly where setuptools expects it, and
        # only once (the module name must match PYBIND11_MODULE in the .cpp).
        expected_path = os.path.abspath(ext_fullpath)
        built_lib = os.path.abspath(built_libs[0])
        if built_lib != expected_path:
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)
            shutil.move(built_lib, expected_path)


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
    ext_modules=[CMakeExtension("py_ams_point_interpolant.ams_point_interpolant")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={
        "py_ams_point_interpolant": ["*.so", "*.pyd", "*.dll"],
    },
    zip_safe=False,
)
