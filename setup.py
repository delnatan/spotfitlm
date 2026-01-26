"""
Build configuration for spotfitlm C extension.

This file is required by setuptools to build the C extension module.
All metadata is now in pyproject.toml.
"""

import platform

import numpy
from setuptools import Extension, setup

# Platform-specific compiler flags
extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    # MSVC is the default on Windows with cibuildwheel
    extra_compile_args = ["/O2"]
else:
    # GCC/Clang on Linux/macOS
    extra_compile_args = ["-O3"]
    extra_link_args = ["-lm"]  # Link math library on Unix

# Define the C extension
spotfitlm_extension = Extension(
    "spotfitlm.libspotfitlm",
    sources=[
        "c_src/gfit.c",
        "c_src/glm_core.c",
        "c_src/matrix_operations.c",
        "c_src/objective_funcs.c",
        "c_src/user_funcs.c",
    ],
    include_dirs=[
        numpy.get_include(),
        "c_src",
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c",
)

setup(ext_modules=[spotfitlm_extension])
