import os
import platform

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

extra_compile_args = []
extra_link_args = []


if platform.system() == "Windows":
    if "GCC" in os.getenv("CC", ""):
        extra_compile_args = ["-O3"]  # GCC-style optimization
    else:  # Default to MSVC (Visual Studio)
        extra_compile_args = ["/O2"]

    extra_link_args = []  # You can add any Windows-specific linker flags here
else:
    extra_compile_args = ["-O3"]  # Linux/macOS-specific optimization
    extra_link_args = [
        "-lm"
    ]  # Link against the math library on Unix-based systems

# the extension will be created as spotfitlm/libspotfitlm*.so
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


# Custom build class to ensure NumPy headers are included
class CustomBuildExt(build_ext):
    def build_extensions(self):
        numpy_include = numpy.get_include()
        for ext in self.extensions:
            if isinstance(ext, Extension):
                ext.include_dirs.append(numpy_include)
        super().build_extensions()


setup(
    ext_modules=[spotfitlm_extension],
    cmdclass={"build_ext": CustomBuildExt},
)
