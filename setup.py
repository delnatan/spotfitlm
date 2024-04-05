import numpy
from setuptools import Extension, find_packages, setup

# December 2023:
# I've removed LAPACK dependencies so these are no longer needed
# ----------------------------------------------------------------------
# for Mac, compile with homebrew-installed lapack
# LAPACK_LIB = "/opt/homebrew/opt/lapack/lib"
# LAPACK_INCLUDE_PATH = "/opt/homebrew/opt/lapack/include"

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
    extra_compile_args=["-O3", "-Wall"],
    # extra_link_args=[f"-L{LAPACK_LIB}"],
    language="c",
)

setup(
    name="spotfitlm",
    version="0.1d",
    description="MLE 2D gaussian fitting with Poisson deviates",
    packages=find_packages(),
    ext_modules=[spotfitlm_extension],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
    ],
)
