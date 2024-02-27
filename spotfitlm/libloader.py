import ctypes
import os
import sys

import numpy as np


def load_library():
    """handle system-specific C-library loading"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    suffix = f"{sys.implementation.cache_tag}-{sys.implementation._multiarch}"

    if sys.platform == "darwin" or sys.platform == "linux":
        extension = "so"
    else:
        extension = "dll"

    lib_filename = f"libspotfitlm.{suffix}.{extension}"
    lib_path = os.path.join(dir_path, lib_filename)

    try:
        print(f"loading {lib_path}")
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise OSError(f"Failed to load shared library {lib_path}.") from e

    lib.fit_symmetric_gaussian.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags="C_CONTIGUOUS"),
    ]

    lib.fit_symmetric_gaussian.restype = None

    return lib
