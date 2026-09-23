"""Make OpenGL use the discrete NVIDIA GPU on Windows hybrid-graphics laptops.

On Optimus laptops a process gets the integrated GPU for OpenGL unless the NVIDIA driver
is already loaded when the first GL context is created. Loading nvcuda.dll is enough and
makes MuJoCo rendering ~60x faster (measured 26 -> 1625 fps at 256x256 on an RTX 5060
laptop). No-op elsewhere; disable with PAI_PREFER_NVIDIA_GL=0.
"""

import os
import sys


def prefer_nvidia_gl() -> None:
    if sys.platform != "win32" or os.environ.get("PAI_PREFER_NVIDIA_GL", "1") == "0":
        return
    try:
        import ctypes

        ctypes.WinDLL("nvcuda.dll")
    except OSError:
        pass  # no NVIDIA driver: keep the default GPU
