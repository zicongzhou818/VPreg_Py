"""Smoke-test the VPreg_py Python, CUDA, ANTsPyX, and interpolation stack."""

from __future__ import annotations

import platform
import sys

import ants
import nibabel
import numpy as np
import scipy
import skimage
import torch
import torch.nn.functional as torch_f
from scipy.interpolate import Akima1DInterpolator


def check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA CUDA device")

    device = torch.device("cuda")
    volume = torch.arange(27, dtype=torch.float32, device=device).reshape(
        1, 1, 3, 3, 3
    )
    theta = torch.eye(3, 4, dtype=torch.float32, device=device).unsqueeze(0)
    grid = torch_f.affine_grid(theta, volume.shape, align_corners=True)
    warped = torch_f.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    torch.testing.assert_close(warped, volume)

    spectrum = torch.fft.fftn(volume)
    if not torch.isfinite(spectrum).all():
        raise RuntimeError("CUDA FFT produced non-finite values")


def check_ants() -> None:
    data = np.zeros((9, 10, 11), dtype=np.float32)
    data[3:6, 4:7, 5:8] = 1.0
    origin = (1.0, -2.0, 3.0)
    spacing = (0.8, 0.9, 1.2)
    direction = np.eye(3, dtype=np.float64)

    image = ants.from_numpy(
        data,
        origin=origin,
        spacing=spacing,
        direction=direction,
    )
    np.testing.assert_allclose(image.numpy(), data)
    np.testing.assert_allclose(image.origin, origin)
    np.testing.assert_allclose(image.spacing, spacing)
    np.testing.assert_allclose(image.direction, direction)

    resampled = ants.resample_image_to_target(image, image, interp_type="linear")
    np.testing.assert_allclose(resampled.numpy(), data, atol=1e-6)


def check_makima() -> None:
    x = np.arange(7, dtype=np.float64)
    y = np.array([-1, -1, -1, 0, 1, 1, 1], dtype=np.float64)
    query = np.linspace(0, 6, 31)
    values = Akima1DInterpolator(x, y, method="makima")(query)
    if not np.isfinite(values).all():
        raise RuntimeError("SciPy Makima interpolation produced non-finite values")


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Expected Python 3.12, found {platform.python_version()}")

    check_cuda()
    check_ants()
    check_makima()

    print(f"Python:       {platform.python_version()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA device:  {torch.cuda.get_device_name(0)}")
    print(f"ANTsPyX:      {ants.__version__}")
    print(f"NumPy:        {np.__version__}")
    print(f"SciPy:        {scipy.__version__}")
    print(f"NiBabel:      {nibabel.__version__}")
    print(f"scikit-image: {skimage.__version__}")
    print("Environment checks passed.")


if __name__ == "__main__":
    main()

