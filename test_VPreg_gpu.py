"""Run VPREG for the provided moving/fixed MRI test pair.

This is the Python counterpart of ``test_VPreg_gpu.m``.  The VPREG core and
evaluation routines are imported from their correspondingly named Python
modules as they are translated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from VPreg_gpu import VPreg_gpu
from utilities import evaluations3D


PROJECT_DIR = Path(__file__).resolve().parent
MOVING_IMAGE_PATH = PROJECT_DIR / "img_mov.nii.gz"
FIXED_IMAGE_PATH = PROJECT_DIR / "img_fix.nii.gz"
TEST_ID = "test_Py_"


def _nifti_stem(path: Path) -> str:
    """Return a NIfTI filename without its .nii or .nii.gz suffix."""
    if path.name.lower().endswith(".nii.gz"):
        return path.name[:-7]
    if path.suffix.lower() == ".nii":
        return path.stem
    raise ValueError(f"Expected a .nii or .nii.gz file, got: {path}")


def _load_float32_nifti(path: Path) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    """Load a NIfTI volume as float32 while retaining its spatial metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI input does not exist: {path}")

    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    return data, image


def _as_float32_numpy(value: Any) -> np.ndarray:
    """Move a PyTorch result to the CPU if needed and return float32 NumPy data."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _save_like_reference(
    data: Any,
    output_path: Path,
    reference: nib.spatialimages.SpatialImage,
) -> None:
    """Save a volume with the affine and header of a reference image."""
    array = _as_float32_numpy(data)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    output_image = reference.__class__(array, reference.affine, header=header)

    # MATLAB niftiwrite(info=...) retains the reference sform but writes no
    # qform.  Set the forms explicitly so the Python outputs follow that
    # behavior instead of nibabel's constructor defaults.
    output_image.set_qform(None, code=0)
    sform_code = int(reference.header["sform_code"])
    output_image.set_sform(reference.affine, code=sform_code or 2)
    nib.save(output_image, str(output_path))


def _save_deformation_field(data: Any, output_path: Path) -> None:
    """Save a deformation field with MATLAB niftiwrite-like default geometry."""
    array = _as_float32_numpy(data)
    output_image = nib.Nifti1Image(array, affine=None)
    output_image.header.set_data_dtype(np.float32)
    nib.save(output_image, str(output_path))


def main() -> None:
    moving, moving_image = _load_float32_nifti(MOVING_IMAGE_PATH)
    fixed, fixed_image = _load_float32_nifti(FIXED_IMAGE_PATH)

    print(f"\nThe {TEST_ID}-th registration started.")

    output_prefix = PROJECT_DIR / (
        f"{_nifti_stem(MOVING_IMAGE_PATH)}_VPreg_gpu_{TEST_ID}"
    )

    (
        phi,
        moving_warped,
        phi_inverse,
        fixed_inverse_warped,
        time_forward,
        time_inverse,
    ) = VPreg_gpu(moving, fixed)

    print(f"The {TEST_ID}-th registration done.")

    # The save helpers make temporary CPU/NumPy views at the NIfTI boundary;
    # the tensors themselves remain on the compute device for evaluations3D.
    _save_like_reference(
        moving_warped,
        Path(f"{output_prefix}Warped.nii.gz"),
        fixed_image,
    )
    _save_deformation_field(phi, Path(f"{output_prefix}1Warp.nii.gz"))
    _save_like_reference(
        fixed_inverse_warped,
        Path(f"{output_prefix}InverseWarped.nii.gz"),
        moving_image,
    )
    _save_deformation_field(
        phi_inverse,
        Path(f"{output_prefix}1InverseWarp.nii.gz"),
    )

    evaluations3D(
        moving,
        fixed,
        moving_warped,
        fixed_inverse_warped,
        phi,
        phi_inverse,
        str(output_prefix),
        time_forward,
        time_inverse,
    )


if __name__ == "__main__":
    main()
