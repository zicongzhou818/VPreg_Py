# VPreg_py

VPreg_py is a GPU-first PyTorch port of the MATLAB VPreg framework for
three-dimensional non-rigid image registration. It is intended primarily for
neuroscience workflows in which rigid or affine alignment is performed first,
followed by deformable registration with VPreg.

> **Research software status:** this port is under active numerical validation.
> It is not intended for clinical diagnosis, treatment planning, or other
> safety-critical use.

## Overview

The current workflow is:

1. Rigidly or affinely align the moving image to the fixed image with ANTsPyX
   or another registration package.
2. Resample both images onto the same three-dimensional voxel grid.
3. Run `VPreg_gpu` to estimate forward and inverse non-rigid coordinate maps.
4. Save the warped images and coordinate maps, then calculate registration
   metrics with `evaluations3D`.

The included tutorial starts at step 3. Automated ANTsPyX rigid/affine
preprocessing is a planned integration rather than part of the current
`test_VPreg_gpu.py` script.

### Main capabilities

- GPU-first optimization with PyTorch tensors
- Forward and inverse three-dimensional coordinate maps
- MATLAB-compatible mutual information, Jacobian, Dice, and SSD calculations
- Linear GPU interpolation for iterative image warping and regridding
- GPU-capable three-dimensional Makima interpolation for final image sampling
- ANTsPyX-compatible Python environment and NIfTI-based tutorial
- MATLAB reference sources and an automated Pytest regression suite

## Supported environment

The verified development environment is:

| Component | Verified version |
|---|---:|
| Python | 3.12 |
| ANTsPyX | 0.6.3 |
| PyTorch | 2.14.0 with CUDA 13.0 |
| NumPy | 2.3.5 |
| SciPy | 1.15.3 |
| NiBabel | 5.4.2 |
| scikit-image | 0.26.0 |
| Operating system | Windows |

An NVIDIA GPU and a compatible display driver are required by the supported
configuration. The CUDA runtime is included in the PyTorch wheel. The bundled
256 × 256 × 150 development example used approximately 4.3 GiB of peak GPU
memory; memory use will vary with image dimensions.

Other operating systems and dependency versions may work but have not yet been
validated.

## Installation

Clone the repository and open PowerShell in its root directory:

```powershell
git clone <your-repository-url>
cd VPreg_py
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cu130.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch is installed separately because its CUDA wheel uses the PyTorch package
index. `requirements.txt` contains the direct scientific and testing
dependencies. `requirements-lock.txt` records the complete package set tested
on the development workstation.

Verify the environment before running a registration:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
```

## Tutorial: register a prealigned image pair

### 1. Prepare the inputs

The tutorial expects two floating-point NIfTI volumes in the repository root:

- `img_mov.nii.gz`: moving image
- `img_fix.nii.gz`: fixed image

Both volumes must be three-dimensional, have the same array shape, and already
occupy the same prealigned voxel grid after rigid or affine preprocessing. Edit
`MOVING_IMAGE_PATH` and `FIXED_IMAGE_PATH` in `test_VPreg_gpu.py` to use other
filenames.

Research images should be distributed only when their provenance, consent,
de-identification, and data-use terms permit public redistribution. If the
example pair is not distributed with the repository, copy an authorized pair
into place before running the tutorial.

### 2. Run VPreg

```powershell
.\.venv\Scripts\python.exe test_VPreg_gpu.py
```

`TEST_ID` controls the output-name suffix. With the current value
`TEST_ID = "test_Py_"`, the script writes:

| Output | Description |
|---|---|
| `img_mov_VPreg_gpu_test_Py_Warped.nii.gz` | Moving image warped onto the fixed-image grid |
| `img_mov_VPreg_gpu_test_Py_1Warp.nii.gz` | Forward coordinate map |
| `img_mov_VPreg_gpu_test_Py_InverseWarped.nii.gz` | Fixed image warped onto the moving-image grid |
| `img_mov_VPreg_gpu_test_Py_1InverseWarp.nii.gz` | Inverse coordinate map |
| `img_mov_VPreg_gpu_test_Py_evaluate_tabular.txt` | Runtime and registration metrics |

The warped scalar images retain the corresponding reference-image spatial
metadata. The saved coordinate maps have shape `(nx, ny, nz, 3)` and contain
absolute `(x, y, z)` voxel coordinates using MATLAB's 1-based convention. They
are not displacement vectors and are not directly interchangeable with ANTs
physical-space transforms.

### 3. Interpret the evaluation report

`evaluations3D` writes forward and inverse values for:

| Field | Interpretation |
|---|---|
| `time` | Optimizer or inverse-solver wall time in seconds; hardware dependent |
| `Dice` | Dice overlap after independent Otsu binarization; higher is better |
| `SSD_ratio` | Registered SSD divided by unregistered SSD; lower is better |
| `MI_incr_%` | Percentage increase in mutual information; higher is better |
| `JDmin`, `JDmax` | Minimum and maximum Jacobian determinant of the coordinate map |

A positive minimum Jacobian indicates that no folding was detected by this
voxel-grid calculation. A range closer to 1 indicates less local volume change,
but the appropriate range depends on the anatomy and registration objective.

One linear-regridding run of the development example on an NVIDIA RTX 4060
produced the following representative result. Exact runtimes and small numerical
differences can vary by hardware and software build.

| Direction | Time (s) | Dice | SSD ratio | MI increase (%) | JD minimum | JD maximum |
|---|---:|---:|---:|---:|---:|---:|
| Forward | 25.288 | 0.98784 | 0.18706 | 47.278 | 0.20891 | 2.4535 |
| Inverse | 0.71785 | 0.96581 | 0.17593 | 26.631 | 0.40909 | 4.1346 |

## Python API

The main function can also be called directly:

```python
from VPreg_gpu import VPreg_gpu

(
    phi,
    moving_warped,
    phi_inverse,
    fixed_inverse_warped,
    forward_time,
    inverse_time,
) = VPreg_gpu(moving, fixed)
```

`moving` and `fixed` may be three-dimensional NumPy arrays, PyTorch tensors, or
ANTsPyX image objects. They must have the same shape and contain real
floating-point data.

- NumPy and ANTsPyX inputs use CUDA automatically when it is available.
- Tensor inputs retain their current device.
- Returned tensors remain on the compute device.
- The optimizer performs explicit updates under `torch.no_grad()` and is not an
  autograd-differentiable registration layer.

Shared public utilities are imported from `utilities.py`:

```python
from utilities import (
    compute_JD3D,
    compute_JD_and_Curl3D,
    evaluations3D,
    gpu_dst_exact,
    gpuPoissonSolver4D,
    makima3d,
    matrixpad3D,
    mutualInfo,
    voxel2zscore,
)
```

## Implementation notes

- Differential calculations, image-force updates, DST/Poisson solves, ordinary
  iterative warps, and restart/regridding composition remain on the compute
  device.
- PyTorch performs volumetric trilinear sampling through five-dimensional
  `grid_sample` tensors.
- The two final unnormalized image warps use the local `makima3d` PyTorch
  implementation. It constructs modified-Akima nodal weights along each axis,
  uses their tensor products to estimate mixed partial derivatives from raw
  voxel differences, and evaluates tricubic Hermite patches in chunks on the
  source tensor's device. Iterative warps and restart/regridding composition
  remain trilinear.
- On the bundled 256 × 256 × 150 example, sampling through MATLAB's saved
  coordinate map differed from MATLAB's Makima-warped image by approximately
  3.8 × 10⁻⁶ mean absolute intensity and 1.0 × 10⁻⁷ relative L2 error.
- The original MATLAB Poisson/DST normalization is intentionally retained for
  initial numerical parity because the optimizer parameters were tuned with
  that scaling. Correcting the normalization and retuning are deferred.

## Project layout

| Path | Purpose |
|---|---|
| `VPreg_gpu.py` | Forward registration and inverse-map workflow |
| `VP_LM_simplified.py` | Simplified inverse coordinate-map solver |
| `utilities.py` | Shared numerical, interpolation, and evaluation routines |
| `test_VPreg_gpu.py` | NIfTI tutorial and output writer |
| `tests/` | Unit, parity, device, and end-to-end regression tests |
| `scripts/check_environment.py` | CUDA, ANTsPyX, FFT, and interpolation smoke test |
| `requirements*.txt` | Direct, CUDA-specific, and locked dependencies |
| `pyproject.toml` | Python project metadata and Pytest configuration |
| `*.m` | Original MATLAB reference implementations |

The former standalone Python modules for mutual information, padding, Jacobian
calculation, and evaluation were consolidated into `utilities.py`.

## Known limitations

- Rigid/affine preprocessing is not yet automated by the tutorial.
- Moving and fixed inputs must already share one voxel grid and array shape.
- The core currently validates array shape but does not compare physical-space
  metadata such as affine, origin, spacing, or direction.
- The verified environment requires an NVIDIA CUDA GPU.
- `makima3d` currently targets uniformly spaced, unit voxel grids with
  MATLAB-style 1-based coordinates; arbitrary nonuniform grid vectors are not
  supported.
- Coordinate-map NIfTI files are not standard ANTs transform files.
- Current optimizer constants retain the calibration of the MATLAB prototype.
- Validation is ongoing and currently uses a limited number of image pairs.

## Data distribution

The two tutorial NIfTI inputs should be added to a public repository only after
confirming that they are authorized for redistribution and sufficiently
de-identified. Empty NIfTI text headers alone do not establish that anatomical
image data are anonymous.

Data provenance and licensing information should accompany any example images
distributed with the project.

## License

VPreg_py is licensed under the **GNU General Public License, version 2 only**
(`GPL-2.0-only`). You may redistribute and modify the software under the terms
of that license. See the
[GNU GPL v2 license](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
for the complete terms and warranty disclaimer.

The public repository should include the full license text in a top-level
`LICENSE` file.

## Citation

If you use this code in your research, please cite the following paper:

> Zicong Zhou, Baihan Zhao, Andreas Mang, and Guojun Liao. (2025). **VPREG: An Optimal Control Formulation for Diffeomorphic Image Registration Based on the Variational Principle Grid Generation Method**. *arXiv preprint arXiv:2510.13109*. [arXiv:2510.13109](https://arxiv.org/abs/2510.13109)
