"""Shared numerical, interpolation, and evaluation utilities for VPreg_py."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


__all__ = [
    "voxel2zscore",
    "mutualInfo",
    "matrixpad3D",
    "compute_JD3D",
    "compute_JD_and_Curl3D",
    "gpuPoissonSolver4D",
    "gpu_dst_exact",
    "evaluations3D",
]


def _as_floating_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.is_complex() or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a real floating-point array; got {tensor.dtype}.")
    return tensor


def voxel2zscore(img: Any) -> torch.Tensor:
    """Normalize all voxels using MATLAB's sample standard deviation.

    This reproduces ``std(img, 0, "all")``, whose denominator is ``N - 1``.
    Tensor inputs retain their dtype and device; array-like inputs become CPU
    tensors.  As in MATLAB, a constant image produces NaNs rather than being
    silently replaced with zeros.
    """
    image = _as_floating_tensor(img, "img")
    if image.numel() == 0:
        raise ValueError("img must contain at least one voxel.")

    image_mean = torch.mean(image)
    image_std = torch.std(image, correction=1)
    scale = torch.reciprocal(image_std)
    return (image - image_mean) * scale


def _as_field_triplet(
    phi_1: Any,
    phi_2: Any,
    phi_3: Any,
    *,
    promote: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = (phi_1, phi_2, phi_3)
    existing = [value for value in values if isinstance(value, torch.Tensor)]
    devices = {value.device for value in existing}
    if len(devices) > 1:
        names = ", ".join(str(device) for device in sorted(devices, key=str))
        raise ValueError(f"All deformation components must use one device; got {names}.")
    device = existing[0].device if existing else None

    tensors = tuple(
        value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
        for value in values
    )
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors[1:]):
        shapes = ", ".join(str(tuple(tensor.shape)) for tensor in tensors)
        raise ValueError(f"Deformation components must have the same shape; got {shapes}.")
    if len(shape) != 3:
        raise ValueError(
            "Deformation components must be three-dimensional; "
            f"got shape {tuple(shape)}."
        )
    if any(size < 2 for size in shape):
        raise ValueError(
            "Every spatial dimension must contain at least two samples; "
            f"got shape {tuple(shape)}."
        )
    for name, tensor in zip(("Phi1", "Phi2", "Phi3"), tensors):
        _as_floating_tensor(tensor, name)

    if promote:
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        return tuple(tensor.to(dtype=dtype) for tensor in tensors)
    return tensors


def _gradient_xyz(
    field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return derivatives along VPreg's x, y, and z array dimensions."""
    return torch.gradient(
        field,
        spacing=1.0,
        dim=(0, 1, 2),
        edge_order=1,
    )


def compute_JD_and_Curl3D(
    Phi1: Any,
    Phi2: Any,
    Phi3: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the Jacobian determinant and curl of a 3-D coordinate field.

    The output order is identical to the MATLAB helper:
    ``(JD_phi, CV_phi1, CV_phi2, CV_phi3)``.  Derivatives use centered
    differences internally and first-order one-sided boundary differences,
    matching MATLAB ``gradient`` for unit voxel spacing.
    """
    phi_1, phi_2, phi_3 = _as_field_triplet(Phi1, Phi2, Phi3)

    phi1_x, phi1_y, phi1_z = _gradient_xyz(phi_1)
    phi2_x, phi2_y, phi2_z = _gradient_xyz(phi_2)
    phi3_x, phi3_y, phi3_z = _gradient_xyz(phi_3)

    jacobian = (
        phi1_x * (phi2_y * phi3_z - phi2_z * phi3_y)
        - phi2_x * (phi1_y * phi3_z - phi1_z * phi3_y)
        + phi3_x * (phi1_y * phi2_z - phi1_z * phi2_y)
    )
    curl_1 = phi3_y - phi2_z
    curl_2 = phi1_z - phi3_x
    curl_3 = phi2_x - phi1_y
    return jacobian, curl_1, curl_2, curl_3


def compute_JD3D(phi_1: Any, phi_2: Any, phi_3: Any) -> torch.Tensor:
    """Compute the voxel-space Jacobian determinant of a 3-D deformation.

    The three inputs are the x, y, and z coordinate components and must have
    identical three-dimensional shapes.  Array-like inputs become CPU tensors;
    existing tensors retain their device.  Derivatives match MATLAB
    ``gradient(..., 1)`` at both interior and boundary samples.
    """
    phi_1, phi_2, phi_3 = _as_field_triplet(
        phi_1, phi_2, phi_3, promote=False
    )
    phi1_x, phi1_y, phi1_z = _gradient_xyz(phi_1)
    phi2_x, phi2_y, phi2_z = _gradient_xyz(phi_2)
    phi3_x, phi3_y, phi3_z = _gradient_xyz(phi_3)
    return (
        phi1_x * (phi2_y * phi3_z - phi2_z * phi3_y)
        - phi2_x * (phi1_y * phi3_z - phi1_z * phi3_y)
        + phi3_x * (phi1_y * phi2_z - phi1_z * phi2_y)
    )


def matrixpad3D(A: Any, x: Any) -> torch.Tensor:
    """Return a 3-D array surrounded by a one-voxel constant boundary."""
    array = A if isinstance(A, torch.Tensor) else torch.as_tensor(A)
    if array.ndim != 3:
        raise ValueError(
            f"A must be three-dimensional; got shape {tuple(array.shape)}."
        )

    boundary = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if boundary.numel() != 1:
        raise ValueError(
            "x must contain exactly one boundary value; "
            f"got shape {tuple(boundary.shape)}."
        )
    if boundary.is_complex() and not array.is_complex():
        raise TypeError("A real-valued array cannot use a complex boundary value.")

    boundary = boundary.to(device=array.device, dtype=array.dtype).reshape(())
    output_shape = tuple(size + 2 for size in array.shape)
    padded = boundary.expand(output_shape).clone()
    padded[1:-1, 1:-1, 1:-1] = array
    return padded


def _as_tensor_pair(A: Any, B: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an image pair without moving existing tensors between devices."""
    a_is_tensor = isinstance(A, torch.Tensor)
    b_is_tensor = isinstance(B, torch.Tensor)

    if a_is_tensor and b_is_tensor:
        if A.device != B.device:
            raise ValueError(
                f"A and B must be on the same device; got {A.device} and {B.device}."
            )
        a, b = A, B
    elif a_is_tensor:
        a = A
        b = torch.as_tensor(B, device=a.device)
    elif b_is_tensor:
        b = B
        a = torch.as_tensor(A, device=b.device)
    else:
        a = torch.as_tensor(A)
        b = torch.as_tensor(B)

    if a.shape != b.shape:
        raise ValueError(
            f"A and B must have the same shape; got {tuple(a.shape)} and {tuple(b.shape)}."
        )
    return a, b


def _im2uint8(image: torch.Tensor) -> torch.Tensor:
    """Reproduce MATLAB ``im2uint8`` for supported PyTorch numeric dtypes."""
    if image.dtype == torch.uint8:
        return image
    if image.dtype == torch.bool:
        return image.to(torch.uint8) * 255
    if image.is_floating_point():
        values = image.to(torch.float64)
        values = torch.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        values = values.clamp_(0.0, 1.0)
        return torch.floor(values * 255.0 + 0.5).to(torch.uint8)
    if image.is_complex():
        raise TypeError("Complex-valued images are not supported.")

    try:
        limits = torch.iinfo(image.dtype)
    except TypeError as error:
        raise TypeError(f"Unsupported image dtype: {image.dtype}.") from error
    if limits.min == 0 and limits.max == 255:
        return image.to(torch.uint8)

    values = image.to(torch.float64)
    scale = 255.0 / float(limits.max - limits.min)
    values = (values - float(limits.min)) * scale
    return torch.floor(values + 0.5).clamp_(0.0, 255.0).to(torch.uint8)


def _bin_indices(
    values: torch.Tensor,
    levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values_float = values.to(torch.float64)
    upper = float(levels - 1)
    valid = (
        torch.isfinite(values_float)
        & (values_float >= 0.0)
        & (values_float <= upper)
    )
    indices = torch.floor(values_float * (float(levels) / upper)).to(torch.int64)
    return indices.clamp_(0, levels - 1), valid


def mutualInfo(A: Any, B: Any) -> torch.Tensor:
    """Return MATLAB-compatible discrete mutual information in bits."""
    a, b = _as_tensor_pair(A, B)
    sample_count = a.numel()

    if a.dtype == torch.bool:
        levels = 2
    else:
        a = _im2uint8(a)
        b = _im2uint8(b)
        levels = 256

    if sample_count == 0:
        return torch.zeros((), dtype=torch.float64, device=a.device)

    a_indices, a_valid = _bin_indices(a.reshape(-1), levels)
    b_indices, b_valid = _bin_indices(b.reshape(-1), levels)
    pa_counts = torch.bincount(a_indices[a_valid], minlength=levels)
    pb_counts = torch.bincount(b_indices[b_valid], minlength=levels)

    pair_valid = a_valid & b_valid
    joint_indices = a_indices[pair_valid] * levels + b_indices[pair_valid]
    pab_counts = torch.bincount(
        joint_indices, minlength=levels * levels
    ).reshape(levels, levels)

    denominator = float(sample_count)
    pa = pa_counts.to(torch.float64) / denominator
    pb = pb_counts.to(torch.float64) / denominator
    pab = pab_counts.to(torch.float64) / denominator
    papb = torch.outer(pa, pb)
    valid = (papb > 1.0e-12) & (pab > 1.0e-12)
    return torch.sum(pab[valid] * torch.log2(pab[valid] / papb[valid]))


def _matlab_dimension(dim: Any) -> int:
    """Convert a MATLAB-style dimension number (1, 2, or 3) to a tensor axis."""
    if isinstance(dim, bool):
        raise TypeError("dim must be one of the integers 1, 2, or 3.")
    try:
        integer = int(dim)
        numeric = float(dim)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("dim must be one of the integers 1, 2, or 3.") from error
    if not math.isfinite(numeric) or numeric != integer or integer not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3; got {dim!r}.")
    return integer - 1


def gpu_dst_exact(data: Any, dim: Any) -> torch.Tensor:
    """Apply the unnormalized DST-I used by ``VPreg_gpu.m``.

    ``dim`` deliberately keeps MATLAB's 1-based values ``1``, ``2``, and
    ``3``.  Despite its historical name, the function runs on either CPU or
    CUDA and never changes the input tensor's device.
    """
    array = _as_floating_tensor(data, "data")
    if array.ndim != 4:
        raise ValueError(f"data must be four-dimensional; got {tuple(array.shape)}.")
    axis = _matlab_dimension(dim)

    singleton_shape = list(array.shape)
    singleton_shape[axis] = 1
    zero = array.new_zeros(singleton_shape)
    padded = torch.cat((zero, array, zero, -torch.flip(array, dims=(axis,))), dim=axis)
    transformed = torch.fft.fft(padded, dim=axis)

    selection = [slice(None)] * 4
    selection[axis] = slice(1, array.shape[axis] + 1)
    return -transformed.imag[tuple(selection)]


def gpuPoissonSolver4D(F_4D: Any, G: Any) -> torch.Tensor:
    """Solve VPreg's three-component Dirichlet Poisson system with DST-I.

    The intentionally retained final normalization is exactly the expression
    in ``VPreg_gpu.m``.  It is eight times the conventional inverse DST-I
    normalization, but changing it here would invalidate the parameters tuned
    around the original MATLAB behavior.
    """
    forcing = _as_floating_tensor(F_4D, "F_4D")
    if forcing.ndim != 4 or forcing.shape[-1] != 3:
        raise ValueError(
            "F_4D must have shape (Nx, Ny, Nz, 3); "
            f"got {tuple(forcing.shape)}."
        )

    if isinstance(G, torch.Tensor) and G.device != forcing.device:
        raise ValueError(
            f"G must be on {forcing.device}, not {G.device}."
        )
    denominator = G if isinstance(G, torch.Tensor) else torch.as_tensor(G, device=forcing.device)
    denominator = _as_floating_tensor(denominator, "G").to(dtype=forcing.dtype)
    if denominator.shape != forcing.shape[:3]:
        raise ValueError(
            f"G must have shape {tuple(forcing.shape[:3])}; "
            f"got {tuple(denominator.shape)}."
        )

    solution = gpu_dst_exact(forcing, 1)
    solution = gpu_dst_exact(solution, 2)
    solution = gpu_dst_exact(solution, 3)
    solution = solution / denominator.unsqueeze(-1)
    solution = gpu_dst_exact(solution, 1)
    solution = gpu_dst_exact(solution, 2)
    solution = gpu_dst_exact(solution, 3)

    nx, ny, nz, _ = forcing.shape
    scale = (2.0 / (nx + 1)) * (2.0 / (ny + 1)) * (2.0 / (nz + 1))
    return solution * (scale * 0.125)


_REPORT_HEADER = (
    "time_fwd Dice_fwd SSD_ratio_fwd MI_incr_%_fwd JDmin_fwd "
    "JDmax_fwd time_inv Dice_inv SSD_ratio_inv MI_incr_%_inv "
    "JDmin_inv JDmax_inv  \n"
)


def _as_common_device_tensors(**values: Any) -> dict[str, torch.Tensor]:
    tensor_values = [value for value in values.values() if isinstance(value, torch.Tensor)]
    devices = {value.device for value in tensor_values}
    if len(devices) > 1:
        device_names = ", ".join(str(device) for device in sorted(devices, key=str))
        raise ValueError(f"All evaluation inputs must use one device; got {device_names}.")
    device = tensor_values[0].device if tensor_values else None
    return {
        name: (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=device)
        )
        for name, value in values.items()
    }


def _validate_image(name: str, image: torch.Tensor) -> None:
    if image.ndim != 3:
        raise ValueError(f"{name} must be three-dimensional; got {tuple(image.shape)}.")
    if not image.is_floating_point():
        raise TypeError(f"{name} must be real floating point; got {image.dtype}.")


def _validate_deformation(name: str, deformation: torch.Tensor) -> None:
    if deformation.ndim != 4 or deformation.shape[-1] != 3:
        raise ValueError(
            f"{name} must have shape (nx, ny, nz, 3); got {tuple(deformation.shape)}."
        )
    if not deformation.is_floating_point():
        raise TypeError(f"{name} must be real floating point; got {deformation.dtype}.")
    if any(size < 2 for size in deformation.shape[:3]):
        raise ValueError(
            f"Every spatial dimension of {name} must have at least two samples; "
            f"got {tuple(deformation.shape[:3])}."
        )


def _cubic_kernel(distance: torch.Tensor) -> torch.Tensor:
    """Keys cubic-convolution kernel used by MATLAB ``imresize3``."""
    absolute = distance.abs()
    squared = absolute.square()
    cubed = squared * absolute
    inner = 1.5 * cubed - 2.5 * squared + 1.0
    outer = -0.5 * cubed + 2.5 * squared - 4.0 * absolute + 2.0
    return torch.where(
        absolute <= 1.0,
        inner,
        torch.where(absolute <= 2.0, outer, torch.zeros_like(distance)),
    )


def _resize_contributions(
    input_length: int,
    output_length: int,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_width = 4.0
    antialias = scale < 1.0
    if antialias:
        kernel_width /= scale

    output_coordinates = torch.arange(
        1, output_length + 1, dtype=torch.float64, device=device
    )
    input_coordinates = (
        output_coordinates / scale + 0.5 * (1.0 - 1.0 / scale)
    )
    left = torch.floor(input_coordinates - kernel_width / 2.0)
    contribution_count = math.ceil(kernel_width) + 2
    offsets = torch.arange(contribution_count, dtype=torch.float64, device=device)
    raw_indices = left[:, None] + offsets[None, :]
    distances = input_coordinates[:, None] - raw_indices

    if antialias:
        weights = scale * _cubic_kernel(scale * distances)
    else:
        weights = _cubic_kernel(distances)
    weights = weights / weights.sum(dim=1, keepdim=True)

    period = 2 * input_length
    wrapped = torch.remainder(raw_indices.to(torch.int64) - 1, period)
    indices = torch.where(
        wrapped < input_length,
        wrapped,
        period - 1 - wrapped,
    )
    nonzero_columns = torch.any(weights != 0.0, dim=0)
    return weights[:, nonzero_columns], indices[:, nonzero_columns]


def _resize_along_dimension(
    volume: torch.Tensor,
    dimension: int,
    output_length: int,
    scale: float,
) -> torch.Tensor:
    weights, indices = _resize_contributions(
        volume.shape[dimension], output_length, scale, volume.device
    )
    moved = volume.movedim(dimension, 0)
    samples = moved.index_select(0, indices.reshape(-1))
    samples = samples.reshape(output_length, indices.shape[1], *moved.shape[1:])
    weight_shape = (output_length, indices.shape[1]) + (1,) * (volume.ndim - 1)
    resized = torch.sum(
        samples * weights.to(dtype=volume.dtype).reshape(weight_shape), dim=1
    )
    return resized.movedim(0, dimension)


def _imresize3_cubic(
    volume: torch.Tensor,
    output_size: tuple[int, int, int],
) -> torch.Tensor:
    """GPU-capable equivalent of default ``imresize3(volume, output_size)``."""
    if volume.ndim != 3:
        raise ValueError(f"volume must be three-dimensional; got {tuple(volume.shape)}.")
    if not volume.is_floating_point():
        raise TypeError(f"volume must be floating point; got {volume.dtype}.")
    if len(output_size) != 3 or any(size <= 0 for size in output_size):
        raise ValueError(
            f"output_size must contain three positive values; got {output_size}."
        )
    if any(size == 0 for size in volume.shape):
        raise ValueError("volume dimensions must be nonempty.")

    output_size = tuple(int(size) for size in output_size)
    if tuple(volume.shape) == output_size:
        return volume
    scales = tuple(
        output_length / input_length
        for input_length, output_length in zip(volume.shape, output_size)
    )
    dimension_order = sorted(range(3), key=scales.__getitem__)
    resized = volume
    for dimension in dimension_order:
        if resized.shape[dimension] != output_size[dimension]:
            resized = _resize_along_dimension(
                resized, dimension, output_size[dimension], scales[dimension]
            )
    return resized


def _graythresh(image: torch.Tensor) -> torch.Tensor:
    """Reproduce MATLAB ``graythresh`` for a floating-point volume."""
    if image.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=image.device)

    quantized = _im2uint8(image).reshape(-1).to(torch.int64)
    counts = torch.bincount(quantized, minlength=256).to(torch.float64)
    probabilities = counts / counts.sum()
    omega = torch.cumsum(probabilities, dim=0)
    bins = torch.arange(1, 257, dtype=torch.float64, device=image.device)
    mu = torch.cumsum(probabilities * bins, dim=0)
    mu_total = mu[-1]
    between_class_variance = (mu_total * omega - mu).square() / (
        omega * (1.0 - omega)
    )

    finite = torch.isfinite(between_class_variance)
    finite_values = torch.where(
        finite,
        between_class_variance,
        torch.full_like(between_class_variance, -torch.inf),
    )
    maximum = finite_values.max()
    maxima = finite & (between_class_variance == maximum)
    zero_based_bins = torch.arange(256, dtype=torch.float64, device=image.device)
    mean_index = (maxima.to(torch.float64) * zero_based_bins).sum() / maxima.sum()
    threshold = mean_index / 255.0
    return torch.where(
        torch.isfinite(maximum), threshold, torch.zeros_like(threshold)
    )


def _imbinarize(image: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Apply MATLAB's strict bright-foreground threshold comparison."""
    if image.is_floating_point():
        cutoff = threshold
    else:
        limits = torch.iinfo(image.dtype)
        cutoff = float(limits.min) + float(limits.max - limits.min) * threshold
    return image > cutoff


def _dice(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Sørensen-Dice coefficient for two logical volumes."""
    if first.shape != second.shape:
        raise ValueError(
            f"Dice masks must have the same shape; got {tuple(first.shape)} and "
            f"{tuple(second.shape)}."
        )
    intersection = torch.count_nonzero(first & second).to(torch.float64)
    denominator = (
        torch.count_nonzero(first).to(torch.float64)
        + torch.count_nonzero(second).to(torch.float64)
    )
    return 2.0 * intersection / denominator


def _as_scalar_float(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar; got shape {tuple(value.shape)}.")
        value = value.detach().item()
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar.") from error


def _format_matlab_scalar(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == 0.0:
        return "0"
    return format(value, ".5g")


def evaluations3D(
    I_m: Any,
    I_f: Any,
    I_m_phi: Any,
    I_f_phi_inv: Any,
    phi: Any,
    phi_inv: Any,
    name_is: str | Path,
    t_phi: Any,
    t_phi_inv: Any,
) -> None:
    """Compute registration metrics and write ``evaluate_tabular.txt``."""
    tensors = _as_common_device_tensors(
        I_m=I_m,
        I_f=I_f,
        I_m_phi=I_m_phi,
        I_f_phi_inv=I_f_phi_inv,
        phi=phi,
        phi_inv=phi_inv,
    )
    I_m = tensors["I_m"]
    I_f = tensors["I_f"]
    I_m_phi = tensors["I_m_phi"]
    I_f_phi_inv = tensors["I_f_phi_inv"]
    phi = tensors["phi"]
    phi_inv = tensors["phi_inv"]

    for image_name, image in (
        ("I_m", I_m),
        ("I_f", I_f),
        ("I_m_phi", I_m_phi),
        ("I_f_phi_inv", I_f_phi_inv),
    ):
        _validate_image(image_name, image)
    _validate_deformation("phi", phi)
    _validate_deformation("phi_inv", phi_inv)

    if I_m_phi.shape != I_f.shape:
        raise ValueError(
            f"I_m_phi must match I_f; got {tuple(I_m_phi.shape)} and "
            f"{tuple(I_f.shape)}."
        )
    if I_f_phi_inv.shape != I_m.shape:
        raise ValueError(
            f"I_f_phi_inv must match I_m; got {tuple(I_f_phi_inv.shape)} and "
            f"{tuple(I_m.shape)}."
        )
    if phi.shape[:3] != I_f.shape:
        raise ValueError(
            f"phi spatial shape must match I_f; got {tuple(phi.shape[:3])} and "
            f"{tuple(I_f.shape)}."
        )
    if phi_inv.shape[:3] != I_m.shape:
        raise ValueError(
            "phi_inv spatial shape must match I_m; got "
            f"{tuple(phi_inv.shape[:3])} and {tuple(I_m.shape)}."
        )

    with torch.no_grad():
        mi_fwd = mutualInfo(I_m_phi, I_f)
        I_m_on_I_f = _imresize3_cubic(I_m, tuple(I_f.shape))
        mi_on_I_f = mutualInfo(I_f, I_m_on_I_f)
        mi_fwd_r = (mi_fwd - mi_on_I_f) / mi_on_I_f

        mi_bwd = mutualInfo(I_m, I_f_phi_inv)
        I_f_on_I_m = _imresize3_cubic(I_f, tuple(I_m.shape))
        mi_on_I_m = mutualInfo(I_f_on_I_m, I_m)
        mi_bwd_r = (mi_bwd - mi_on_I_m) / mi_on_I_m

        BW_I_m_phi = _imbinarize(I_m_phi, _graythresh(I_m_phi))
        BW_I_f = _imbinarize(I_f, _graythresh(I_f))
        dice_fwd = _dice(BW_I_m_phi, BW_I_f)

        ssd_original_on_I_f = torch.sum((I_m_on_I_f - I_f).square())
        ssd_ratio_fwd = torch.sum((I_m_phi - I_f).square()) / ssd_original_on_I_f
        ssd_original_on_I_m = torch.sum((I_f_on_I_m - I_m).square())
        ssd_ratio_inv = (
            torch.sum((I_f_phi_inv - I_m).square()) / ssd_original_on_I_m
        )

        jacobian_fwd = compute_JD3D(phi[..., 0], phi[..., 1], phi[..., 2])
        jdmax_fwd = jacobian_fwd.max()
        jdmin_fwd = jacobian_fwd.min()

        BW_I_f_phi_inv = _imbinarize(I_f_phi_inv, _graythresh(I_f_phi_inv))
        BW_I_m = _imbinarize(I_m, _graythresh(I_m))
        dice_inv = _dice(BW_I_f_phi_inv, BW_I_m)

        jacobian_inv = compute_JD3D(
            phi_inv[..., 0], phi_inv[..., 1], phi_inv[..., 2]
        )
        jdmax_inv = jacobian_inv.max()
        jdmin_inv = jacobian_inv.min()

    values = [
        _as_scalar_float(t_phi, "t_phi"),
        _as_scalar_float(dice_fwd, "dice_fwd"),
        _as_scalar_float(ssd_ratio_fwd, "ssd_ratio_fwd"),
        _as_scalar_float(mi_fwd_r * 100.0, "mi_incr_fwd"),
        _as_scalar_float(jdmin_fwd, "jdmin_fwd"),
        _as_scalar_float(jdmax_fwd, "jdmax_fwd"),
        _as_scalar_float(t_phi_inv, "t_phi_inv"),
        _as_scalar_float(dice_inv, "dice_inv"),
        _as_scalar_float(ssd_ratio_inv, "ssd_ratio_inv"),
        _as_scalar_float(mi_bwd_r * 100.0, "mi_incr_inv"),
        _as_scalar_float(jdmin_inv, "jdmin_inv"),
        _as_scalar_float(jdmax_inv, "jdmax_inv"),
    ]
    formatted = [_format_matlab_scalar(value) for value in values]
    data_line = (
        f"{formatted[0]}  {formatted[1]}  {formatted[2]}  {formatted[3]}   "
        f"{formatted[4]}  {formatted[5]} {formatted[6]} {formatted[7]}  "
        f"{formatted[8]}  {formatted[9]}  {formatted[10]}  {formatted[11]} \n"
    )

    filename = Path(f"{name_is}evaluate_tabular.txt")
    with filename.open("w", encoding="utf-8") as report:
        report.write(_REPORT_HEADER)
        report.write(data_line)
    print(f"Evaluation written to {filename}")
