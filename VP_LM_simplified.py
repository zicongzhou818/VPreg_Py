"""Simplified GPU inverse solver translated from MATLAB ``VP_LM_gpu.m``."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


__all__ = ["VP_LM_gpu"]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be an integer.") from error
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be an integer.") from error
    if not math.isfinite(numeric) or numeric != integer:
        raise ValueError(f"{name} must be a finite integer; got {value!r}.")
    if integer < 3:
        raise ValueError(f"{name} must be at least 3; got {integer}.")
    return integer


def _as_deformation_tensors(
    expected_shape: tuple[int, int, int],
    **values: Any,
) -> dict[str, torch.Tensor]:
    """Convert inputs to a shared floating dtype and existing tensor device."""
    tensor_values = [value for value in values.values() if isinstance(value, torch.Tensor)]
    devices = {value.device for value in tensor_values}
    if len(devices) > 1:
        device_names = ", ".join(str(device) for device in sorted(devices, key=str))
        raise ValueError(f"All deformation inputs must use one device; got {device_names}.")
    device = tensor_values[0].device if tensor_values else None

    tensors: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        tensor = (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=device)
        )
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}; got {tuple(tensor.shape)}."
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be real floating point; got {tensor.dtype}.")
        tensors[name] = tensor

    dtype = next(iter(tensors.values())).dtype
    for tensor in tensors.values():
        dtype = torch.promote_types(dtype, tensor.dtype)
    return {
        name: tensor.to(device=device, dtype=dtype)
        for name, tensor in tensors.items()
    }


def _normalize_coordinate(coordinate: torch.Tensor, length: int) -> torch.Tensor:
    """Map MATLAB coordinates 1..length to grid_sample coordinates -1..1."""
    return 2.0 * (coordinate - 1.0) / float(length - 1) - 1.0


def _interpn_linear_components(
    values: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    query_z: torch.Tensor,
) -> torch.Tensor:
    """Linearly sample three fields using MATLAB's 1-based coordinates.

    ``values`` has shape ``(3, Nx, Ny, Nz)``.  The returned tensor has the
    same shape.  Queries outside the source grid become NaN, matching MATLAB
    ``interpn`` when no extrapolation value is supplied.
    """
    if values.ndim != 4 or values.shape[0] != 3:
        raise ValueError(
            "values must have shape (3, Nx, Ny, Nz); "
            f"got {tuple(values.shape)}."
        )
    spatial_shape = tuple(values.shape[1:])
    for name, query in (
        ("query_x", query_x),
        ("query_y", query_y),
        ("query_z", query_z),
    ):
        if query.shape != spatial_shape:
            raise ValueError(
                f"{name} must have shape {spatial_shape}; got {tuple(query.shape)}."
            )
        if query.device != values.device or query.dtype != values.dtype:
            raise ValueError(f"{name} must match values' dtype and device.")

    nx, ny, nz = spatial_shape
    # PyTorch treats a 5-D input as (N, C, D, H, W), with grid coordinates
    # ordered as (W, H, D).  VPreg arrays are ordered (x, y, z), hence z/y/x.
    grid = torch.stack(
        (
            _normalize_coordinate(query_z, nz),
            _normalize_coordinate(query_y, ny),
            _normalize_coordinate(query_x, nx),
        ),
        dim=-1,
    ).unsqueeze(0)
    sampled = F.grid_sample(
        values.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]

    in_bounds = (
        (query_x >= 1.0)
        & (query_x <= float(nx))
        & (query_y >= 1.0)
        & (query_y <= float(ny))
        & (query_z >= 1.0)
        & (query_z <= float(nz))
    )
    return sampled.masked_fill(~in_bounds.unsqueeze(0), torch.nan)


def _format_scalar(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == 0.0:
        return "0"
    return format(value, ".5g")


def VP_LM_gpu(
    Nx: Any,
    Ny: Any,
    Nz: Any,
    given_x: Any,
    given_y: Any,
    given_z: Any,
    target_x: Any,
    target_y: Any,
    target_z: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Find an inverse coordinate map for a given 3D deformation.

    The six outputs retain the MATLAB order: three composed fields followed by
    the three inverse-map coordinate fields.  Existing CUDA tensors remain on
    the GPU, and outputs stay there for use by ``VPreg_gpu``.  Array-like inputs
    become CPU tensors.

    This solver performs explicit accept/reject iterations and is intentionally
    not differentiated by autograd.
    """
    nx = _positive_integer(Nx, "Nx")
    ny = _positive_integer(Ny, "Ny")
    nz = _positive_integer(Nz, "Nz")
    shape = (nx, ny, nz)
    tensors = _as_deformation_tensors(
        shape,
        given_x=given_x,
        given_y=given_y,
        given_z=given_z,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
    )
    given_x = tensors["given_x"]
    given_y = tensors["given_y"]
    given_z = tensors["given_z"]
    target_x = tensors["target_x"]
    target_y = tensors["target_y"]
    target_z = tensors["target_z"]
    device = given_x.device
    dtype = given_x.dtype

    better = True
    accepted_iterations = 0
    total_iterations = 0
    maximum_iterations = 65
    step_size = 1.0e-1
    minimum_step_size = 5.0e-4
    step_increase = 1.1
    step_decrease = 0.9
    ratio = 1.0
    ratio_tolerance = 1.0e-9

    axes = (
        torch.arange(1, nx + 1, dtype=dtype, device=device),
        torch.arange(1, ny + 1, dtype=dtype, device=device),
        torch.arange(1, nz + 1, dtype=dtype, device=device),
    )
    identity_x, identity_y, identity_z = torch.meshgrid(*axes, indexing="ij")

    with torch.no_grad():
        phi_m1 = identity_x.clone()
        phi_m2 = identity_y.clone()
        phi_m3 = identity_z.clone()

        original = torch.stack((given_x, given_y, given_z), dim=0)
        phi1_comp = given_x.clone()
        phi2_comp = given_y.clone()
        phi3_comp = given_z.clone()

        ssd_initial = torch.sum(
            (phi1_comp - target_x).square()
            + (phi2_comp - target_y).square()
            + (phi3_comp - target_z).square()
        )
        ssd_old = ssd_initial
        ssd_new = ssd_initial

        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        while (
            total_iterations < maximum_iterations
            and step_size > minimum_step_size
            and ratio > ratio_tolerance
        ):
            total_iterations += 1
            if better:
                accepted_iterations += 1
                gradient_x = phi1_comp - target_x
                gradient_y = phi2_comp - target_y
                gradient_z = phi3_comp - target_z

            phi_m1n = phi_m1.clone()
            phi_m2n = phi_m2.clone()
            phi_m3n = phi_m3.clone()
            phi_m1n[interior] = (
                phi_m1[interior] - step_size * gradient_x[interior]
            )
            phi_m2n[interior] = (
                phi_m2[interior] - step_size * gradient_y[interior]
            )
            phi_m3n[interior] = (
                phi_m3[interior] - step_size * gradient_z[interior]
            )

            composed_new = _interpn_linear_components(
                original,
                phi_m1n,
                phi_m2n,
                phi_m3n,
            )
            phi1_compn, phi2_compn, phi3_compn = composed_new.unbind(0)
            ssd_new = torch.sum(
                (phi1_compn - target_x).square()
                + (phi2_compn - target_y).square()
                + (phi3_compn - target_z).square()
            )
            ratio = (ssd_new / ssd_initial).item()

            if (ssd_new < ssd_old).item():
                step_size *= step_increase
                phi_m1 = phi_m1n
                phi_m2 = phi_m2n
                phi_m3 = phi_m3n
                phi1_comp = phi1_compn
                phi2_comp = phi2_compn
                phi3_comp = phi3_compn
                ssd_old = ssd_new
                better = True
            else:
                better = False
                step_size *= step_decrease

    print(
        "inverse found "
        f"ts:{_format_scalar(step_size)} "
        f"r:{_format_scalar(ratio)} "
        f"ei/ti:{accepted_iterations}/{total_iterations}"
    )
    return phi1_comp, phi2_comp, phi3_comp, phi_m1, phi_m2, phi_m3
