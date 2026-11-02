"""GPU-first PyTorch translation of MATLAB ``VPreg_gpu.m``."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from utilities import (
    _imresize3_cubic,
    compute_JD3D,
    compute_JD_and_Curl3D,
    gpuPoissonSolver4D,
    makima3d,
    matrixpad3D,
    mutualInfo,
    voxel2zscore,
)
from VP_LM_simplified import VP_LM_gpu


__all__ = ["VPreg_gpu"]


def _array_payload(value: Any) -> Any:
    """Extract an array from ANTsPyX-style image objects when necessary."""
    if isinstance(value, torch.Tensor):
        return value
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return numpy_method()
    return value


def _as_image_pair(I_m: Any, I_f: Any) -> tuple[torch.Tensor, torch.Tensor]:
    values = (_array_payload(I_m), _array_payload(I_f))
    existing = [value for value in values if isinstance(value, torch.Tensor)]
    devices = {value.device for value in existing}
    if len(devices) > 1:
        names = ", ".join(str(device) for device in sorted(devices, key=str))
        raise ValueError(f"I_m and I_f must use one device; got {names}.")

    if existing:
        device = existing[0].device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensors = tuple(
        (
            value.detach().to(device=device, dtype=torch.float32)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=device, dtype=torch.float32)
        )
        for value in values
    )
    for name, image in zip(("I_m", "I_f"), tensors):
        if image.ndim != 3:
            raise ValueError(f"{name} must be three-dimensional; got {tuple(image.shape)}.")
        if any(size < 3 for size in image.shape):
            raise ValueError(
                f"Every dimension of {name} must contain at least three voxels; "
                f"got {tuple(image.shape)}."
            )
    if tensors[0].shape != tensors[1].shape:
        raise ValueError(
            "VPreg_gpu requires moving and fixed images on the same prealigned grid; "
            f"got {tuple(tensors[0].shape)} and {tuple(tensors[1].shape)}."
        )
    return tensors


def _identity_grid(
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axes = [
        torch.arange(1, size + 1, dtype=dtype, device=device) for size in shape
    ]
    return torch.meshgrid(*axes, indexing="ij")


def _normalized_coordinate(coordinate: torch.Tensor, length: int) -> torch.Tensor:
    return 2.0 * (coordinate - 1.0) / float(length - 1) - 1.0


def _interpn_linear_channels(
    values: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    query_z: torch.Tensor,
) -> torch.Tensor:
    """Sample ``(C, Nx, Ny, Nz)`` fields at MATLAB 1-based coordinates."""
    if values.ndim != 4:
        raise ValueError(
            f"values must have shape (channels, Nx, Ny, Nz); got {tuple(values.shape)}."
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
    # grid_sample's float32 coordinate normalization can perturb even an exact
    # integer identity query by a few ULPs.  MATLAB interpn returns the source
    # samples exactly in that case, so bypass the sampler before constructing
    # its normalized grid.
    identity_x = torch.arange(1, nx + 1, dtype=values.dtype, device=values.device)[
        :, None, None
    ].expand(spatial_shape)
    identity_y = torch.arange(1, ny + 1, dtype=values.dtype, device=values.device)[
        None, :, None
    ].expand(spatial_shape)
    identity_z = torch.arange(1, nz + 1, dtype=values.dtype, device=values.device)[
        None, None, :
    ].expand(spatial_shape)
    if (
        torch.equal(query_x, identity_x)
        and torch.equal(query_y, identity_y)
        and torch.equal(query_z, identity_z)
    ):
        return values

    # grid_sample uses (N, C, D, H, W), while its final grid coordinate is
    # ordered (W, H, D).  VPreg arrays are (x, y, z), hence z/y/x below.
    grid = torch.stack(
        (
            _normalized_coordinate(query_z, nz),
            _normalized_coordinate(query_y, ny),
            _normalized_coordinate(query_x, nx),
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


def _interpn_linear(
    values: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    query_z: torch.Tensor,
) -> torch.Tensor:
    return _interpn_linear_channels(
        values.unsqueeze(0), query_x, query_y, query_z
    )[0]


def _gradient_xyz(
    field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.gradient(
        field,
        spacing=1.0,
        dim=(0, 1, 2),
        edge_order=1,
    )


def _poisson_denominator(
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    terms = []
    for size in shape:
        indices = torch.arange(1, size + 1, dtype=torch.float64, device=device)
        term = 4.0 * torch.sin((math.pi / (2.0 * (size + 1))) * indices).square()
        terms.append(term)
    denominator = (
        terms[0][:, None, None]
        + terms[1][None, :, None]
        + terms[2][None, None, :]
    )
    return denominator.to(dtype=dtype)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Use MATLAB/IEEE division behavior without Python zero-division errors."""
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return math.nan
    return math.copysign(math.inf, numerator)


def _format_scalar(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == 0.0:
        return "0"
    return format(value, ".5g")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def VPreg_gpu(
    I_m: Any,
    I_f: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
]:
    """Register a moving 3-D image to a fixed image with VPreg.

    The return order matches MATLAB exactly: forward deformation, warped
    moving image, inverse deformation, inverse-warped fixed image, forward
    time, and inverse-solver time.  NumPy and ANTsPyX image inputs use CUDA
    automatically when it is available; tensor inputs retain their device.
    Returned tensors stay on that device so downstream PyTorch work does not
    require an unnecessary transfer.

    VPreg assumes rigid/affine preprocessing has already placed both images on
    the same voxel grid.  This solver is an explicit optimizer and is not
    differentiated by autograd.
    """
    forward_start = time.perf_counter()
    moving_original, fixed_original = _as_image_pair(I_m, I_f)
    device = moving_original.device
    dtype = moving_original.dtype
    shape = tuple(moving_original.shape)
    nx, ny, nz = shape

    with torch.no_grad():
        moving = voxel2zscore(moving_original)
        fixed = voxel2zscore(fixed_original)

        jacobian_threshold = 0.55
        regrid_bound = 8
        alternating = regrid_bound - 2

        stage = 0
        stagnate = 1
        sequential_initial_step = 3.3e-3
        field_initial_step = 3.3e-4
        step_size = sequential_initial_step
        minimum_step = 1.0e-6
        mode = "seq-mod"
        sequential_control = 0.1
        field_control = 0.01
        step_increase = 1.0 + sequential_control
        step_decrease = 1.0 - sequential_control
        maximum_evaluations = 150
        ratio_tolerance = 0.70
        local_ratio = 1.0

        f = torch.ones(shape, dtype=dtype, device=device)
        g = torch.zeros((*shape, 3), dtype=dtype, device=device)

        ssd_old = torch.sum((moving - fixed).square()).item()
        ssd_initial = ssd_old
        ssd_local_initial = ssd_old
        initial_mi = mutualInfo(moving, fixed).item()

        better = True
        true_better = False
        total_iterations = 0
        evaluation_iterations = 0

        identity_x, identity_y, identity_z = _identity_grid(
            shape, device=device, dtype=dtype
        )
        phi_x = identity_x
        phi_y = identity_y
        phi_z = identity_z
        phi_x_intermediate = identity_x
        phi_y_intermediate = identity_y
        phi_z_intermediate = identity_z

        moving_local = moving
        minimum_jacobian = 1.0
        all_iterations = 0
        denominator = _poisson_denominator(shape, device=device, dtype=dtype)

        lap_phi = torch.zeros((*shape, 3), dtype=dtype, device=device)
        lap_b = torch.zeros_like(lap_phi)
        delta_g = torch.zeros_like(lap_phi)
        fg = torch.zeros_like(lap_phi)
        penalty = 5.0
        regrid_iteration = 0
        global_ratio = 1.0
        mi_increase = 0.0

        while regrid_iteration <= regrid_bound:
            while (
                local_ratio > ratio_tolerance
                and step_size > minimum_step
                and stage < stagnate
                and evaluation_iterations < maximum_evaluations
                and minimum_jacobian >= jacobian_threshold
            ):
                total_iterations += 1
                all_iterations += 1

                if better:
                    evaluation_iterations += 1
                    if regrid_iteration < alternating:
                        fi, gi_1, gi_2, gi_3 = compute_JD_and_Curl3D(
                            phi_x, phi_y, phi_z
                        )
                        fi_x, fi_y, fi_z = _gradient_xyz(fi)
                        _, curl_gi_1, curl_gi_2, curl_gi_3 = (
                            compute_JD_and_Curl3D(gi_1, gi_2, gi_3)
                        )

                        moving_temp = _interpn_linear(
                            moving_local, phi_x, phi_y, phi_z
                        )
                        image_difference = moving_temp - fixed
                        moving_x, moving_y, moving_z = _gradient_xyz(moving_temp)

                        lap_phi[..., 0] = image_difference * moving_x + penalty * (
                            fi_x - curl_gi_1
                        )
                        lap_phi[..., 1] = image_difference * moving_y + penalty * (
                            fi_y - curl_gi_2
                        )
                        lap_phi[..., 2] = image_difference * moving_z + penalty * (
                            fi_z - curl_gi_3
                        )
                        update = gpuPoissonSolver4D(-lap_phi, denominator)
                    else:
                        moving_temp = _interpn_linear(
                            moving_local, phi_x, phi_y, phi_z
                        )
                        image_difference = moving_temp - fixed
                        moving_x, moving_y, moving_z = _gradient_xyz(moving_temp)

                        lap_b[..., 0] = image_difference * moving_x
                        lap_b[..., 1] = image_difference * moving_y
                        lap_b[..., 2] = image_difference * moving_z
                        b_field = gpuPoissonSolver4D(lap_b, denominator)

                        b1_x, b1_y, b1_z = _gradient_xyz(b_field[..., 0])
                        b2_x, b2_y, b2_z = _gradient_xyz(b_field[..., 1])
                        b3_x, b3_y, b3_z = _gradient_xyz(b_field[..., 2])

                        delta_f = -f * (b1_x + b2_y + b3_z)
                        delta_g[..., 0] = -b3_y + b2_z
                        delta_g[..., 1] = b3_x - b1_z
                        delta_g[..., 2] = -b2_x + b1_y

                if regrid_iteration < alternating:
                    phi_x = identity_x + step_size * matrixpad3D(
                        update[1:-1, 1:-1, 1:-1, 0], 0.0
                    )
                    phi_y = identity_y + step_size * matrixpad3D(
                        update[1:-1, 1:-1, 1:-1, 1], 0.0
                    )
                    phi_z = identity_z + step_size * matrixpad3D(
                        update[1:-1, 1:-1, 1:-1, 2], 0.0
                    )
                else:
                    f_new = f - delta_f * step_size
                    g_new = g - delta_g * step_size

                    f_x, f_y, f_z = _gradient_xyz(f_new)
                    g1_x, g1_y, g1_z = _gradient_xyz(g_new[..., 0])
                    g2_x, g2_y, g2_z = _gradient_xyz(g_new[..., 1])
                    g3_x, g3_y, g3_z = _gradient_xyz(g_new[..., 2])

                    fg[..., 0] = f_x - g3_y + g2_z
                    fg[..., 1] = f_y + g3_x - g1_z
                    fg[..., 2] = f_z - g2_x + g1_y
                    update = gpuPoissonSolver4D(fg, denominator)

                    phi_x = identity_x + matrixpad3D(
                        update[1:-1, 1:-1, 1:-1, 0], 0.0
                    )
                    phi_y = identity_y + matrixpad3D(
                        update[1:-1, 1:-1, 1:-1, 1], 0.0
                    )
                    phi_z = identity_z + matrixpad3D(
                        update[1:-1, 1:-1, 1:-1, 2], 0.0
                    )

                jacobian, _, _, _ = compute_JD_and_Curl3D(phi_x, phi_y, phi_z)
                minimum_jacobian = jacobian.min().item()
                moving_warped = _interpn_linear(
                    moving_local, phi_x, phi_y, phi_z
                )
                forward_mi = mutualInfo(moving_warped, fixed).item()
                mi_increase = _safe_ratio(forward_mi - initial_mi, initial_mi)
                ssd = torch.sum((moving_warped - fixed).square()).item()
                local_ratio = _safe_ratio(ssd, ssd_local_initial)
                global_ratio = _safe_ratio(ssd, ssd_initial)

                if ssd < ssd_old:
                    stage = 0
                    moving_better = moving_warped
                    step_size *= step_increase
                    ssd_old = ssd
                    better = True
                    if regrid_iteration >= alternating:
                        f = f_new
                        g = g_new
                        mode = "f&g-mod"
                    phi_x_better = phi_x
                    phi_y_better = phi_y
                    phi_z_better = phi_z
                    true_better = True
                else:
                    better = False
                    step_size *= step_decrease
                    stage += 1

            print(
                f"{mode} c#:{regrid_iteration} "
                f"e/t:{evaluation_iterations}/{total_iterations} "
                f"s:{_format_scalar(step_size)} "
                f"r/rr:{_format_scalar(global_ratio)}/{_format_scalar(local_ratio)} "
                f"Jmin:{_format_scalar(minimum_jacobian)} "
                f"rMI:{_format_scalar(mi_increase * 100.0)}"
            )

            minimum_jacobian = 1.0
            if true_better:
                moving_local = moving_better
                composed = _interpn_linear_channels(
                    torch.stack(
                        (
                            phi_x_intermediate,
                            phi_y_intermediate,
                            phi_z_intermediate,
                        ),
                        dim=0,
                    ),
                    phi_x_better,
                    phi_y_better,
                    phi_z_better,
                )
                (
                    phi_x_intermediate,
                    phi_y_intermediate,
                    phi_z_intermediate,
                ) = composed.unbind(0)

                moving_true = _interpn_linear(
                    moving,
                    phi_x_intermediate,
                    phi_y_intermediate,
                    phi_z_intermediate,
                )
                ssd_old = torch.sum((moving_true - fixed).square()).item()
                forward_ratio = _safe_ratio(ssd_old, ssd_initial)
                ssd_local_initial = ssd_old
                full_jacobian = compute_JD3D(
                    phi_x_intermediate,
                    phi_y_intermediate,
                    phi_z_intermediate,
                )
                full_jacobian_minimum = full_jacobian.min().item()
                true_better = False
                print(
                    f" r_fwd:{_format_scalar(forward_ratio)} "
                    f"J_min_comp:{_format_scalar(full_jacobian_minimum)}"
                )

            phi_x = identity_x
            phi_y = identity_y
            phi_z = identity_z
            regrid_iteration += 1
            stage = 0
            better = True
            local_ratio = 1.0
            evaluation_iterations = 0
            total_iterations = 0

            if regrid_iteration < alternating:
                step_size = sequential_initial_step
                step_increase = 1.0 + sequential_control
                step_decrease = 1.0 - sequential_control
            else:
                f = torch.ones(shape, dtype=dtype, device=device)
                g = torch.zeros((*shape, 3), dtype=dtype, device=device)
                step_size = field_initial_step
                step_increase = 1.0 + field_control
                step_decrease = 1.0 - field_control
                ratio_tolerance = 0.75
                jacobian_threshold = 0.688
                stagnate += 1
            maximum_evaluations -= 10

        phi = torch.stack(
            (phi_x_intermediate, phi_y_intermediate, phi_z_intermediate), dim=-1
        )
        moving_warped = makima3d(
            moving_original,
            phi_x_intermediate,
            phi_y_intermediate,
            phi_z_intermediate,
        )
        moving_warped = _imresize3_cubic(moving_warped, shape)

    _synchronize(device)
    forward_time = time.perf_counter() - forward_start
    print(f"toc_phi = {_format_scalar(forward_time)}")

    inverse_start = time.perf_counter()
    (
        _,
        _,
        _,
        phi_x_inverse,
        phi_y_inverse,
        phi_z_inverse,
    ) = VP_LM_gpu(
        nx,
        ny,
        nz,
        phi_x_intermediate,
        phi_y_intermediate,
        phi_z_intermediate,
        identity_x,
        identity_y,
        identity_z,
    )
    _synchronize(device)
    inverse_time = time.perf_counter() - inverse_start
    print(f"toc_phi_inv = {_format_scalar(inverse_time)}")

    with torch.no_grad():
        phi_inverse = torch.stack(
            (phi_x_inverse, phi_y_inverse, phi_z_inverse), dim=-1
        )
        fixed_inverse_warped = makima3d(
            fixed_original,
            phi_x_inverse,
            phi_y_inverse,
            phi_z_inverse,
        )

    return (
        phi,
        moving_warped,
        phi_inverse,
        fixed_inverse_warped,
        forward_time,
        inverse_time,
    )
