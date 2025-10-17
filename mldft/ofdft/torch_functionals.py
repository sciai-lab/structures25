"""Implementation of functionals in PyTorch.

The code is based on https://github.com/sail-sg/jax_xc and
https://gitlab.com/libxc/libxc/-/blob/master/src/gga_k_apbe.c
https://gitlab.com/libxc/libxc/-/blob/master/src/maple2c/gga_exc/gga_k_apbe.c
"""

import torch
from loguru import logger
from pyscf import dft, gto

from mldft.utils.coeffs_to_grid import coeffs_to_rho_and_derivatives
from mldft.utils.grids import compute_max_block_size, get_grid_blocks


@torch.jit.script
def cbrt(x: torch.Tensor) -> torch.Tensor:
    """Cube root helper function allowing negative values.

    Args:
        x: input tensor

    Returns:
        torch.Tensor: cube root of x
    """
    return torch.sign(x) * torch.pow(torch.abs(x), 1 / 3)


@torch.jit.script
def _unpolarized_gga_k_apbe(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor = torch.tensor(0.804),
    mu: torch.Tensor = torch.tensor(0.23889),
    zeta_threshold: torch.Tensor = torch.tensor(2.220446049250313e-16),
    dens_threshold: torch.Tensor = torch.tensor(1e-15),
) -> torch.Tensor:
    """Calculation for the kinetic energy density of the APBE functional.

    Args:
        rho: density tensor of shape (ngrid)
        sigma: gradient squared tensor of shape (ngrid)
        kappa: parameter kappa of the APBE functional
        mu: parameter mu of the APBE functional
        zeta_threshold: some threshold
        dens_threshold: threshold of the density, return 0 if rho < dens_threshold

    Returns:
        torch.Tensor: kinetic energy density of the APBE functional
    """
    # Some pre-calculations using the zeta_threshold
    device = rho.device
    kappa.to(device)
    mu.to(device)
    zeta_threshold.to(device)
    dens_threshold.to(device)
    if 1.0 <= zeta_threshold:
        t12 = -zeta_threshold + 1.0
    else:
        t12 = torch.tensor(0.0, device=device)
    t13 = 1.0 + t12
    t16 = cbrt(zeta_threshold) ** 2
    t18 = cbrt(t13)
    t19 = t18**2
    if t13 <= zeta_threshold:
        pre_factor = t16 * zeta_threshold
    else:
        pre_factor = t19 * t13

    # Real calculation
    pi = torch.tensor(torch.pi, device=device)
    pre_factor2 = (
        pre_factor * 2.0 * 3.0 / 20.0 * cbrt(torch.tensor(3, device=device)) ** 2 * pi ** (4 / 3)
    )
    mask = rho / 0.2e1 >= dens_threshold
    res = torch.zeros_like(rho)
    res[mask] = pre_factor2 * (
        cbrt(rho[mask]) ** 2
        * (
            1.0
            + kappa
            * (
                1.0
                - kappa
                / (
                    kappa
                    + mu
                    * cbrt(torch.tensor(6, device=device))
                    * sigma[mask]
                    * cbrt(torch.tensor(2, device=device)) ** 2
                    / pi ** (4 / 3)
                    / cbrt(rho[mask]) ** 2
                    / rho[mask] ** 2
                    / 24.0
                )
            )
        )
    )
    return res


@torch.jit.script
def _unpolarized_gga_c_pbe(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    beta: torch.Tensor = torch.tensor(0.06672455060314922),
    gamma: torch.Tensor = torch.tensor(0.031090690869654894),
    BB: torch.Tensor = torch.tensor(1.0),
    zeta_threshold: torch.Tensor = torch.tensor(2.220446049250313e-16),
    dens_threshold: torch.Tensor = torch.tensor(1e-12),
):
    """Calculation for the correlation energy density of the PBE functional."""
    result = torch.zeros_like(rho)
    mask = rho >= dens_threshold
    rho_masked = rho[mask]
    sigma_masked = sigma[mask]
    device = rho.device
    pi = torch.tensor(torch.pi, device=device)
    t1 = cbrt(torch.tensor(3, device=device))
    t3 = cbrt(1.0 / pi)
    t5 = cbrt(torch.tensor(4, device=device))
    t6 = t5**2
    t7 = cbrt(rho_masked)
    t10 = t1 * t3 * t6 / t7
    t13 = torch.sqrt(t10)
    t16 = t10**0.15e1
    t18 = t1**2
    t19 = t3**2
    t21 = t7**2
    t24 = t18 * t19 * t5 / t21
    t30 = torch.log(
        1.0
        + 0.16081979498692535067e2
        / (0.379785e1 * t13 + 0.8969 * t10 + 0.204775 * t16 + 0.123235 * t24)
    )
    t32 = 0.621814e-1 * (0.1e1 + 0.53425e-1 * t10) * t30
    t33 = 0.1e1 <= zeta_threshold
    t34 = cbrt(zeta_threshold)
    if t33:
        t36 = t34 * zeta_threshold
    else:
        t36 = torch.tensor(1.0, device=device)
    t39 = cbrt(torch.tensor(2.0, device=device))
    t54 = torch.log(
        0.1e1
        + 0.29608749977793437516e2
        / (0.51785e1 * t13 + 0.905775 * t10 + 0.1100325 * t16 + 0.1241775 * t24)
    )
    t57 = (
        0.19751673498613801407e-1
        * (0.2e1 * t36 - 0.2e1)
        / (0.2e1 * t39 - 0.2e1)
        * (0.1e1 + 0.278125e-1 * t10)
        * t54
    )
    t58 = t34**2
    t59 = torch.where(t33, t58, 1)
    t60 = t59**2
    t61 = t60 * t59
    t63 = rho_masked**2
    t76 = 0.1e1 / gamma
    t81 = torch.exp(-(-t32 + t57) * t76 / t61)
    t83 = 0.1e1 / (t81 - 0.1e1)
    t85 = sigma_masked**2
    t88 = t63**2
    t91 = t39**2
    t93 = t60**2
    t102 = (
        sigma_masked / t7 / t63 * t39 / t60 * t18 / t3 * t5 / 0.96e2
        + BB * beta * t76 * t83 * t85 / t21 / t88 * t91 / t93 * t1 / t19 * t6 / 0.3072e4
    )
    t112 = torch.log(0.1e1 + beta * t102 * t76 / (beta * t76 * t83 * t102 + 0.1e1))
    result[mask] = gamma * t61 * t112 - t32 + t57
    return result


@torch.jit.script
def _unpolarized_gga_x_pbe(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    kappa: torch.Tensor = torch.tensor(0.804),
    mu: torch.Tensor = torch.tensor(0.2195149727645171),
    zeta_threshold: torch.Tensor = torch.tensor(2.220446049250313e-16),
    dens_threshold: torch.Tensor = torch.tensor(1e-15),
):
    """Calculation for the exchange energy density of the PBE functional."""
    result = torch.zeros_like(rho)
    mask = rho / 2.0 > dens_threshold
    rho_masked = rho[mask]
    sigma_masked = sigma[mask]
    device = rho.device
    pi = torch.tensor(torch.pi, device=device)
    kappa.to(device)
    mu.to(device)
    zeta_threshold.to(device)
    dens_threshold.to(device)

    t3 = cbrt(torch.tensor(3, device=device))
    t4 = cbrt(pi)
    t7 = 0.1e1 <= zeta_threshold
    t8 = zeta_threshold - 0.1e1
    t10 = torch.where(t7, -t8, 0)
    t11 = torch.where(t7, t8, t10)
    t12 = 0.1e1 + t11
    t14 = cbrt(zeta_threshold)
    t16 = cbrt(t12)
    t18 = torch.where(t12 <= zeta_threshold, t14 * zeta_threshold, t16 * t12)
    t19 = cbrt(rho_masked)
    t21 = cbrt(torch.tensor(6.0, device=device))
    t23 = pi**2
    t24 = cbrt(t23)
    t25 = t24**2
    t28 = cbrt(torch.tensor(2, device=device))
    t29 = t28**2
    t31 = rho_masked**2
    t32 = t19**2
    t47 = (
        -0.3e1
        / 0.8e1
        * t3
        / t4
        * t18
        * t19
        * (
            0.1e1
            + kappa
            * (0.1e1 - kappa / (kappa + mu * t21 / t25 * sigma_masked * t29 / t32 / t31 / 0.24e2))
        )
    )
    result[mask] = 0.2e1 * t47
    return result


str_to_torch_functionals = {
    "PBE": [_unpolarized_gga_x_pbe, _unpolarized_gga_c_pbe],
    "APBE": [_unpolarized_gga_k_apbe],
    "GGA_K_APBE": [_unpolarized_gga_k_apbe],
    "GGA_C_PBE": [_unpolarized_gga_c_pbe],
    "GGA_X_PBE": [_unpolarized_gga_x_pbe],
}


def torch_functional(
    density_and_gradient: torch.Tensor, functional: str, get_gradient: bool = True, **kwargs
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    r"""Wrapper for the computation of the kinetic energy density of the functional.

    Before calling the actual computation, the input is converted to torch tensors and the
    squared gradient is calculated.

    Args:
        density_and_gradient: density and density gradient tensor of shape (d, ngrid)
        functional: functional to use
        get_gradient: whether to return the gradient of the kinetic energy density
        kwargs: additional arguments for the functional

    Returns:
        tuple of the energy density of the functional divided by the density, gradient of the
        kinetic energy density of the functional wrt. the density rho on the grid and
        gradient of the kinetic energy density of the functional wrt.
        :math:`\sigma = |\nabla \rho|^2` on the grid
    """
    if get_gradient:
        density_and_gradient = density_and_gradient.requires_grad_(True)

    rho_torch = density_and_gradient[0]
    grad_torch = density_and_gradient[1:4]
    sigma = torch.einsum("ij,ij->j", grad_torch, grad_torch)

    energy_density = sum(
        f(rho_torch, sigma, **kwargs) for f in str_to_torch_functionals[functional]
    )

    if not get_gradient:
        return energy_density, None, None

    real_energy_density = energy_density * rho_torch
    energy_density_gradient_rho = torch.autograd.grad(
        outputs=real_energy_density.sum(),
        inputs=rho_torch,
        retain_graph=True,
    )[0]
    energy_density_gradient_s = torch.autograd.grad(
        outputs=real_energy_density.sum(),
        inputs=sigma,
        retain_graph=True,
    )[0]

    return energy_density, energy_density_gradient_rho, energy_density_gradient_s


def eval_torch_functionals(
    coeffs: torch.Tensor,
    ao: torch.Tensor,
    grid_weights: torch.Tensor,
    functionals: list[str],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Computes the density and evaluates the given functionals on the grid.

    Args:
        coeffs: coefficients of the basis functions
        ao: atomic orbitals of the molecule in the basis
        grid_weights: weights of grid on which to evaluate the functionals, same as used for
            the ao calculation
        functionals: list of functionals to evaluate

    Returns:
        List of tuples containing the energy and gradient of the given functionals.
    """
    output = {}
    coeffs.requires_grad = True
    rho = coeffs_to_rho_and_derivatives(coeffs, ao, max_derivative_order=1)

    for i, functional in enumerate(functionals):
        energy_density = torch_functional(rho, functional, get_gradient=False)[0]
        energy = torch.einsum("i,i,i->", rho[0], energy_density, grid_weights)
        # If it is the last iteration, we don't need to retain the graph
        retain_graph = i < len(functionals) - 1
        energy_gradient = torch.autograd.grad(energy, coeffs, retain_graph=retain_graph)[0]
        output[functional] = (energy, energy_gradient)
        del energy_density

    return output


def eval_torch_functionals_blocked(
    mol: gto.Mole,
    grid: dft.Grids,
    coeffs: torch.Tensor,
    functionals: list[str],
    pre_computed_aos: torch.Tensor | None = None,
    max_memory: float = 4000.0,
):
    """Evaluate torch functionals on the grid in a blocked fashion to reduce memory usage.

    Used in label generation, where based on memory usage the aos are saved or not.
    """
    max_block_size = compute_max_block_size(max_memory, mol.nao, heuristic_multiplier=8)
    grid_chunks = get_grid_blocks(grid.size, max_block_size)
    logger.trace(
        f"Computing functionals in {len(grid_chunks)} grid blocks of size {max_block_size}"
    )
    for i, (block_start, block_end) in enumerate(grid_chunks):
        if pre_computed_aos is None:
            ao_torch = dft.numint.eval_ao(mol, grid.coords[block_start:block_end], deriv=1)
            ao_torch = torch.as_tensor(ao_torch, dtype=torch.float64)
        else:
            ao_torch = pre_computed_aos[:, block_start:block_end]

        grid_weights = torch.as_tensor(grid.weights[block_start:block_end], dtype=torch.float64)
        functional_values = eval_torch_functionals(coeffs, ao_torch, grid_weights, functionals)

        if i == 0:
            output = functional_values
        else:
            # Assuming functional_values is a dict of tuples (value, derivative)
            for key in functional_values:
                output[key] = (
                    output[key][0] + functional_values[key][0],
                    output[key][1] + functional_values[key][1],
                )
    return output


def eval_torch_functionals_blocked_fast(
    ao: torch.Tensor,
    grid_weights: torch.Tensor,
    coeffs: torch.Tensor,
    functionals: list[str],
    max_memory: float = 4000.0,
):
    """Evaluate torch functionals on precomputed AOs in a blocked fashion to reduce memory usage.

    Used in density optimization, where the AOs are precomputed to achieve high speed.
    The estimated total size in MB is  8 * mol.nao * grid.size * 8 / mega_byte where 8 was empirically determined and
    8 comes from the size of a double. The maximum block size is then calculated to fit into the given max_memory.

    Args:
        ao: Atomic orbitals of the molecule in the basis as a tensor.
        grid_weights: Weights of the grid points on which to evaluate the functionals.
        coeffs: Coefficients of the basis functions.
        functionals: list of functionals to compute.
        max_memory: Guess of the maximum memory that should be taken by the aos in MB. Total usage might be higher.
            Defaults to the pyscf default of 4000MB.
    """
    max_block_size = compute_max_block_size(max_memory, ao.shape[-1], heuristic_multiplier=8)
    grid_size = grid_weights.shape[0]
    grid_chunks = get_grid_blocks(grid_size, max_block_size)

    for i, (block_start, block_end) in enumerate(grid_chunks):
        ao_block = ao[:, block_start:block_end]
        grid_weights_block = grid_weights[block_start:block_end]
        functional_values = eval_torch_functionals(
            coeffs, ao_block, grid_weights_block, functionals
        )
        if i == 0:
            output = functional_values
        else:
            for key in functional_values:
                output[key] = (
                    output[key][0] + functional_values[key][0],
                    output[key][1] + functional_values[key][1],
                )
    return output
