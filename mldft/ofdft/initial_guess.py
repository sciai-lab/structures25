"""Module for generating initial guesses for the electronic structure."""

import numpy as np
import torch.nn
from loguru import logger
from pyscf import gto
from pyscf.scf.hf import init_guess_by_huckel, init_guess_by_minao

from mldft.datagen.methods.density_fitting import density_fitting_mol
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.convert_transforms import to_torch
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.utils.molecules import build_molecule_np


def label_guess(sample: OFData) -> torch.Tensor:
    """Return the ground truth label of the ground state as the initial guess.

    Args:
        sample: OFData object containing the ground state coeffs.

    Returns:
        Tensor: Ground state coefficients.
    """
    # clone() is necessary to avoid in-place operations on the ground state coefficients
    return sample.ground_state_coeffs.clone()


def perturbed_label_guess(sample: OFData, scale: float) -> torch.Tensor:
    """Return the perturbed ground truth label of the ground state as the initial guess.

    Args:
        sample: OFData object containing the ground state coeffs.
        scale: Perturbation to apply to the ground state coefficients.

    Returns:
        Tensor: Perturbed ground state coefficients.
    """
    coeffs = sample.ground_state_coeffs
    guess = coeffs + scale * torch.randn_like(coeffs)
    guess *= sample.n_electron / (sample.dual_basis_integrals @ guess)

    # trajectories are saved via mol id, so we need to change it to not overwrite the original
    sample.mol_id = sample.mol_id + "_" + str(hash(guess))

    return guess


def hueckel_guess_np(mol: gto.Mole, ks_basis: str = "6-31G(2df,p)") -> np.ndarray:
    """Return the Hückel initial guess.

    The Hückel initial guess is performed for the density matrix in the
    orbital basis. This is then transformed into the density basis using density
    fitting.

    Args:
        mol: Molecule object in the density (OF) basis.
        ks_basis: Orbital (KS) basis in which the Hückel guess is performed.

    Returns:
        Initial guess of the coefficients in the density basis.
    """
    mol_orbital = build_molecule_np(mol.atom_charges(), mol.atom_coords(), basis=ks_basis)
    dm = init_guess_by_huckel(mol_orbital)
    # density fitting into density basis
    coeffs = density_fitting_mol(dm, mol_orbital, mol)
    return coeffs


def hueckel_guess(
    sample: OFData,
    mol: gto.Mole,
    ks_basis: str = "6-31G(2df,p)",
    transform: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the Hückel initial guess.

    The Hückel initial guess is performed for the density matrix in the
    orbital basis. This is then transformed into the density basis using density
    fitting. Finally, the coefficients are rescaled to ensure the correct number
    of electrons.

    Args:
        sample: OFData object containing transformed basis_integrals and coeffs.
        mol: Molecule object containing information on the density (OF) basis.
        ks_basis: Orbital (KS) basis in which the Hückel guess is performed.
        normalize: Whether to rescale the coefficients to ensure the correct number of electrons.

    Returns:
        Initial guess of the coefficients in the transformed density basis.
    """
    coeffs = hueckel_guess_np(mol, ks_basis)
    coeffs = torch.as_tensor(coeffs, dtype=torch.float64, device=sample.coeffs.device)
    if transform:
        coeffs = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
    if normalize:  # rescale to correct number of electrons
        n_guess = sample.dual_basis_integrals @ coeffs
        coeffs *= mol.nelectron / n_guess
    return coeffs


def minao_guess_np(mol: gto.Mole, ks_basis: str = "6-31G(2df,p)") -> np.ndarray:
    """Return the MINAO initial guess.

    The MINAO initial guess is performed for the density matrix in the
    orbital basis. This is then transformed into the density basis using density
    fitting.

    Args:
        mol: Molecule object containing information on the density (OF) basis.
        ks_basis: Orbital (KS) basis in which the MINAO guess is performed.

    Returns:
        Initial guess of the coefficients in the density basis.
    """
    mol_orbital = build_molecule_np(mol.atom_charges(), mol.atom_coords(), basis=ks_basis)
    dm = init_guess_by_minao(mol_orbital)
    # density fitting into density basis
    coeffs = density_fitting_mol(dm, mol_orbital, mol)
    return coeffs


def minao_guess(
    sample: OFData,
    mol: gto.Mole,
    ks_basis: str = "6-31G(2df,p)",
    transform: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the optionally transformed and normalized MINAO initial guess.

    The MINAO initial guess is performed for the density matrix in the
    orbital basis. It is transformed into the density basis using density
    fitting. Finally, the coefficients can optionally be transformed according to the samples

    Args:
        sample: OFData object containing transformed basis_integrals and coeffs
        mol: Molecule object containing information on the density (OF) basis.
        ks_basis: Orbital (KS) basis in which the MINAO guess is performed.
        transform: Whether to transform the coefficients into the density basis.
        normalize: Whether to rescale the coefficients to ensure the correct number of electrons.

    Returns:
        Initial guess of the coefficients in the transformed density basis.
    """
    coeffs = minao_guess_np(mol, ks_basis)
    coeffs = torch.as_tensor(coeffs, dtype=torch.float64, device=sample.coeffs.device)
    if transform:
        coeffs = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
    if normalize:  # rescale to correct number of electrons
        n_guess = sample.dual_basis_integrals @ coeffs
        coeffs *= mol.nelectron / n_guess
    return coeffs


def proj_minao_guess(
    sample: OFData,
    mol: gto.Mole,
    module: torch.nn.Module,
    ks_basis: str = "6-31G(2df,p)",
    normalize: bool = True,
) -> torch.Tensor:
    """Return the "projected" MINAO initial guess.

    First, the `py:meth:minao_guess` is performed for the density matrix in the
    orbital basis. This is then transformed into the density basis using density
    fitting. Finally, the coefficients are rescaled to ensure the correct number
    of electrons.

    Args:
        sample: OFData object containing transformed basis_integrals and coeffs.
        mol: Molecule object containing information on the density (OF) basis.
        module: Module used to predict the correction to / projection of the initial guess.
        ks_basis: Orbital (KS) basis in which the MINAO guess is performed.
        normalize: Whether to rescale the coefficients to ensure the correct number of electrons.

    Returns:
        Initial guess of the coefficients in the transformed density basis.
    """
    # Transformation already happens in minao_guess
    coeffs_minao = minao_guess(sample, mol, ks_basis)
    sample_for_proj = to_torch(sample.detach().clone(), float_dtype=module.dtype)
    sample_for_proj.coeffs = coeffs_minao.type(module.dtype)
    coeffs_delta = module(sample_for_proj)[2]
    coeffs_delta = coeffs_delta.type(torch.float64)
    coeffs = coeffs_minao - coeffs_delta
    if normalize:
        n_guess = sample.dual_basis_integrals @ coeffs
        coeffs *= mol.nelectron / n_guess
    return coeffs


def dumb_guess(sample: OFData, mol: gto.Mole) -> torch.Tensor:
    """Generate a dumb initial guess for the electronic structure calculation.

    The dumb guess sets the coefficients of the atomic basis functions to 1 for
    functions with a positive normalization constant (norm > 0). The coefficients are
    then scaled to ensure the correct number of electrons. This currently only
    works for the density basis and is only intended for testing purposes.

    Args:
        sample: OFData object containing the basis_integrals.
        mol: Molecule object containing information about the atomic
            structure and basis set.

    Returns:
        np.ndarray: Initial guess for the coefficients of the density.
    """
    coeffs = torch.zeros(mol.nao, dtype=torch.float64, device=sample.coeffs.device)
    # get nonzero basis functions
    basis_integrals = torch.as_tensor(
        get_normalization_vector(mol), dtype=sample.coeffs.dtype, device=sample.coeffs.device
    )
    if hasattr(sample, "transformation_matrix"):
        basis_integrals = transform_tensor_with_sample(
            sample, basis_integrals, representation=Representation.VECTOR
        )
    else:
        logger.warning("Sample has no transformation matrix, using untransformed basis integrals.")
    coeffs[basis_integrals > 0] = 1
    n_guess = sample.dual_basis_integrals @ coeffs
    coeffs *= mol.nelectron / n_guess
    return coeffs
