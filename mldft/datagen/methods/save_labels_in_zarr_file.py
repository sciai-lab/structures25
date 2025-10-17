""" Saves the calculated labels for OFDFT - calculations into a .zarr file."""

from pathlib import Path

import numpy as np
import zarr
from loguru import logger
from pyscf import gto

from mldft.ofdft.basis_integrals import get_normalization_vector


def save_density_fitted_data(
    path: Path,
    mol_with_density_basis: gto.Mole,
    kohn_sham_basis: str,
    path_to_basis_info: str,
    molecular_data: dict,
    mol_id: str,
    **dataset_kwargs,
):
    """Saves the labels for machine learning into a .zarr file. See Notes/Dataformat for an overview
    of the structure of the zarr file and the overall dataformat.
    Args:
        path: Path to the location where the zarr file should be saved
        mol_with_density_basis: the molecule object in the orbital basis
        kohn_sham_basis: nwchem string of the basis in which the ks-calculation was carried out
        path_to_basis_info: relative path to the basis info file
        molecular_data: the dict which contains the results of the density fitting
        mol_id: string to identify the molecule
        dataset_kwargs: keyword arguments to be passed to zarr

    Notes:
        strings and integers are saved as numpy arrays and can be accessed with [()]
    """
    logger.info(f"Saving labels for molecule {mol_id} to {path}")
    of_coefficients = molecular_data["of_coeffs"]
    n_scf_steps, n_basis_functions = of_coefficients.shape
    dataset_kwargs.setdefault("compressor", None)
    atomic_positions = mol_with_density_basis.atom_coords()
    atomic_numbers = mol_with_density_basis.atom_charges()
    with zarr.ZipStore(path, mode="w") as zipstore:
        root = zarr.open(zipstore, mode="w")
        # Create the groups and datasets
        geometry = root.create_group("geometry")
        geometry.create_dataset("mol_id", data=mol_id, **dataset_kwargs)
        geometry.create_dataset("atom_pos", data=atomic_positions, **dataset_kwargs)
        geometry.create_dataset(
            "atomic_numbers",
            data=atomic_numbers,
            dtype=np.uint8,
            **dataset_kwargs,
        )

        of_labels = root.create_group("of_labels")
        of_labels.create_dataset("n_scf_steps", data=n_scf_steps, **dataset_kwargs)
        of_labels.create_dataset("basis", data=path_to_basis_info, **dataset_kwargs)
        of_energies = of_labels.create_group("energies")
        # Saving the spatial of labels
        of_spatial = of_labels.create_group("spatial")
        # We save the basis integrals here to have them saved similarly for all label types (in transformed folders they
        # are required always).
        dual_basis_integrals = get_normalization_vector(
            mol_with_density_basis, clip_minimum=1.6e-15
        )
        # We stack them so that it is easier to read them later, it does not need more memory since zarr notices.
        dual_basis_integrals_stacked = np.stack([dual_basis_integrals for _ in range(n_scf_steps)])
        # Without transformations, the basis integrals are the same as the dual basis integrals
        of_spatial.create_dataset(
            "dual_basis_integrals",
            data=dual_basis_integrals_stacked,
            chunks=(1, n_basis_functions),
            dtype=np.float64,
        )
        # Saving the ks labels for comparison
        ks_labels = root.create_group("ks_labels")
        ks_labels.create_dataset("basis", data=kohn_sham_basis, **dataset_kwargs)
        ks_energies = ks_labels.create_group("energies")
        # Read out the data from the dict, save according to the name of the key
        for key, item in molecular_data.items():
            if key.startswith("of_"):
                # Use chunking for the multi-dimensional datasets (gradients and coefficients)
                if item.ndim == 1:
                    of_energies.create_dataset(key[3:], data=item, **dataset_kwargs)
                else:
                    of_spatial.create_dataset(
                        key[3:],
                        data=item,
                        chunks=(1, n_basis_functions),
                        **dataset_kwargs,
                    )
            elif key.startswith("ks_"):
                ks_energies.create_dataset(key[3:], data=item, **dataset_kwargs)
