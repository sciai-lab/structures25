# Collection of functions to calculate the different flavors of overlap matrices

from typing import Optional

import torch

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ofdft.basis_integrals import get_overlap_matrix
from mldft.utils.molecules import build_molecule_ofdata


def get_sample_overlap_matrix(
    basis_info: BasisInfo, sample: OFData, spin: Optional[int] = None
) -> torch.tensor:
    """Constructs the overlap matrix for the given sample.

    Args:
        basis_info: The basis_info for the given sample.
        sample: The sample for which the overlap matrix is constructed.
        spin: The spin of the molecule. Defaults to None.

    Returns:
        overlap_matrix: The overlap matrix for the given sample.
    """
    mol = build_molecule_ofdata(ofdata=sample, basis=basis_info, spin=None)
    sample_overlap_matrix = get_overlap_matrix(mol)

    return sample_overlap_matrix
