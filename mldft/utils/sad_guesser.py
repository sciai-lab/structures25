from enum import Enum

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.of_data import OFData
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics


def compute_spherical_average(
    tensor_per_basis_func: torch.Tensor, l_per_basis_func: torch.Tensor
) -> torch.Tensor:
    r"""Spherically average a tensor over the basis functions.

    Args:
        tensor_per_basis_func: Tensor to average. Shape (n_basis, \*).
        l_per_basis_func: See ``l_per_basis_func`` in :py:class:`~mldft.ml.data.components.basis_info.BasisInfo`.
            Shape (n_basis,).
    """
    result = tensor_per_basis_func.clone()
    result[l_per_basis_func != 0] = 0  # only keep the s-orbitals
    return result


class SADNormalizationMode(Enum):
    """Enum for the normalization mode of the SAD guesser. Recommended is
    :attr:`PER_ATOM_WEIGHTED`.

    Attributes:
        NONE: No normalization.
        PER_ATOM_SIMPLE: Normalize the SAD guess per atom, i.e. scale the coefficients for each atom by a single factor
            to make it so the number of electrons in the coefficients belonging to the atom in the guess match its
            atomic number.
        PER_ATOM_WEIGHTED: Normalize the SAD guess per atom, i.e. adjust the coefficients for each atom such that the
            number of electrons in the coefficients belonging to the atom in the guess match its atomic number.
            The adjustment is weighted according to the standard deviation of the coefficients, more precisely, it is
            proportional to the standard deviation squared times the basis integral of the coefficients.
        PER_MOLECULE_SIMPLE: Normalize the SAD guess per molecule, i.e. scale the SAD guess with a single factor for
            each molecule such that the number of electrons in the guess matches the number of electrons in the
            molecule.
        PER_MOLECULE_WEIGHTED: Similar to :attr:`PER_ATOM_WEIGHTED`, but the adjustment is normalized per molecule,
            not per atom.
    """

    NONE = "none"
    PER_ATOM_SIMPLE = "per_atom_simple"
    PER_ATOM_WEIGHTED = "per_atom_weighted"
    PER_MOLECULE_SIMPLE = "per_molecule_simple"
    PER_MOLECULE_WEIGHTED = "per_molecule_weighted"


class SADGuesser(torch.nn.Module):
    """Class to generate SAD (Superposition of Atomic Densities) guess for samples or batches."""

    def __init__(
        self,
        sad_coeffs_per_basis_func: torch.Tensor,
        normalization_mode: SADNormalizationMode,
        coeff_std_per_basis_func: torch.Tensor | None = None,
    ) -> None:
        """Initialize the SAD guesser directly by specifying the SAD coefficients per basis
        function.

        Args:
            sad_coeffs_per_basis_func: SAD coefficients per basis function. Shape (n_basis,).
            normalization_mode: Normalization mode for the SAD guess, see :class:`SADNormalizationMode`.
            coeff_std_per_basis_func: Standard deviation of the coefficients per basis function. Required for
                weighted normalization.
        """
        super().__init__()
        self.normalization_mode = normalization_mode
        # we save the non-normalized SAD coefficients and normalize in the forward pass, as we do not know the
        # values of the basis integrals here
        self.register_buffer("sad_coeffs_per_basis_func", sad_coeffs_per_basis_func)

        # for weighted normalization, we need the standard deviation of the coefficients per basis function
        if normalization_mode in [
            SADNormalizationMode.PER_ATOM_WEIGHTED,
            SADNormalizationMode.PER_MOLECULE_WEIGHTED,
        ]:
            assert (
                coeff_std_per_basis_func is not None
            ), "coeff_std_per_basis_func is required for weighted normalization."
        if coeff_std_per_basis_func is not None:
            self.register_buffer("coeff_std_per_basis_func", coeff_std_per_basis_func)
        else:
            self.coeff_std_per_basis_func = None

    @classmethod
    def from_dataset_statistics(
        cls,
        dataset_statistics: DatasetStatistics,
        normalization_mode: SADNormalizationMode,
        basis_info: BasisInfo = None,
        weigher_key: str = "ground_state_only",
        spherical_average: bool = True,
    ) -> "SADGuesser":
        """Initialize the SAD guesser using a
        :class:`~mldft.ml.preprocess.dataset_statistics.DatasetStatistics`.

        Args:
            dataset_statistics: Dataset statistics object.
            normalization_mode: Normalization mode for the SAD guess, see :class:`SADNormalizationMode`.
            basis_info: Basis information object.
            weigher_key: Selects the sample weigher for the dataset statistics. Defaults to "ground_state_only".
            spherical_average: If True, the SAD coefficients are spherically averaged.
        """
        sad_coeffs_per_basis_func = dataset_statistics.load_statistic(weigher_key, "coeffs/mean")
        sad_coeffs_per_basis_func = torch.as_tensor(sad_coeffs_per_basis_func)
        coeff_std_per_basis_func = dataset_statistics.load_statistic(weigher_key, "coeffs/std")
        coeff_std_per_basis_func = torch.as_tensor(coeff_std_per_basis_func)
        if spherical_average:
            assert basis_info is not None, "Basis information is required for spherical averaging."
            l_per_basis_func = torch.as_tensor(basis_info.l_per_basis_func)
            sad_coeffs_per_basis_func = compute_spherical_average(
                sad_coeffs_per_basis_func, l_per_basis_func
            )
            coeff_std_per_basis_func = compute_spherical_average(
                coeff_std_per_basis_func, l_per_basis_func
            )

        return cls(
            sad_coeffs_per_basis_func=sad_coeffs_per_basis_func,
            coeff_std_per_basis_func=coeff_std_per_basis_func,
            normalization_mode=normalization_mode,
        )

    def forward(self, sample: OFData | OFBatch) -> torch.Tensor:
        """Generate SAD guess for samples or batches. The guess will be cast to the dtype of the
        coeffs in the sample.

        Args:
            sample (OFData | OFBatch): Sample or batch to generate SAD guess for

        Returns:
            torch.Tensor: SAD guess. Shape (n_samples, basis).
        """
        # create the guess
        coeffs_guess = self.sad_coeffs_per_basis_func[sample.basis_function_ind]

        # normalize the SAD guess if requested
        if self.normalization_mode in [
            SADNormalizationMode.PER_MOLECULE_SIMPLE,
            SADNormalizationMode.PER_MOLECULE_WEIGHTED,
        ]:
            coeffs_batch = sample.coeffs_batch if isinstance(sample, Batch) else None
            n_electrons_guess_per_mol = global_add_pool(
                sample.dual_basis_integrals * coeffs_guess,
                coeffs_batch,
            )
            n_electrons_per_mol = (
                global_add_pool(sample.atomic_numbers, sample.batch)
                if isinstance(sample, Batch)
                else sample.n_electron
            )
            if self.normalization_mode == SADNormalizationMode.PER_MOLECULE_SIMPLE:
                factor = n_electrons_per_mol / n_electrons_guess_per_mol
                if isinstance(sample, OFBatch):
                    factor = factor[coeffs_batch]
                coeffs_guess = coeffs_guess * factor
            else:  # PER_MOLECULE_WEIGHTED
                delta = (
                    self.coeff_std_per_basis_func[sample.basis_function_ind] ** 2
                    * sample.dual_basis_integrals
                )
                delta_factor = n_electrons_per_mol - n_electrons_guess_per_mol
                delta_factor = delta_factor / global_add_pool(
                    delta * sample.dual_basis_integrals, coeffs_batch
                )
                if isinstance(sample, OFBatch):
                    delta_factor = delta_factor[coeffs_batch]  # shape (n_coeffs)
                delta = delta * delta_factor
                coeffs_guess = coeffs_guess + delta
        elif self.normalization_mode in [
            SADNormalizationMode.PER_ATOM_SIMPLE,
            SADNormalizationMode.PER_ATOM_WEIGHTED,
        ]:
            n_electrons_guess_per_atom = global_add_pool(
                sample.dual_basis_integrals * coeffs_guess, batch=sample.coeff_ind_to_node_ind
            )
            n_electrons_per_atom = sample.atomic_numbers
            if self.normalization_mode == SADNormalizationMode.PER_ATOM_SIMPLE:
                factor = n_electrons_per_atom / n_electrons_guess_per_atom
                factor = factor[sample.coeff_ind_to_node_ind]
                coeffs_guess = coeffs_guess * factor
            else:  # PER_ATOM_WEIGHTED
                delta = (
                    self.coeff_std_per_basis_func[sample.basis_function_ind] ** 2
                    * sample.dual_basis_integrals
                )
                delta_factor = n_electrons_per_atom - n_electrons_guess_per_atom
                delta_factor = delta_factor / global_add_pool(
                    delta * sample.dual_basis_integrals, sample.coeff_ind_to_node_ind
                )
                delta = delta * delta_factor[sample.coeff_ind_to_node_ind]
                coeffs_guess = coeffs_guess + delta
        else:
            assert (
                self.normalization_mode == SADNormalizationMode.NONE
            ), f"Unknown normalization mode {self.normalization_mode}"
        return coeffs_guess.to(sample.coeffs.dtype)
