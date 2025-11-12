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

    @classmethod
    def from_default_qm9_stats(cls) -> "SADGuesser":
        """Create a SADGuesser based on our QM9 ground state statistics and the default even-
        tempered 2.5 basis set derived from pyscf's 6-31G(2df,p) basis set.

        The resulting guesser will not be compatible with other basis sets.
        """
        # fmt: off
        l_per_basis_func = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,]  # noqa: E231
        sad_coeffs = [6.879346140081655202e-03,1.346554349488622571e-02,3.000781907930980333e-02,9.763035801535269176e-02,1.028251730984331147e-01,2.534381918784077772e-02,3.671842566552804138e-02,6.373159173621188722e-02,1.937477416673811237e-01,6.121374071774996484e-01,1.360681592254237238e+00,2.325081813805514042e+00,1.555229770191303151e+00,8.254073501290468129e-02,-9.431598263205175081e-02,5.272629553588834561e-01,2.423813404172450725e-01,5.797804986204746491e-02,7.943979795443729952e-02,2.863147507704857930e-01,8.487544166677977175e-01,1.884134912330501566e+00,3.025561513761827914e+00,1.780303605135681533e+00,-8.981499311060532620e-02,5.404303980263149704e-02,8.578269425948997773e-01,1.812718739537717283e-01,8.442177081487703405e-02,8.322765841619282623e-02,3.838363890684770574e-01,1.071167036513013926e+00,2.403716348003120729e+00,3.753503548539403489e+00,2.085555526542524518e+00,-2.306240490263972154e-01,2.811697365198463916e-01,1.190218322572906384e+00,2.196238208966461714e-01,1.062529340257765409e-01,7.702809347959792274e-02,4.555141904087024463e-01,1.223097169230626280e+00,2.819749603222720946e+00,4.499604593228534455e+00,2.659899411338153108e+00,-3.120198276057970510e-01,4.853861864105580848e-01,1.585582373756699859e+00,3.726822900782292280e-01]  # noqa: E231
        sad_stds = [1.113725298493478720e-03,2.703552701117143200e-03,4.884482331405855769e-03,5.070841684378680006e-03,1.388588786874754739e-02,3.680523833173065477e-03,2.493173392241223446e-03,5.931043092808873859e-03,8.355420011504790731e-03,9.124503544427047669e-03,1.052342185783440055e-02,9.855435356307708228e-03,1.053301937850749555e-02,1.311107454420325051e-02,1.836696582930844382e-02,3.170677121643356716e-02,6.030128431865436345e-02,2.967053884693083860e-03,6.478440972368789894e-03,9.231438994659538558e-03,8.561191400808007557e-03,1.325591275445986254e-02,9.298445656035303139e-03,1.251855068537524769e-02,1.880712839284329196e-02,1.357498077050115458e-02,2.071859687772064154e-02,6.987596643479006275e-02,3.995106160399852259e-03,9.344949314298911514e-03,1.305453443720905674e-02,1.433696460712484758e-02,1.636427634878890452e-02,1.453965752051571483e-02,1.644739136643109945e-02,1.871998638392197156e-02,1.713333171322247578e-02,1.618684120544441538e-02,6.034636811189027228e-02,9.968336141305022755e-03,2.016786485738897988e-02,2.201986247114876699e-02,1.978658290386435537e-02,1.695697664556283790e-02,1.610080827149495616e-02,1.548998597559896745e-02,1.597413985401835781e-02,2.016928340049425547e-02,3.146472134374125179e-02,3.329750857684422860e-02]  # noqa: E231
        # fmt: on
        l_per_basis_func = torch.tensor(l_per_basis_func)
        sad_coeffs_compact = torch.tensor(sad_coeffs)
        sad_stds_compact = torch.tensor(sad_stds)
        sad_coeffs_full = torch.zeros(len(l_per_basis_func))
        sad_stds_full = torch.zeros(len(l_per_basis_func))
        sad_coeffs_full[l_per_basis_func == 0] = sad_coeffs_compact
        sad_stds_full[l_per_basis_func == 0] = sad_stds_compact
        return cls(
            sad_coeffs_per_basis_func=sad_coeffs_full,
            coeff_std_per_basis_func=sad_stds_full,
            normalization_mode=SADNormalizationMode.PER_ATOM_WEIGHTED,
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
