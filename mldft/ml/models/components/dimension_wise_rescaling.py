"""Dimension-wise Rescaling is one of the enhancement modules to cope with vast gradient ranges as
described in the second paragraph of section 4.3 in [M-OFDFT]_.

Rescaling takes place after the application of local frames and natural reparametrization.
To handle the still remaining scale trade off between density coefficients and gradients,
dimension-wise rescaling is introduced. As dimensions vary for different molecules,
density coefficients are centered and rescaled by atomic number specific biases
:math:`\\bar{\\mathbf{p}}_{Z, \\tau}` and scales :math:`\\lambda_{Z, \\tau}`, corresponding to
atomic number :math:`Z` and dimension :math:`\\tau`.
The scaling factor is found by simultaneously (up)scaling the coefficients and downscaling the
gradient labels, until the gradient arrives at an appropriate scale or the coefficient exceeds
its chosen scale.
The target gradient scale :math:`s_{\\text{grad}}` is set to :math:`0.05` and for coefficients,
the maximal scale is set to :math:`s_{\\text{coeff}} = 50`, as larger coefficient scales can
later be compressed by the Shrink Gate module. As the maximum gradient scale of a model can
indicate the capability of its fitting the gradient label, the maximum gradient is used as a proxy
for the label scale.
"""

from __future__ import annotations

import numpy as np
import torch
from e3nn import o3
from torch import Tensor
from torch.nn import Module
from torch_scatter import scatter

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics


class DimensionWiseRescaling(Module):
    """Center and rescale density coefficients and gradients dimension-wise."""

    def __init__(
        self,
        coeff_biases: Tensor | np.ndarray,
        coeff_stds: Tensor | np.ndarray,
        max_grads: Tensor | np.ndarray,
        s_coeff: float = 50,
        s_grad: float = 0.05,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize the DimensionWiseRescaling module.

        Args:
            coeff_biases (Tensor | np.ndarray): Concatenated tensor of dimensionwise coeff average
                per atomic number. Can be split using basis information.
            coeff_stds (Tensor | np.ndarray): Concatenated tensor of dimensionwise coeff standard
                deviation per atomic number. Can be split using basis information.
            max_grads (Tensor | np.ndarray): Concatenated tensor of dimensionwise max gradient per
                atomic number. Can be split using basis information.
            s_coeff (float): Maximal coefficient scale. Defaults to 50 (compare [M-OFDFT]_).
            s_grad (float): Target gradient scale. Defaults to 0.05 (compare [M-OFDFT]_).
            epsilon (float): Small number to avoid division by zero. Defaults to 1e-8.
        """
        super().__init__()

        # Leave in for now as test statistics come as np.ndarrays
        if isinstance(coeff_biases, np.ndarray):
            coeff_biases = torch.as_tensor(coeff_biases, dtype=torch.get_default_dtype())
        if isinstance(coeff_stds, np.ndarray):
            coeff_stds = torch.as_tensor(coeff_stds, dtype=torch.get_default_dtype())
        if isinstance(max_grads, np.ndarray):
            max_grads = torch.as_tensor(max_grads, dtype=torch.get_default_dtype())
        self.register_buffer("coeff_biases", coeff_biases)
        self.register_buffer("coeff_stds", coeff_stds)
        self.register_buffer("max_grads", max_grads)

        self.s_coeff = s_coeff  # not used in the forward pass, therefore no buffer
        self.s_grad = s_grad
        self.epsilon = epsilon

        self.init_lambda_z()  # initialize scaling factor

    @classmethod
    def from_dataset_statistics(
        cls,
        dataset_statistics: DatasetStatistics,
        basis_info: BasisInfo = None,
        weigher_key: str = "has_energy_label",
        equivariant: bool = False,
        s_coeff: int = 50,
        s_grad: float = 0.05,
        epsilon: float = 1e-8,
    ) -> DimensionWiseRescaling:
        """Instantiate class from a DatasetStatistics object holding the required maximum
        gradients, coefficient biases and coefficient standard deviations.

        Args:
            dataset_statistics (DatasetStatistics): DatasetStatistics object containing the
                needed coefficient mean (bias), their standard deviations and the maximum gradient.
            basis_info: Basis information object. Required if equivariant is True.
            equivariant: Whether to use the same scale factors for all components (different m's) of non-scalar fields.
            weigher_key (str): Selects the sample weigher for the dataset statistics.
            s_coeff (float): Maximal coefficient scale. Defaults to 50 (compare [M-OFDFT]_).
            s_grad (float): Target gradient scale. Defaults to 0.05 (compare [M-OFDFT]_).
            epsilon (float): Small number to avoid division by zero. Defaults to 1e-8.

        Returns:
            DimensionWiseRescaling: A new DimensionWiseRescaling object with the loaded data.
        """

        coeff_biases = dataset_statistics.load_statistic(weigher_key, "coeffs/mean")
        coeff_stds = dataset_statistics.load_statistic(weigher_key, "coeffs/std")
        if not equivariant:
            max_grads = dataset_statistics.load_statistic(
                weigher_key, "gradient_max_after_atom_ref"
            )
        else:
            max_grads = dataset_statistics.load_statistic(
                weigher_key, "gradient_max_after_scalar_atom_ref"
            )
            # max-pool over the different features.
            # Note: it might be better-motivated to instead compute the maximum gradient l2-norm
            max_grads = scatter(
                src=torch.as_tensor(max_grads),
                index=torch.as_tensor(basis_info.basis_func_to_shell),
                dim=0,
                reduce="max",
            ).numpy()
            max_grads = max_grads[basis_info.basis_func_to_shell]

            scalar_mask = basis_info.l_per_basis_func == 0

            coeff_biases[~scalar_mask] = 0

            irreps = "+".join([str(irreps) for irreps in basis_info.irreps_per_atom])
            norm = o3.Norm(o3.Irreps(irreps))
            coeff_stds = norm(torch.as_tensor(coeff_stds)).numpy()
            coeff_stds = coeff_stds[basis_info.basis_func_to_shell]

        return cls(
            coeff_biases=coeff_biases,
            coeff_stds=coeff_stds,
            max_grads=max_grads,
            s_coeff=s_coeff,
            s_grad=s_grad,
            epsilon=epsilon,
        )

    @classmethod
    def apply_to_dataset_statistics(
        cls,
        dataset_statistics: DatasetStatistics,
        output_path: str,
        weigher_key: str,
        **kwargs,
    ) -> DatasetStatistics:
        """Apply the dimension wise rescaling to a DatasetStatistics object, i.e. compute what the
        statistics would be after applying dimension wise rescaling to the dataset.

        Args:
            dataset_statistics (DatasetStatistics): DatasetStatistics object to be transformed, and which the
                dimension wise rescaling's parameters are based on.
            output_path (str): Path where the transformed statistics will be saved.
            weigher_key (str): Selects the sample weigher for the dataset statistics.
            **kwargs: Additional arguments to be passed to the constructor, see :meth:`from_dataset_statistics`.

        Returns:
            DatasetStatistics: The transformed DatasetStatistics object.
        """
        import os

        dimension_wise_rescaling = cls.from_dataset_statistics(
            dataset_statistics, weigher_key=weigher_key, **kwargs
        )
        lambda_z = dimension_wise_rescaling.lambda_z.numpy()
        epsilon = dimension_wise_rescaling.epsilon

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        transformed_statistics = DatasetStatistics(output_path, create_store=True)

        # Iterate through all keys in the original dataset_statistics
        for key in dataset_statistics.keys_recursive():
            weigher_key, statistic_key = key.split("/", 1)
            original_data = dataset_statistics.load_statistic(weigher_key, statistic_key)

            # Transform the data based on the statistic type
            if statistic_key == "coeffs/mean":
                transformed_data = (
                    original_data - dimension_wise_rescaling.coeff_biases.numpy()
                ) * lambda_z
            elif statistic_key in ["coeffs/std", "coeffs/abs_max"]:
                transformed_data = original_data * lambda_z
            elif statistic_key in [
                "gradient_label/abs_max",
                "gradient_label/max_after_atom_ref",
                "gradient_label/mean",
                "gradient_max_after_atom_ref",
            ]:
                transformed_data = original_data / (lambda_z + epsilon)
            else:
                continue  # do not save statistics unaffected by the rescaling or for which it is not implemented

            # Save the transformed data to the new DatasetStatistics object
            transformed_statistics.save_statistic(
                weigher_key, statistic_key, transformed_data, overwrite=False
            )

        return transformed_statistics

    def init_lambda_z(self) -> None:
        r"""Initialize the coefficient and gradient scaling :math:`\lambda_Z`.

        This is done according to eq (4) in [M-OFDFT]_:

        .. math::

            \lambda_{Z, \tau}
            = \begin{cases}
            \text{min} \left\{
            \frac{\mathrm{max\_grad}_{Z, \tau}}{s_{\text{grad}}} ,
            \frac{s_{\text{coeff}}}{\mathrm{std\_coeff}_{Z,\tau}},\right\}
            &\text{if } \mathrm{max\_grad}_{Z, \tau} > s_{\text{grad}}, \\
            1, &\text{otherwise}
            \end{cases}

        where :math:`\tau` refers to dimension in the given basis and :math:`Z` is the atomic
        number.
        """

        lambda_z = torch.where(
            self.max_grads > self.s_grad,
            torch.minimum(self.max_grads / self.s_grad, self.s_coeff / self.coeff_stds),
            torch.ones_like(self.max_grads),
        )

        self.register_buffer("lambda_z", lambda_z)

    def forward(self, data_sample: OFData) -> Tensor:
        """Compute shifted and rescaled coeffs.

        Arguments:
            data_sample (OFData): A sample from the OFData dataset. From this we extract
                density coefficients which are then rescaled by lambda_z.

        Returns:
            Tensor: The rescaled density coefficients.
        """

        coeffs = data_sample["coeffs"]
        coeffs_scaled = self.lambda_z[data_sample.basis_function_ind] * (
            coeffs - self.coeff_biases[data_sample.basis_function_ind]
        )
        return coeffs_scaled

    def rescale_coeffs_and_gradient(self, data_sample: OFData) -> tuple[Tensor, Tensor]:
        """Compute shifted and rescaled coeffs and inversely scaled gradients.

        Arguments:
            data_sample (OFData): A sample from the OFData dataset. From this we extract
                density coefficients and gradient labels which are then rescaled by lambda_z.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the rescaled density coefficients and the
                rescaled gradients.
        """

        coeffs_scaled = self.forward(data_sample)

        grads = data_sample["gradient_label"]
        grads_scaled = grads / (self.lambda_z[data_sample.basis_function_ind] + self.epsilon)

        return coeffs_scaled, grads_scaled
