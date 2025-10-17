import numpy as np
import pytest
import torch

from mldft.ml.models.components.dimension_wise_rescaling import DimensionWiseRescaling


@pytest.fixture
def dummy_stats(dummy_sample, dummy_basis_info):
    """Return dummy statistics for the DimensionWiseRescaling class.

    Args:
        dummy_sample (OFData): Dummy sample.
        dummy_basis_info (BasisInfo): Dummy basis info.

    Returns:
        tuple: Tuple of tensors containing the coefficients biases, coefficients standard
          deviations and maximum gradients for each atomic number.
    """

    # list of arrays pertaining to atomic number z
    grads = [
        np.array(
            [
                dummy_sample.split_field_by_atom(dummy_sample["gradient_label"])[i]
                for i in np.where(dummy_sample["atomic_numbers"] == z)[0]
            ]
        )
        for z in dummy_basis_info.atomic_numbers  # loop over unique atomic numbers
    ]

    # list of arrays pertaining to atomic number z
    coeffs = [
        np.array(
            [
                dummy_sample.split_field_by_atom(dummy_sample["coeffs"])[i]
                for i in np.where(dummy_sample["atomic_numbers"] == z)[0]
            ]
        )
        for z in dummy_basis_info.atomic_numbers  # loop over unique atomic numbers
    ]

    max_grads = torch.cat([torch.max(torch.from_numpy(g), dim=0)[0] for g in grads])

    # Compute the max and std of coeffs for each atomic number
    mean_coeffs = torch.cat([torch.max(torch.from_numpy(c), dim=0)[0] for c in coeffs])
    std_coeffs = torch.cat([torch.std(torch.from_numpy(c), dim=0) for c in coeffs])

    return max_grads, mean_coeffs, std_coeffs


def test_init(dummy_basis_info, dummy_stats):
    """Test for correct class initialization."""

    # Unpack dummy_stats
    max_grads, coeff_biases, coeff_stds = dummy_stats

    # Initialize a DimensionWiseRescaling instance
    rescaling = DimensionWiseRescaling(
        coeff_biases=coeff_biases,
        coeff_stds=coeff_stds,
        max_grads=max_grads,
        s_coeff=1.0,
        s_grad=1.0,
        epsilon=1e-8,
    )

    # Check if the buffers and attributes are correctly initialized
    assert torch.all(rescaling.coeff_biases == coeff_biases)
    assert torch.all(rescaling.coeff_stds == coeff_stds)
    assert torch.all(rescaling.max_grads == max_grads)
    assert rescaling.s_coeff == 1.0
    assert rescaling.s_grad == 1.0
    assert rescaling.epsilon == 1e-8


@pytest.mark.parametrize("equivariant", (False, True))
def test_from_file(dummy_basis_info, dummy_dataset_statistics, equivariant):
    """Test whether class initialization by file works."""

    rescaling = DimensionWiseRescaling.from_dataset_statistics(
        dataset_statistics=dummy_dataset_statistics,
        weigher_key="constant",
        basis_info=dummy_basis_info if equivariant else None,
        equivariant=equivariant,
    )

    if not equivariant:
        assert torch.all(
            rescaling.coeff_biases
            == torch.as_tensor(
                dummy_dataset_statistics.load_statistic("constant", "coeffs/mean"),
                dtype=torch.get_default_dtype(),
            )
        )
        assert torch.all(
            rescaling.coeff_stds
            == torch.as_tensor(
                dummy_dataset_statistics.load_statistic("constant", "coeffs/std"),
                dtype=torch.get_default_dtype(),
            )
        )
        assert torch.all(
            rescaling.max_grads
            == torch.as_tensor(
                dummy_dataset_statistics.load_statistic("constant", "gradient_max_after_atom_ref"),
                dtype=torch.get_default_dtype(),
            )
        )


def test_forward(dummy_basis_info, dummy_dataset_statistics, dummy_sample_torch):
    """Test the forward method of the DimensionWiseRescaling class."""

    rescaling = DimensionWiseRescaling(
        coeff_biases=dummy_dataset_statistics.load_statistic("constant", "coeffs/mean"),
        coeff_stds=dummy_dataset_statistics.load_statistic("constant", "coeffs/std"),
        max_grads=dummy_dataset_statistics.load_statistic("constant", "gradient_label/abs_max"),
    )

    # Call the forward method
    coeffs_scaled, grads_scaled = rescaling.rescale_coeffs_and_gradient(dummy_sample_torch)

    # Check the shapes of the output tensors
    assert coeffs_scaled.shape == dummy_sample_torch["coeffs"].shape
    assert grads_scaled.shape == dummy_sample_torch["gradient_label"].shape

    # Check finiteness of the output tensors
    assert torch.isfinite(coeffs_scaled).all()
    assert torch.isfinite(grads_scaled).all()


def test_lambda_init(dummy_basis_info):
    """Test the init_lambda_z method of the DimensionWiseRescaling class."""

    # coeff_biases remain untouched by the init_lambda_z method
    coeff_biases = torch.cat(
        [torch.ones(basis_dim) for basis_dim in dummy_basis_info.basis_dim_per_atom]
    )

    s_coeff_1 = 1
    s_grad_1 = 1
    # for this case we expect the lambda_z to be 1 as max_grad < s_grad everywhere
    coeff_stds_1 = torch.cat(
        [torch.ones(basis_dim) * 1 / 2 for basis_dim in dummy_basis_info.basis_dim_per_atom]
    )
    max_grads_1 = torch.cat(
        [torch.ones(basis_dim) * 1 / 3 for basis_dim in dummy_basis_info.basis_dim_per_atom]
    )

    rescaling_1 = DimensionWiseRescaling(
        coeff_biases=coeff_biases,
        coeff_stds=coeff_stds_1,
        max_grads=max_grads_1,
        s_coeff=s_coeff_1,
        s_grad=s_grad_1,
    )

    # assert that lambda_z is 1 everywhere and has the same shape as biases, stds, and grads
    assert torch.all(rescaling_1.lambda_z == torch.ones_like(coeff_stds_1))

    s_coeff_2 = 10
    s_grad_2 = 1
    # for this case we expect the lambda_z to be max_grad/s_grad as max_grad > s_grad,
    # and max_grad/s_grad < s_coeff/coeff_stds for all elements
    coeff_stds_2 = torch.cat(
        [torch.ones(basis_dim) * 1 for basis_dim in dummy_basis_info.basis_dim_per_atom]
    )
    max_grads_2 = torch.cat(
        [torch.ones(basis_dim) * 2 for basis_dim in dummy_basis_info.basis_dim_per_atom]
    )

    rescaling_2 = DimensionWiseRescaling(
        coeff_biases=coeff_biases,
        coeff_stds=coeff_stds_2,
        max_grads=max_grads_2,
        s_coeff=s_coeff_2,
        s_grad=s_grad_2,
    )

    assert torch.all(rescaling_2.lambda_z == max_grads_2 / s_grad_2)

    s_coeff_3 = 1 / 3
    s_grad_3 = 1 / 2
    # for this case we expect the lambda_z to be s_coeff/std_coeff as max_grad > s_grad,
    # and  s_coeff / coeff_stds < max_grad/s_grad
    coeff_stds_3 = torch.cat(
        [torch.ones(basis_dim) * 1 for basis_dim in dummy_basis_info.basis_dim_per_atom]
    )
    max_grads_3 = torch.cat(
        [torch.ones(basis_dim) * 1 for basis_dim in dummy_basis_info.basis_dim_per_atom]
    )

    rescaling_3 = DimensionWiseRescaling(
        coeff_biases=coeff_biases,
        coeff_stds=coeff_stds_3,
        max_grads=max_grads_3,
        s_coeff=s_coeff_3,
        s_grad=s_grad_3,
    )

    # assert that lambda_z is 1 everywhere and has the same shape as biases, stds, and grads
    assert torch.all(rescaling_3.lambda_z == s_coeff_3 / coeff_stds_3)


def test_vectorized_lambda(dummy_basis_info, dummy_dataset_statistics, dummy_sample_torch):
    """Test the vectorized_lambda method of the DimensionWiseRescaling class."""

    rescaling = DimensionWiseRescaling(
        coeff_biases=dummy_dataset_statistics.load_statistic("constant", "coeffs/mean"),
        coeff_stds=dummy_dataset_statistics.load_statistic("constant", "coeffs/std"),
        max_grads=dummy_dataset_statistics.load_statistic("constant", "gradient_label/abs_max"),
    )

    # Call the forward method
    coeffs_scaled, grads_scaled = rescaling.rescale_coeffs_and_gradient(dummy_sample_torch)

    # compute with the non-vectorized method
    atom_ind = dummy_basis_info.atomic_number_to_atom_index[dummy_basis_info.atomic_numbers]
    n_basis_per_atom = dummy_basis_info.basis_dim_per_atom[atom_ind]
    lambda_z_split = torch.split(rescaling.lambda_z, list(n_basis_per_atom), dim=0)
    coeff_biases_split = torch.split(rescaling.coeff_biases, list(n_basis_per_atom), dim=0)
    coeffs_scaled_list = []
    grads_scaled_list = []

    atomic_numbers = dummy_sample_torch["atomic_numbers"]
    coeffs = dummy_sample_torch["coeffs"]
    grads = dummy_sample_torch["gradient_label"]

    if isinstance(coeffs, np.ndarray):
        coeffs = torch.from_numpy(coeffs)
    if isinstance(grads, np.ndarray):
        grads = torch.from_numpy(grads)

    coeffs = torch.split(coeffs, dummy_sample_torch.n_basis_per_atom.tolist())
    grads = torch.split(grads, dummy_sample_torch.n_basis_per_atom.tolist())

    for z, coeff, grad in zip(atomic_numbers, coeffs, grads):
        # assert isinstance(z, int), f"Atomic number {z} is not an integer."
        z = z.item()
        nonvectorized_coeff_scaled = (
            coeff - coeff_biases_split[dummy_basis_info.atomic_number_to_atom_index[z]]
        ) * lambda_z_split[dummy_basis_info.atomic_number_to_atom_index[z]]
        coeffs_scaled_list.append(nonvectorized_coeff_scaled)

        nonvectorized_grad_scaled = grad / (
            lambda_z_split[dummy_basis_info.atomic_number_to_atom_index[z]] + rescaling.epsilon
        )
        grads_scaled_list.append(nonvectorized_grad_scaled)

    nonvectorized_coeffs_scaled = torch.cat(coeffs_scaled_list)
    nonvectorized_grads_scaled = torch.cat(grads_scaled_list)

    # check that the vectorized and non-vectorized methods give the same result
    assert torch.allclose(coeffs_scaled, nonvectorized_coeffs_scaled)
    assert torch.allclose(grads_scaled, nonvectorized_grads_scaled)
