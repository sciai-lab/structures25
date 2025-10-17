import pytest
import torch.testing
from torch_geometric.nn import global_add_pool

from mldft.utils.sad_guesser import SADGuesser, SADNormalizationMode


@pytest.fixture(params=SADNormalizationMode)
def normalization_mode(request):
    """Fixture for the three different normalization modes."""
    return request.param


@pytest.mark.parametrize("spherical_average", [True, False])
def test_sad_guesser(
    dummy_basis_info,
    dummy_dataset_statistics,
    dummy_sample_torch,
    normalization_mode,
    spherical_average,
):
    """Test the SAD guesser."""
    sad_guesser = SADGuesser.from_dataset_statistics(
        dataset_statistics=dummy_dataset_statistics,
        normalization_mode=normalization_mode,
        basis_info=dummy_basis_info,
        weigher_key="constant",
        spherical_average=spherical_average,
    )

    # Call the forward method
    sad_guess = sad_guesser(dummy_sample_torch)

    # Check the shape of the output
    assert sad_guess.shape == dummy_sample_torch.coeffs.shape

    # Check correct normalization
    if normalization_mode != SADNormalizationMode.NONE:
        n_electron_guess = (dummy_sample_torch.dual_basis_integrals * sad_guess).sum().item()
        print(f"{n_electron_guess-dummy_sample_torch.n_electron=:.2e}")
        # allowing for 1e-3 error in the number of electrons hurts, but this is just due to casting to float32.
        assert n_electron_guess == pytest.approx(dummy_sample_torch.n_electron, abs=1e-3)


@pytest.mark.parametrize("spherical_average", [True, False])
def test_sad_guesser_batched(
    dummy_basis_info,
    dummy_dataset_statistics,
    dummy_loader,
    normalization_mode,
    spherical_average,
):
    """Test the SAD guesser with batched input."""
    sad_guesser = SADGuesser.from_dataset_statistics(
        dataset_statistics=dummy_dataset_statistics,
        normalization_mode=normalization_mode,
        basis_info=dummy_basis_info,
        weigher_key="constant",
        spherical_average=spherical_average,
    )

    for batch in dummy_loader:
        # Call the forward method
        sad_guess = sad_guesser(batch)

        # Check the shape of the output
        assert sad_guess.shape == batch.coeffs.shape

        # Check correct normalization
        if normalization_mode != SADNormalizationMode.NONE:
            n_electron_guess = global_add_pool(
                batch.dual_basis_integrals * sad_guess, batch.coeffs_batch
            )
            n_electron = global_add_pool(batch.atomic_numbers, batch.batch)
            torch.testing.assert_close(n_electron_guess, n_electron, check_dtype=False)
