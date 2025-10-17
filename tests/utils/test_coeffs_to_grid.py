import numpy as np
import pytest

from mldft.utils.coeffs_to_grid import (
    coeffs_to_grad_rho,
    coeffs_to_laplace_rho,
    coeffs_to_rho,
    coeffs_to_rho_and_derivatives,
)


# test cases
# current torch einsum does not like numpy arrays with dtype=int it seems
# at least forcing float dtype resolved the issue
# I think this is fine as the ao and coeffs arrays will be floats in any realistic case
@pytest.mark.parametrize(
    "coeffs, ao, rho_expected",
    [
        (
            np.array([1, 2], dtype=float),
            np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
            np.array([5, 11, 17], dtype=float),
        ),
        (
            np.array([0.5, 0.5, 0.5]),
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=float),
            np.array([3.0, 7.5]),
        ),
    ],
)
def test_coeffs_to_rho(coeffs, ao, rho_expected):
    """Test the function coeffs_to_rho."""
    assert np.allclose(coeffs_to_rho(coeffs, ao), rho_expected)


# additional edge cases
def test_invalid_input():
    """Test that an invalid input raises an AssertionError."""
    with pytest.raises(AssertionError):
        coeffs = np.array([1, 2, 3, 4])
        ao = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        coeffs_to_rho(coeffs, ao)


def test_output_shape():
    """Test that the output shape is correct."""
    coeffs = np.array([1, 2, 3])
    ao = np.array([[1, 2, 3], [4, 5, 6]])
    rho = coeffs_to_rho(coeffs, ao)
    assert rho.shape == (2,)


@pytest.mark.parametrize(
    "coeffs, ao, all_derivs",
    [
        (
            np.array([0.3, 0.7]),
            np.ones((10, 4, 2)),
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [3, 3, 3, 3],
                ]
            ),
        ),
        (
            np.array([1, 2, 3]),
            np.arange(60).reshape(10, 2, 3),
            np.array([[8, 26], [44, 62], [80, 98], [116, 134], [744, 798]]),
        ),
    ],
)
def test_coeffs_rho_et_alii(coeffs, ao, all_derivs):
    """Test that the functions coeffs_to_rho, coeffs_to_grad_rho, coeffs_to_laplace_rho, and
    coeffs_to_rho_and_derivatives."""
    assert np.allclose(coeffs_to_rho(coeffs, ao), all_derivs[0])
    assert np.allclose(coeffs_to_rho(coeffs, ao[0]), all_derivs[0])
    assert np.allclose(coeffs_to_rho(coeffs, ao[:1]), all_derivs[0])

    assert np.allclose(coeffs_to_grad_rho(coeffs, ao), all_derivs[1:4])
    assert np.allclose(coeffs_to_grad_rho(coeffs, ao[:4]), all_derivs[1:4])

    assert np.allclose(coeffs_to_laplace_rho(coeffs, ao), all_derivs[4])

    assert np.allclose(
        coeffs_to_rho_and_derivatives(coeffs, ao, max_derivative_order=0), all_derivs[0]
    )
    assert np.allclose(
        coeffs_to_rho_and_derivatives(coeffs, ao, max_derivative_order=1),
        all_derivs[:4],
    )
    assert np.allclose(
        coeffs_to_rho_and_derivatives(coeffs, ao, max_derivative_order=2), all_derivs
    )
