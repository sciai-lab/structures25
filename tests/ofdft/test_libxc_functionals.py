from functools import partial

import numpy as np
import pytest
from pyscf import dft

from mldft.ofdft import libxc_functionals

INTEGRATION_THRESHOLD = {"rel": 2e-6, "abs": 2e-6}


@pytest.mark.parametrize(
    "ao,max_derivative_order,should_raise",
    [
        (
            np.random.rand(1, 10, 10),
            0,
            False,
        ),  # Valid 3D array with max_derivative_order = 0
        (
            np.random.rand(4, 10, 10),
            1,
            False,
        ),  # Valid 3D array with max_derivative_order = 1
        (
            np.random.rand(3, 10, 10),
            1,
            True,
        ),  # Invalid 3D array with max_derivative_order = 0
        (
            np.random.rand(10, 10),
            0,
            False,
        ),  # Valid 2D array with max_derivative_order = 0
        (
            np.random.rand(10, 10),
            1,
            True,
        ),  # Invalid 2D array with max_derivative_order = 1
        (np.random.rand(5, 5, 5, 5), 0, True),  # Invalid dimensionality
        (
            np.random.rand(10, 10),
            -1,
            True,
        ),  # Invalid derivative order
    ],
)
def test_check_ao_dimensions_wrt_derivatives(ao, max_derivative_order, should_raise):
    """Checks whether _check_ao_dimensions_wrt_derivatives raises correctly."""
    if should_raise:
        with pytest.raises((AssertionError, ValueError)):
            libxc_functionals._check_ao_dimensions_wrt_derivatives(ao, max_derivative_order)
    else:
        libxc_functionals._check_ao_dimensions_wrt_derivatives(ao, max_derivative_order)


@pytest.mark.parametrize(
    "kinetic_functional, xc_functional",
    [
        ("LDA_K_TF", "PBE,PBE"),
        (None, "PBE,PBE"),
    ],
)
def test_eval_libxc_functionals(molecule_medium, kinetic_functional, xc_functional):
    """Test whether a libxc functional evaluation returns the correct types.

    No exact checks on the correct evaluation of the returned functional
    """
    grid = dft.Grids(molecule_medium)
    grid.level = 1
    grid.build()

    functionals = [kinetic_functional, xc_functional]
    max_deriv = libxc_functionals.required_derivative(functionals)

    ao = dft.numint.eval_ao(molecule_medium, grid.coords, deriv=max_deriv)

    coeffs = np.random.random(molecule_medium.nao)
    energy, potential_vector = libxc_functionals.eval_libxc_functionals(
        coeffs, functionals, grid, ao
    )
    assert isinstance(energy, list)
    assert isinstance(potential_vector, np.ndarray)
    assert potential_vector.dtype == np.float64


def gradient_finite_differences(functional, coeffs, delta=1e-6):
    """Numerically computes the gradient of a functional.

    Args:
        functional: A functional.
        coeffs: The coefficients of the Kohn-Sham orbitals.
        delta: The step size for the numerical derivative.

    Returns:
        The gradient of the functional.
    """
    grad = np.zeros_like(coeffs)
    for i in range(len(coeffs)):
        energy_p, _ = functional(coeffs + delta * np.eye(len(coeffs))[i])
        energy_m, _ = functional(coeffs - delta * np.eye(len(coeffs))[i])
        grad[i] = (energy_p - energy_m) / (2 * delta)

    return grad


@pytest.mark.parametrize("libxc_key", ["LDA_X", "GGA_X_PBE", "GGA_C_PBE"])
def test_functional_potential(molecule_small, libxc_key):
    """Tests the functional potential against a numerical gradient."""
    grid = dft.Grids(molecule_small)
    grid.level = 3
    grid.build()

    max_deriv = libxc_functionals.required_derivative([libxc_key])

    ao = dft.numint.eval_ao(molecule_small, grid.coords, deriv=max_deriv)
    functional = partial(
        libxc_functionals.eval_libxc_functionals,
        functionals=libxc_key,
        grid=grid,
        ao=ao,
    )

    np.random.seed(0)
    coeffs = np.random.random(molecule_small.nao)
    _, potential_vector = functional(coeffs)
    numerical_gradient = gradient_finite_differences(functional, coeffs, delta=1e-8)
    assert pytest.approx(potential_vector, **INTEGRATION_THRESHOLD) == numerical_gradient


def test_functional_energies_instances(molecule_medium):
    """Tests whether the functional returns new instances of Energies.

    If no instance of Eneries has been supplied at the creation of the functional.
    """
    libxc_key = "LDA_X"
    grid = dft.Grids(molecule_medium)
    grid.level = 3
    grid.build()

    max_deriv = libxc_functionals.required_derivative([libxc_key])

    ao = dft.numint.eval_ao(molecule_medium, grid.coords, deriv=max_deriv)

    coeffs = np.random.random(molecule_medium.nao)
    energies_1, _ = libxc_functionals.eval_libxc_functionals(coeffs, libxc_key, grid, ao)
    coeffs = np.random.random(molecule_medium.nao)
    energies_2, _ = libxc_functionals.eval_libxc_functionals(coeffs, libxc_key, grid, ao)

    assert energies_1 != energies_2


@pytest.mark.parametrize(
    "libxc_code, expected",
    [
        ("LDA_K_TF", 0),
        ("GGA_C_LYP,LDA_X", 1),
        ("PBE,PBE", 1),
        ("MGGA_X_BR89,GGA_C_LYP", 2),
    ],
)
def test_xc_type_to_deriv(libxc_code, expected):
    """Tests the _xc_type_to_deriv function to produce the correct return."""
    deriv = libxc_functionals._xc_type_to_deriv(libxc_code)
    assert deriv == expected


@pytest.mark.parametrize(
    "libxc_code, expected",
    [
        ("bla", KeyError),
    ],
)
def test_xc_type_to_deriv_raises(libxc_code, expected):
    """Tests the _xc_type_to_deriv to give correct error raises."""
    with pytest.raises(expected):
        libxc_functionals._xc_type_to_deriv(libxc_code)


@pytest.mark.parametrize(
    "libxc_codes, expected",
    [
        ("GGA_C_LYP,MGGA_X_BR89", NotImplementedError),
        ("HYB_GGA_XC_B3LYP", NotImplementedError),
    ],
)
def test_check_kxc_implementation_raises(libxc_codes, expected):
    """Tests the check_kxc_implementation to give correct error raises if not in OFDFT implemented
    functionals are requested."""
    with pytest.raises(expected):
        libxc_functionals.check_kxc_implementation(libxc_codes)


@pytest.mark.parametrize(
    "libxc_codes, expected",
    [
        (["LDA_K_TF"], 0),
        (["GGA_K_VW"], 1),
        (["GGA_K_VW", "LDA_X"], 1),
        (["GGA_K_VW", "LDA_X,GGA_C_LYP"], 1),
        (["GGA_K_VW", "MGGA_X_BR89,GGA_C_LYP"], 2),
    ],
)
def test_required_derivative(libxc_codes, expected):
    """Tests the _xc_type_to_deriv function to produce the correct derivative."""
    deriv = libxc_functionals.required_derivative(libxc_codes)
    assert deriv == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ("BLYP", "B88,LYP"),
        ("PBE", "PBE,PBE"),
        ("LDA_X", "LDA_X"),
    ],
)
def test_translate_xc_code(code, expected):
    libxc_code = libxc_functionals.translate_xc_code(code)
    assert libxc_code == expected


@pytest.mark.parametrize(
    "libxc_key", ["LDA_X", "GGA_X_PBE", "GGA_C_LYP", "PBE,PBE", "LDA_X,GGA_C_LYP"]
)
def test_get_energy_and_potential(molecule_small, libxc_key):
    """Only barebone test for the get_energy_and_potential function.

    Ensures that it produces some output without raising errors. The pyscf function against which
    it is tested is effectivley the same that is used inside the function.
    """
    mol = molecule_small
    nao = mol.nao

    grid = dft.Grids(mol)
    grid.level = 5
    grid.build()

    dm = np.random.random((nao, nao))
    ni = dft.numint.NumInt()

    deriv = libxc_functionals.required_derivative(libxc_key)
    xctype = dft.numint.libxc.xc_type(libxc_key)
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=deriv)
    rho = dft.numint.eval_rho1(mol, ao, dm, xctype=xctype)

    exc_expected, vxc_expected, _, _ = ni.eval_xc(libxc_key, rho, deriv=1)
    den = rho * grid.weights if xctype == "LDA" else rho[0] * grid.weights
    exc_expected = np.dot(exc_expected, den)
    exc, vxc = libxc_functionals._get_energy_and_functional_derivatives(libxc_key, rho, grid)

    assert np.isclose(exc_expected, exc)
    assert np.allclose(vxc, vxc_expected)
