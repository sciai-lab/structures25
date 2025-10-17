from types import NoneType

import numpy as np
import pytest
from pyscf import dft, gto

from mldft.ofdft.energies import Energies
from mldft.ofdft.ofstate import OFState


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"mol": gto.M(atom="H 0 0 0; F 0 0 1")},
            {"mol": gto.Mole, "grid": NoneType, "coeffs": NoneType, "energy": NoneType},
        ),
        (
            {
                "mol": gto.M(atom="H 0 0 0; F 0 0 1"),
                "coeffs": np.random.random(10),
                "grid": dft.Grids(gto.M(atom="H 0 0 0; F 0 0 1")),
            },
            {
                "mol": gto.Mole,
                "grid": dft.Grids,
                "coeffs": np.ndarray,
                "energy": NoneType,
            },
        ),
        (
            {
                "mol": gto.M(atom="H 0 0 0; F 0 0 1"),
                "coeffs": np.random.random(10),
                "grid": dft.Grids(gto.M(atom="H 0 0 0; F 0 0 1")),
                "energy": Energies(),
            },
            {
                "mol": gto.Mole,
                "grid": dft.Grids,
                "coeffs": np.ndarray,
                "energy": Energies,
            },
        ),
    ],
)
def test_OFState_init(kwargs, expected):
    """Checks whether the OFState class is initialized correctly."""
    ofstate = OFState(**kwargs)

    assert isinstance(ofstate.mol, expected["mol"])
    assert isinstance(ofstate.grid, expected["grid"])
    assert isinstance(ofstate.coeffs, expected["coeffs"])
    assert isinstance(ofstate.energy, expected["energy"])
