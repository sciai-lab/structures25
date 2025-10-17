import pytest
import torch
from lightning import LightningModule
from pyscf import dft, gto

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft.energies import Energies
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.utils.molecules import build_mol_with_even_tempered_basis
from mldft.utils.utils import set_default_torch_dtype


@set_default_torch_dtype(torch.float64)
def test_functional_factory():
    """Basis test for the functional factory using a dummy torch module."""

    mol = gto.M(atom="O 0 0 0; C 0 1 0", basis="6-31G(2df,p)")
    mol = build_mol_with_even_tempered_basis(mol, beta=2.5)
    n = mol.nao

    grid = dft.Grids(mol)
    grid.build()
    ao = torch.Tensor(dft.numint.eval_ao(mol, grid.coords))

    class DummyModule(LightningModule):
        """Dummy module for testing the functional factory."""

        def __init__(self, target_key: str):
            """Initialize the dummy module."""
            super().__init__()
            self.target_key = target_key

        def sample_forward(self, sample: OFData):
            """Apply the DummyModule to the sample.

            Use 42 as energy and a tensor full of ones in the correct size as gradient.
            """
            sample.add_item("pred_energy", torch.tensor(42), Representation.SCALAR)
            sample.add_item(
                "pred_gradient", torch.full(size=(n,), fill_value=1), Representation.GRADIENT
            )
            return sample

    model = DummyModule(target_key="dummy")
    model.training = True

    func_factory = FunctionalFactory(model, "libxc_LDA")
    functional = func_factory.construct(mol, torch.zeros((n, n)), torch.zeros(n), grid, ao)

    basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis([1, 6, 8])
    pos = mol.atom_coords()
    atomic_numbers = mol.atom_charges()
    sample = OFData.construct_new(
        basis_info, pos, atomic_numbers, torch.zeros((n,)), dual_basis_integrals="infer_from_basis"
    )
    energies, grad = functional(sample)

    assert grad.shape == (n,)
    assert isinstance(energies, Energies)
    assert energies["dummy"] == 42
    assert list(energies.energies_dict.keys()) == ["nuclear_repulsion", "libxc_LDA", "dummy"]

    with pytest.raises(ValueError):
        FunctionalFactory("hartree", model, "hartree")
