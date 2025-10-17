import numpy as np
import pytest
import torch

from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.initial_guess_delta_module import (
    InitialGuessDeltaModule,
)
from mldft.utils.utils import set_default_torch_dtype


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("batch_size", [1, 32])
def test_initial_guess_delta_module(dummy_basis_info, dummy_dataset_torch, batch_size):
    """Tests the input / output shapes of the InitialGuessDeltaModule by emulating the atom hot
    encoding."""

    # Get a batch of data
    dataloader = OFLoader(dummy_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    # the length of the atom hot encoded vector is the sum of the basis dimensions per atom type
    vector_length = np.sum(dummy_basis_info.basis_dim_per_atom)

    # emulate atom hot encoding
    coeffs_list = list(batch.split_field_by_atom(batch.coeffs))
    tmp = []
    for coeffs in coeffs_list:
        tmp.append(torch.nn.functional.pad(coeffs, (0, vector_length - coeffs.shape[0]), value=0))

    assert (
        tmp[0].shape[0] == vector_length
    ), "The length of the atom hot encoded vector is not correct!"

    coeffs = torch.stack(tmp)

    assert (
        coeffs.shape[0] == batch.pos.shape[0]
    ), "The number of atoms in the batch is not correct!"

    module = InitialGuessDeltaModule(vector_length, dummy_basis_info, [64, 64])

    out = module.forward(coeffs, batch.basis_function_ind, batch.coeff_ind_to_node_ind)

    assert out.shape[0] == batch.coeffs.shape[0], "The output shape of the module is not correct!"
