import numpy as np
import torch
from torch.nn import Embedding

from mldft.ml.models.components.density_coeff_embedding import AtomHotEmbedding
from mldft.ml.models.components.node_embedding import NodeEmbedding
from mldft.ml.models.components.shrink_gate_module import PerBasisFuncShrinkGateModule

torch.manual_seed(42)

N_ATOMS = 85  # if not otherwise specified


def create_random_data(dummy_sample, n_edges=67, distance_channels=32):
    """Creating random data consisting of atomic numbers, edge features and index, and density
    coefficients.

    Args:
        n_edges (int): Number of edges in the molecule.
        distance_channels (int): Number of distance (edge_feature) channels.

    Returns:
        atomic_numbers (Tensor[int]): Tensor containing the atomic types Z for each atom.
        density_coeffs (list[Tensor]): List of tensors, each element containing the coefficients
            for one atom. These coefficients can be of arbitrary length.
        edge_attributes (Tensor[float]): Tensor containing the edge features (distances).
        edge_index (Tensor[int]): Tensor containing the edge indices.
    """

    atom_ind = dummy_sample["atom_ind"]
    density_coeffs = dummy_sample["coeffs"]

    n_atoms = atom_ind.shape[0]

    assert n_edges <= n_atoms * (n_atoms - 1) / 2, "Too many edges for the given number of atoms"
    edge_index = torch.randint(0, n_atoms, (2, n_edges))
    edge_attributes = torch.rand((n_edges, distance_channels))

    return atom_ind, density_coeffs, edge_attributes, edge_index


def node_embedding_setup(
    dummy_basis_info,
    out_channels=20,
    dst_in_channels=32,
    p_hidden_channels=32,
    dst_hidden_channels=32,
    p_num_layers=3,
    dst_num_layers=3,
    p_activation=torch.nn.SiLU,
    dst_activation=torch.nn.SiLU,
    lambda_mul=0.5,
    lambda_co=1.5,
):
    """Example setup for the node_embedding tests."""

    torch.manual_seed(42)
    model = NodeEmbedding(
        n_atoms=dummy_basis_info.n_types,
        basis_dim_per_atom=dummy_basis_info.basis_dim_per_atom,
        basis_atomic_numbers=dummy_basis_info.atomic_numbers,
        atomic_number_to_atom_index=dummy_basis_info.atomic_number_to_atom_index,
        out_channels=out_channels,
        dst_in_channels=dst_in_channels,
        p_hidden_channels=p_hidden_channels,
        dst_hidden_channels=dst_hidden_channels,
        p_num_layers=p_num_layers,
        dst_num_layers=dst_num_layers,
        p_activation=p_activation,
        dst_activation=dst_activation,
        lambda_mul=lambda_mul,
        lambda_co=lambda_co,
        use_per_basis_func_shrink_gate=True,
    )

    return model


def test_initialization(dummy_basis_info):
    """Test the initialization of the NodeEmbedding module."""

    # Initialize with non-default parameters
    node_embedding = node_embedding_setup(
        dummy_basis_info,
        out_channels=23,
        dst_in_channels=36,
        p_hidden_channels=35,
        dst_hidden_channels=37,
        p_num_layers=5,
        dst_num_layers=7,
        p_activation=torch.nn.ReLU,
        dst_activation=torch.nn.GELU,
        lambda_mul=0.2,
        lambda_co=0.8,
    )

    # Assertions to check if components are initialized correctly
    assert isinstance(
        node_embedding.z_embed, Embedding
    ), "Embedding layer not initialized correctly"
    assert isinstance(
        node_embedding.shrink_gate, PerBasisFuncShrinkGateModule
    ), "ShrinkGateModule not initialized correctly"


def test_reset_parameters(dummy_basis_info):
    """Test the parameter reset of the NodeEmbedding module."""

    # Initialize the NodeEmbedding module
    node_embedding = node_embedding_setup(dummy_basis_info)

    # Save the initial state of the parameters
    initial_state = {name: param.clone() for name, param in node_embedding.named_parameters()}

    # Call reset_parameters
    node_embedding.reset_parameters()

    re_initial_state = {name: param.clone() for name, param in node_embedding.named_parameters()}
    for name, param in re_initial_state.items():
        # Constant parameters should not change
        if name == "shrink_gate.inner_factor" or name == "shrink_gate.outer_factor":
            assert torch.equal(
                initial_state[name], param
            ), f"Parameter {name} did not reset correctly"
        # All other parameters should change randomly
        else:
            assert not torch.equal(
                initial_state[name], param
            ), f"Parameter {name} did not reset correctly"


def test_aggregate_distances(dummy_basis_info, dummy_sample):
    """Test the aggregation of distances (edge features) for each atom over connected atoms."""
    # Initialize NodeEmbedding module
    n_atoms = dummy_sample["atom_ind"].shape[0]
    n_edges = (
        torch.randint(low=n_atoms // 4, high=n_atoms, size=(1,)).squeeze().item()
    )  # Example number of edges
    dst_in_channels = 16  # Example number of distance input channels

    node_embedding = node_embedding_setup(dummy_basis_info, n_atoms, dst_in_channels)
    _, _, edge_attributes, edge_index = create_random_data(dummy_sample, n_edges, dst_in_channels)

    # Create small test data for edge_attributes and edge_index
    num_edges = 10  # Example number of edges
    edge_attributes = torch.rand((num_edges, dst_in_channels))  # Random edge features
    edge_index = torch.randint(0, n_atoms, (2, num_edges))  # Random edge indices

    # Call aggregate_distances
    aggregated_features = node_embedding.aggregate_distances(edge_attributes, edge_index, n_atoms)

    # Assertions
    # Check the shape of the output
    assert aggregated_features.shape == (
        n_atoms,
        dst_in_channels,
    ), "Output tensor has incorrect shape"

    # Manually perform the aggregation for comparison
    manual_aggregation = torch.zeros((n_atoms, dst_in_channels))
    for edge, attr in zip(edge_index.t(), edge_attributes):
        source_node = edge[0]
        manual_aggregation[source_node] += attr

    # Compare the manually aggregated features with those from the function
    assert torch.allclose(
        aggregated_features, manual_aggregation
    ), "Aggregated features do not match expected values"


def test_forward(dummy_basis_info, dummy_sample):
    """Test the forward pass of the NodeEmbedding module with known hyperparameters."""

    n_basis_per_atom = torch.from_numpy(dummy_sample["n_basis_per_atom"])
    n_edges = 12
    out_channels = 40
    dst_in_channels = 16
    node_embedding = node_embedding_setup(
        dummy_basis_info=dummy_basis_info,
        out_channels=out_channels,
        dst_in_channels=dst_in_channels,
    )

    # Create test input data
    atom_ind, density_coeffs, distances, edge_index = create_random_data(
        dummy_sample, n_edges, dst_in_channels
    )

    n_atoms = atom_ind.shape[0]
    basis_function_ind = torch.from_numpy(dummy_sample.basis_function_ind)

    # Call the forward method
    output = node_embedding.forward(
        density_coeffs,
        atom_ind,
        basis_function_ind,
        n_basis_per_atom,
        torch.asarray(dummy_sample.coeff_ind_to_node_ind),
        distances,
        edge_index,
    )

    # Assertions
    # Check the shape of the output
    assert output.shape == (n_atoms, out_channels), "Output tensor has incorrect shape"


def test_coeff_embedding(dummy_basis_info, dummy_sample):
    """Test the density coefficient embedding."""

    atom_ind = torch.from_numpy(dummy_sample["atom_ind"]).int()
    density_coeffs = torch.from_numpy(dummy_sample["coeffs"])
    n_atoms = atom_ind.shape[0]
    n_basis_per_atom = torch.from_numpy(dummy_sample["n_basis_per_atom"]).int()
    basis_function_ind = torch.from_numpy(dummy_sample.basis_function_ind)

    atom_hot_embedding = AtomHotEmbedding(embed_dim=dummy_basis_info.n_basis)

    # Call the forward method (atom_hot_embed)

    embedded_coeffs = atom_hot_embedding(
        density_coeffs,
        basis_function_ind,
        n_basis_per_atom,
        coeff_ind_to_node_ind=torch.asarray(dummy_sample.coeff_ind_to_node_ind),
    )

    # Check the shape of the output
    assert embedded_coeffs.shape[0] == n_atoms, "Embedded coeffs do not correspond to n_atoms"
    assert embedded_coeffs.shape[1] == sum(
        dummy_basis_info.basis_dim_per_atom
    ), "Embedded coeffs do not correspond to concatenated basis dimensions"


def test_vectorized_embedding(dummy_basis_info, dummy_sample_torch):
    """Test the vectorized embedding."""

    embed_dim = dummy_basis_info.n_basis
    atom_ind = dummy_sample_torch.atom_ind
    basis_dim_per_atom = dummy_basis_info.basis_dim_per_atom
    atom_ptr = np.zeros(len(basis_dim_per_atom), dtype=np.int32)
    atom_ptr[1:] = np.cumsum(basis_dim_per_atom)[:-1]

    atom_hot_embedding = AtomHotEmbedding(embed_dim=dummy_basis_info.n_basis)

    # Call the vectorized forward method (atom_hot_embed)
    vec_coeffs = atom_hot_embedding(
        dummy_sample_torch.coeffs,
        dummy_sample_torch.basis_function_ind,
        dummy_sample_torch.n_basis_per_atom,
        coeff_ind_to_node_ind=dummy_sample_torch.coeff_ind_to_node_ind,
    )

    # Make non-vectorized embedding
    coeffs_list = torch.split(
        dummy_sample_torch.coeffs, dummy_sample_torch.n_basis_per_atom.tolist(), dim=0
    )
    non_vec_coeffs = dummy_sample_torch.coeffs.new_zeros(size=(atom_ind.shape[0], int(embed_dim)))

    for i, index in enumerate(atom_ind):
        non_vec_coeffs[
            i, atom_ptr[index] : (atom_ptr[index] + basis_dim_per_atom[index])
        ] = coeffs_list[i]

    # Check if non-vectorized and vectorized embedding are equal
    assert torch.allclose(
        vec_coeffs, non_vec_coeffs
    ), "Vectorized and non-vectorized embedding are not equal"
