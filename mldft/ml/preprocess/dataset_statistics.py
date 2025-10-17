"""Module to compute and store statistics for the AtomRef and DimensionWiseRescaling modules."""
import os
import shutil
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr
from loguru import logger
from torch_geometric.utils import unbatch
from torch_scatter import scatter
from tqdm.auto import tqdm

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.atom_ref import AtomRef
from mldft.ml.models.components.sample_weighers import SampleWeigher
from mldft.utils.log_utils.config_in_tensorboard import dict_to_tree
from mldft.utils.rich_utils import rich_to_str


class DatasetStatistics:
    """Class to store statistics as needed by :class:`~mldft.ml.models.components.atom_ref.AtomRef`
    and :class:`~mldft.ml.models.components.dimension_wise_rescaling.DimensionWiseRescaling`. Also
    see :mod:`mldft.ml.compute_dataset_statistics` for how to compute them.

    The fields are computed per atom type and concatenated.
    They can be split using :meth:`~mldft.ml.data.components.basis_info.BasisInfo.split_field_by_atom_type`,
    with a corresponding :class:`~mldft.ml.data.components.basis_info.BasisInfo` object.
    """

    def __init__(self, path: str | Path, create_store: bool = False) -> None:
        """Initialize the dataset statistics.

        Args:
            path: Path to the .zarr file.
            create_store: Whether to create the store if it does not exist (If it exists, it will not be overwritten).
        """
        self.path = Path(path)
        assert str(self.path).endswith(
            ".zarr"
        ), f"File {path} must be a .zarr file, got {self.path} instead."
        if not self.path.exists():
            if not create_store:
                raise FileNotFoundError(f"File {self.path} does not exist.")
            else:
                # create the path and set permissions to be editable by the group
                self.path.mkdir(exist_ok=True)
                try:
                    self.path.parent.chmod(0o770)
                except PermissionError:
                    logger.warning(f"Could not set permissions for {self.path.parent}.")
                # self.path.touch()
                try:
                    self.path.chmod(0o770)
                except PermissionError:
                    logger.warning(f"Could not set permissions for {self.path}.")

    @staticmethod
    def from_dataset(
        path: str | Path, dataset: OFDataset | OFLoader, basis_info, **kwargs
    ) -> "DatasetStatistics":
        """Compute statistics from a dataset, given its loader.

        Args:
            path: Path to save the dataset statistics to.
            dataset: The dataset to compute the statistics from, or its loader. If the dataset is
                given directly, a loader with batch size 128 and 8 workers is used.
            basis_info: Basis info object for the dataset.
            **kwargs: Additional keyword arguments to pass to the :class:`DatasetStatisticsFitter`.
        """
        if isinstance(dataset, OFDataset):
            dataset = OFLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
        statistics_fitter = DatasetStatisticsFitter(basis_info=basis_info, **kwargs)
        return statistics_fitter.fit(path, dataset)

    def load_statistic(self, weigher_key: str, statistic_key: str) -> np.ndarray:
        """Get an attribute from the statistics.

        Args:
            weigher_key: The key of the weigher, e.g. 'has_energy_label'.
            statistic_key: The statistic to get, e.g. 'coeffs/mean'.

        Returns:
            The requested statistic as a numpy array
        """
        with zarr.DirectoryStore(str(self.path)) as store:
            root = zarr.open(store, mode="r")
            try:
                return root[f"{weigher_key}/{statistic_key}"][()]
            except KeyError:
                if weigher_key not in root:
                    raise KeyError(
                        f"Weigher {weigher_key} not found in the statistics. "
                        f"Available weighers: {list(root.keys())}"
                    )
                else:
                    raise KeyError(
                        f"Statistic {statistic_key} not found for weigher {weigher_key}. "
                        f"Available statistics: {list(root[weigher_key].keys())}"
                    )

    def save_statistic(
        self, weigher_key: str, statistic_key: str, data: np.ndarray, overwrite: bool = False
    ) -> None:
        """Save a statistic to the zarr store. If the statistic is already present, a check is
        made: If the data is the same, nothing is done. If the data is different, either a warning
        or an error is raised, depending on the value of the overwrite flag.

        Args:
            weigher_key: The key of the weigher, e.g. 'coeffs/mean'.
            statistic_key: The attribute to save, e.g. 'mean'.
            data: The data to save.
            overwrite: Whether to overwrite an existing statistic.
        """
        with zarr.DirectoryStore(str(self.path)) as store:
            root = zarr.open(store, mode="a")
            if f"{weigher_key}/{statistic_key}" in root:
                existing_data = root[f"{weigher_key}/{statistic_key}"][()]
                if not np.allclose(existing_data, data):
                    if overwrite:
                        logger.info(f"Overwriting {weigher_key}/{statistic_key} at {self.path}.")
                        root[f"{weigher_key}/{statistic_key}"] = data
                    else:
                        raise ValueError(
                            f"Data for {weigher_key}/{statistic_key} already exists and is different."
                        )
                else:
                    # data matches, nothing to do
                    return
            else:
                root.create_dataset(f"{weigher_key}/{statistic_key}", data=data)

            # add creation time and username as attributes
            with warnings.catch_warnings():
                warnings.filterwarnings(message=".*Duplicate name:.*", action="ignore")
                root[f"{weigher_key}/{statistic_key}"].attrs[
                    "created_at"
                ] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                root[f"{weigher_key}/{statistic_key}"].attrs["created_by"] = os.environ.get("USER")

    def keys_recursive(self):
        """Get all keys in the zarr store recursively."""

        def get_paths_recursive(group, path=None):
            for key in group.keys():
                if isinstance(group[key], zarr.Group):
                    yield from get_paths_recursive(
                        group[key], path=key if path is None else path + "/" + key
                    )
                else:
                    yield key if path is None else path + "/" + key

        with zarr.DirectoryStore(str(self.path)) as store:
            root = zarr.open(store, mode="r")
            return list(get_paths_recursive(root))

    def save_to(self, new_path: str | Path, overwrite: bool = False) -> None:
        """Save the statistics to a new path.

        If statistics already exist at the new path, they will be added using
        :meth:`save_statistic`.
        """
        new_path = Path(new_path)
        if not new_path.exists():
            # just copy the zarr directory to the new path
            shutil.copytree(self.path, new_path)
        else:
            new_statistics = DatasetStatistics(new_path)
            for key in self.keys_recursive():
                weigher_key, statistic_key = key.split("/", 1)
                data = self.load_statistic(weigher_key, statistic_key)
                new_statistics.save_statistic(
                    weigher_key, statistic_key, data, overwrite=overwrite
                )

    @staticmethod
    def field_summary_string(data: zarr.Array) -> str:
        if data.size == 1:
            return f"{data[()].item():.3g}"
        return f"shape={data.shape}, mean={data[:].mean():.3g}, std={data[:].std():.3g}, abs_max={np.abs(data[:]).max():.3g}"

    def __repr__(self):
        """Returns a string representation of the dataset statistics, with all keys and shapes."""

        def zarr_group_to_nested_summary_dict(group) -> dict:
            """Convert a zarr group to a nested dictionary with the shape of each dataset."""
            result = {}
            for key in group.keys():
                if isinstance(group[key], zarr.Group):
                    result[key] = zarr_group_to_nested_summary_dict(group[key])
                else:
                    result[key] = self.field_summary_string(group[key])
            return result

        with zarr.DirectoryStore(str(self.path)) as store:
            root = zarr.open(store, mode="r")
            shape_dict = zarr_group_to_nested_summary_dict(root)

        return rich_to_str(dict_to_tree(shape_dict, name="DatasetStatistics", guide_style="dim"))


def sample_weights_to_atom_weights(sample: OFData, sample_mask: torch.Tensor) -> torch.Tensor:
    """Convert a sample mask to an atom mask.

    Args:
        sample: The sample to get the atom mask from.
        sample_mask: The mask for the samples.

    Returns:
        atom_mask: The mask for the atoms corresponding to the samples.
    """
    return torch.repeat_interleave(sample_mask, sample.get_n_atom_per_molecule())


def sample_weights_to_basis_function_weights(
    sample: OFData, sample_mask: torch.Tensor
) -> torch.Tensor:
    """Convert a sample mask to a basis function mask.

    Args:
        sample: The sample to get the basis function mask from.
        sample_mask: The mask for the samples.

    Returns:
        basis_function_mask: The mask for the basis functions corresponding to the samples.
    """
    return sample_mask[sample.coeffs_batch]


class DatasetStatisticsFitter:
    """Computes / fits the statistics needed for AtomRef and DimensionwiseRescaling for a
    dataset."""

    def __init__(
        self,
        basis_info: BasisInfo,
        atom_ref_fit_sample_weighers: list[SampleWeigher],
        initial_guess_sample_weighers: list[SampleWeigher],
        with_bias: bool = True,
        min_atoms_per_type: int = 10,
        n_batches: int | None = None,
    ):
        """Initialize the DatasetStatisticsFitter.

        Args:
            basis_info: Basis info object for the dataset.
            atom_ref_fit_sample_weighers: Sample weighers to use for fitting the linear model for AtomRef.
            initial_guess_sample_weighers: Sample weighers to use for initial guess statistics.
            with_bias: Whether to include a T_global bias in the fit.
            min_atoms_per_type: Minimum number of atoms per atom type in the dataset.
            n_batches: Number of batches to use for fitting the statistics. If None, all batches are used.
                Useful for debugging.
        """
        self.basis_info = basis_info
        self.n_atom_types = basis_info.n_types
        self.n_batches = n_batches

        self.with_bias = with_bias
        assert min_atoms_per_type >= 0, "min_atoms_per_type must be positive or zero."
        self.min_atoms_per_type = min_atoms_per_type

        # specifies which weighers are to be used in the different modes

        # maps the summary string of the weigher (to be used as key in the statistics zarr file) to the weigher
        self.weigher_dict = dict()
        # specifies for which fields of the samples (or callables of the sample) the mean, std and abs-max
        # should be calculated.
        # strings (e.g. 'coeffs', 'gradient_label') are directly used as field names
        # for two-item tuples
        # (currently only ('initial_guess_delta', lambda sample: sample.coeffs - sample.ground_state_coeffs)),
        # the second element is a callable used to calculate the field from the sample, the first element is the
        # field name
        self.fields_per_weigher = defaultdict(list)
        self.fit_keys = set()
        self.initial_guess_keys = set()
        for weigher in atom_ref_fit_sample_weighers:
            key = weigher.summary_string()
            self.weigher_dict[key] = weigher
            self.fields_per_weigher[key].extend(["coeffs", "gradient_label"])
            self.fit_keys.add(key)
        for weigher in initial_guess_sample_weighers:
            key = weigher.summary_string()
            self.weigher_dict[key] = weigher
            self.fields_per_weigher[key].append(
                ("initial_guess_delta", lambda sample: sample.coeffs - sample.ground_state_coeffs)
            )
            self.initial_guess_keys.add(key)

        logger.info(
            "Using the following weighers and fields for statistics calculation:\n"
            + "\n".join(
                [f"{key:30s}: {fields}" for key, fields in self.fields_per_weigher.items()]
            )
        )

    @staticmethod
    def parse_field_name(field_info: str | tuple[str, callable]) -> str:
        """Parse the field info to get the field name.

        Args:
            field_info: The field info to parse. Can be the name of the attribute of the sample,
                or a tuple with the field name and a callable to get the field data from the sample.

        Returns:
            str: The field name.
        """
        if isinstance(field_info, str):
            field_name = field_info
        else:
            field_name, _ = field_info
        return field_name

    @staticmethod
    def parse_field_info(
        field_info: str | tuple[str, callable], sample: OFData
    ) -> tuple[str, torch.Tensor]:
        """Parse the field info to get the field name and data from the sample.

        Args:
            field_info: The field info to parse. Can be the name of the attribute of the sample,
                or a tuple with the field name and a callable to get the field data from the sample.
            sample: The sample to get the data from.

        Returns:
            tuple[str, torch.Tensor]: The field name and the data from the sample.
        """
        if isinstance(field_info, str):
            field_name = field_info
            field_data = getattr(sample, field_name)
        else:
            field_name, field_callable = field_info
            field_data = field_callable(sample)
        return field_name, field_data

    def fit(
        self,
        path: str | Path,
        loader: OFLoader,
        device: str | torch.DeviceObjType = "auto",
    ) -> DatasetStatistics:
        """Calculate dataset statistics from a data loader and fit a linear model to the target
        energies.

        Args:
            path: Path to save the dataset statistics to.
            loader: The data loader to fit the statistics to.
            device: Device to use for the statistics. If "auto", cuda is used if available.

        Returns:
            DatasetStatistics: The fitted dataset statistics.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = torch.get_default_dtype()
        sample_count = {key: 0 for key in self.weigher_dict.keys()}
        atom_weighted_count_per_type = {
            key: torch.zeros(self.n_atom_types, device=device) for key in self.weigher_dict.keys()
        }
        cumulative_fields = {
            key: {
                self.parse_field_name(field_info): torch.zeros(
                    self.basis_info.n_basis, device=device
                )
                for field_info in field_infos
            }
            for key, field_infos in self.fields_per_weigher.items()
        }
        max_abs_fields = {
            key: {
                self.parse_field_name(field_info): torch.zeros(
                    self.basis_info.n_basis, device=device
                )
                for field_info in field_infos
            }
            for key, field_infos in self.fields_per_weigher.items()
        }

        mol_ids = []
        for i, sample in enumerate(
            tqdm(loader, desc="calculating statistics: averaging over atoms")
        ):
            if self.n_batches is not None and i >= self.n_batches:
                break
            assert isinstance(sample, OFData)
            sample = sample.to(device)
            mol_ids.extend(sample.mol_id)

            for key, sample_weigher in self.weigher_dict.items():
                per_sample_weights = sample_weigher.get_weights(sample).to(dtype)
                per_atom_weights = sample_weights_to_atom_weights(sample, per_sample_weights)
                per_basis_function_weights = sample_weights_to_basis_function_weights(
                    sample, per_sample_weights
                )

                sample_count[key] += per_sample_weights.sum()

                atom_weighted_count_per_type[key] += torch.bincount(
                    sample.atom_ind, weights=per_atom_weights, minlength=self.n_atom_types
                )

                for field_info in self.fields_per_weigher[key]:
                    field_name, field_data = self.parse_field_info(field_info, sample)
                    scatter(
                        src=field_data * per_basis_function_weights,
                        index=sample.basis_function_ind,
                        out=cumulative_fields[key][field_name],
                        dim=0,
                        reduce="sum",
                    )
                    scatter(
                        src=torch.abs(field_data) * per_basis_function_weights,
                        index=sample.basis_function_ind,
                        out=max_abs_fields[key][field_name],
                        dim=0,
                        reduce="max",
                    )

        # remove duplicates to get unique mol_ids
        mol_ids = np.unique(mol_ids)
        mol_id_to_ind = {mol_id: ind for ind, mol_id in enumerate(mol_ids)}

        for key in atom_weighted_count_per_type.keys():
            assert torch.all(atom_weighted_count_per_type[key].ge(self.min_atoms_per_type)), (
                f"Too little weight found for some atom types. "
                f"{atom_weighted_count_per_type[key]=}, {self.min_atoms_per_type=}"
            )
            atom_weighted_count_per_type[key] = torch.clamp(
                atom_weighted_count_per_type[key], min=1
            )

        atom_weighted_count_per_basis_func = {
            key: torch.repeat_interleave(
                atom_weighted_count_per_type[key],
                torch.as_tensor(self.basis_info.basis_dim_per_atom, device=device),
            )
            for key in self.weigher_dict.keys()
        }

        mean_fields = {
            key: {
                field_name: field_data / atom_weighted_count_per_basis_func[key]
                for field_name, field_data in field_dict.items()
            }
            for key, field_dict in cumulative_fields.items()
        }

        # Construct from the dataset:
        # energy_targets: a list of all kinetic energies in the dataset, minus the estimate from the linear model
        #   based on the mean_gradient_per_type. shape (#data_points)
        # composition_matrix: a matrix of shape (#data_points, #atomic_numbers) where row i encodes the
        #   composition of data point i.
        # for example, if the atom types are (H, C, O), The row for a water molecule would be [2, 0, 1]

        cumulative_energy_target = {
            key: torch.zeros(len(mol_ids), device=device) for key in self.fit_keys
        }
        # for the equivariant AtomRef using only the l==0 features
        cumulative_scalar_energy_target = {
            key: torch.zeros(len(mol_ids), device=device) for key in self.fit_keys
        }
        composition_dict = {}

        # weight used to average the total energy per mol_id over different scf iterations
        mol_id_total_energy_weight = {
            key: torch.zeros(len(mol_ids), device=device) for key in self.fit_keys
        }

        max_abs_gradient_after_atom_ref = {
            key: torch.zeros_like(mean_fields[key]["coeffs"]) for key in self.fit_keys
        }
        mean_gradient_after_atom_ref = {
            key: torch.zeros_like(mean_fields[key]["coeffs"]) for key in self.fit_keys
        }

        field_variances = {
            key: {
                field_name: torch.zeros_like(field_data)
                for field_name, field_data in field_dict.items()
            }
            for key, field_dict in cumulative_fields.items()
        }

        scalar_mask = (torch.as_tensor(self.basis_info.l_per_basis_func, device=device) == 0).to(
            dtype
        )

        for i, sample in enumerate(
            tqdm(loader, desc="calculating statistics: creating linear regression targets")
        ):
            if self.n_batches is not None and i >= self.n_batches:
                break
            assert isinstance(sample, OFData)  # to make pycharm happy
            sample = sample.to(device)

            # sample_fit_weight = self.fit_sample_weigher(sample)
            # basis_function_label_mask = sample_weights_to_basis_function_weights(sample, sample_fit_weight)

            mol_inds = torch.as_tensor([mol_id_to_ind[i] for i in sample.mol_id], device=device)

            # get the composition for each mol_ind
            atom_ind_per_mol = unbatch(sample.atom_ind, sample.batch)
            for i, mol_ind in enumerate(mol_inds):
                if mol_ind not in composition_dict:
                    composition_dict[mol_ind.item()] = torch.bincount(
                        atom_ind_per_mol[i], minlength=self.n_atom_types
                    )

            for key, sample_weigher in self.weigher_dict.items():
                per_sample_weights = sample_weigher.get_weights(sample).to(dtype)
                per_basis_function_weights = sample_weights_to_basis_function_weights(
                    sample, per_sample_weights
                )

                # for standard deviation computation for all the fields
                for field_info in self.fields_per_weigher[key]:
                    field_name, field_data = self.parse_field_info(field_info, sample)
                    scatter(
                        src=(
                            (field_data - mean_fields[key][field_name][sample.basis_function_ind])
                            ** 2
                        )
                        * per_basis_function_weights,
                        index=sample.basis_function_ind,
                        out=field_variances[key][field_name],
                        dim=0,
                        reduce="sum",
                    )

                # extra things: energy targets, max abs gradient after AtomRef, mean gradient after AtomRef

                # energy targets
                if key in self.fit_keys:
                    # add the energy of each sample to the target for the molecule
                    for out in (
                        cumulative_energy_target[key],
                        cumulative_scalar_energy_target[key],
                    ):
                        scatter(
                            src=sample.energy_label * per_sample_weights,
                            index=mol_inds,
                            out=out,
                            dim=0,
                            reduce="sum",
                        )
                    # subtract the energy as it will be predicted by the linear model from the target
                    scatter(
                        src=-mean_fields[key]["gradient_label"][sample.basis_function_ind]
                        * sample.coeffs
                        * per_basis_function_weights,
                        index=mol_inds[sample.coeffs_batch],
                        out=cumulative_energy_target[key],
                        dim=0,
                        reduce="sum",
                    )
                    # same for the equivariant case, where only the l=0 gradients will be used
                    scatter(
                        src=-mean_fields[key]["gradient_label"][sample.basis_function_ind]
                        * scalar_mask[sample.basis_function_ind]
                        * sample.coeffs
                        * per_basis_function_weights,
                        index=mol_inds[sample.coeffs_batch],
                        out=cumulative_scalar_energy_target[key],
                        dim=0,
                        reduce="sum",
                    )

                    # weight used to average the total energy per mol_id over different scf iterations
                    scatter(
                        src=per_sample_weights,
                        index=mol_inds,
                        out=mol_id_total_energy_weight[key],
                        dim=0,
                        reduce="sum",
                    )

                    # max abs gradient after AtomRef
                    scatter(
                        src=torch.abs(
                            sample.gradient_label
                            - mean_fields[key]["gradient_label"][sample.basis_function_ind]
                        )
                        * per_basis_function_weights,
                        index=sample.basis_function_ind,
                        out=max_abs_gradient_after_atom_ref[key],
                        dim=0,
                        reduce="max",
                    )
                    # mean gradient after AtomRef (should be zero)
                    scatter(
                        src=(
                            sample.gradient_label
                            - mean_fields[key]["gradient_label"][sample.basis_function_ind]
                        )
                        * per_basis_function_weights,
                        index=sample.basis_function_ind,
                        out=mean_gradient_after_atom_ref[key],
                        dim=0,
                        reduce="sum",
                    )

        mean_gradient_after_atom_ref = {
            key: value / atom_weighted_count_per_basis_func[key]
            for key, value in mean_gradient_after_atom_ref.items()
        }

        # check if the mean gradient after AtomRef is zero, if not, print a warning or raise an error
        for key in self.fit_keys:
            if not torch.allclose(
                mean_gradient_after_atom_ref[key],
                torch.zeros_like(mean_gradient_after_atom_ref[key]),
            ):
                assert torch.allclose(
                    mean_gradient_after_atom_ref[key],
                    torch.zeros_like(mean_gradient_after_atom_ref[key]),
                    atol=1e-4,
                ), f"Mean gradient after AtomRef is not zero for weigher {key}: {mean_gradient_after_atom_ref=}"

                logger.warning(
                    f"Mean gradient after AtomRef is not zero for weigher {key}: {mean_gradient_after_atom_ref=}"
                )

        # the l>0 fields are not used in the scalar AtomRef, so the corresponding gradients are not impacted
        max_abs_gradient_after_scalar_atom_ref = {
            key: torch.where(
                scalar_mask.bool(),
                max_abs_gradient_after_atom_ref[key],
                max_abs_fields[key]["gradient_label"],
            )
            for key in self.fit_keys
        }

        field_variances = {
            key: {
                field_name: field_variance / atom_weighted_count_per_basis_func[key]
                for field_name, field_variance in field_dict.items()
            }
            for key, field_dict in field_variances.items()
        }

        field_stds = {
            key: {
                field_name: torch.sqrt(field_variance)
                for field_name, field_variance in field_dict.items()
            }
            for key, field_dict in field_variances.items()
        }

        composition_matrix = torch.stack([composition_dict[i] for i in range(len(mol_ids))], dim=0)

        energy_targets = {
            key: (cumulative_energy_target[key] / mol_id_total_energy_weight[key]).cpu().numpy()
            for key in self.fit_keys
        }
        scalar_energy_targets = {
            key: (cumulative_scalar_energy_target[key] / mol_id_total_energy_weight[key])
            .cpu()
            .numpy()
            for key in self.fit_keys
        }

        composition_matrix = composition_matrix.cpu().numpy()

        global_bias = defaultdict(dict)
        per_atom_type_bias = defaultdict(dict)

        for key in self.fit_keys:
            mol_mask = (mol_id_total_energy_weight[key] > 0).cpu()
            if not mol_mask.any():
                raise ValueError(
                    f"No data points found for weigher {key}. "
                    f"Check the sample weigher or the dataset."
                )
            mol_mask = mol_mask.numpy().astype(bool)

            # fit such that energy_targets ~= composition_matrix @ per_atom_type_bias + global_bias
            for energy_target, name in (
                (energy_targets[key], "all l"),
                (scalar_energy_targets[key], "l=0"),
            ):
                if self.with_bias:
                    x = np.column_stack((composition_matrix, np.ones(len(energy_target))))[
                        mol_mask
                    ]
                    fit_params = np.linalg.lstsq(x, energy_target[mol_mask], rcond=None)[0]

                    per_atom_type_bias[name][key] = fit_params[:-1]
                    global_bias[name][key] = fit_params[-1]
                else:
                    x = composition_matrix[mol_mask]
                    fit_params = np.linalg.lstsq(x, energy_target[mol_mask], rcond=None)[0]

                    per_atom_type_bias[name][key] = fit_params
                    global_bias[name][key] = 0.0

        dataset_statistics_dict = dict(
            mean=mean_fields,
            std=field_stds,
            abs_max=max_abs_fields,
            gradient_max_after_atom_ref=max_abs_gradient_after_atom_ref,
            atom_ref_atom_type_bias=per_atom_type_bias["all l"],
            atom_ref_global_bias=global_bias["all l"],
            gradient_max_after_scalar_atom_ref=max_abs_gradient_after_scalar_atom_ref,
            scalar_atom_ref_atom_type_bias=per_atom_type_bias["l=0"],
            scalar_atom_ref_global_bias=global_bias["l=0"],
        )

        def to_numpy(obj):
            """Convert all torch tensors in the object to numpy arrays."""
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy()
            if isinstance(obj, dict):
                return {key: to_numpy(value) for key, value in obj.items()}
            return obj

        def add_statistics_dict(statistics_dict):
            statistics_dict = to_numpy(statistics_dict)
            for statistics_name, statistics_per_weigher in statistics_dict.items():
                for weigher_key, statistics in statistics_per_weigher.items():
                    if isinstance(statistics, dict):
                        for field_name, field_data in statistics.items():
                            dataset_statistics.save_statistic(
                                weigher_key,
                                f"{field_name}/{statistics_name}",
                                field_data,
                            )
                    else:
                        dataset_statistics.save_statistic(weigher_key, statistics_name, statistics)

        logger.info(f"Saving dataset statistics to {path}")
        dataset_statistics = DatasetStatistics(path, create_store=True)

        add_statistics_dict(dataset_statistics_dict)

        # compute statistics on the energy difference after AtomRef

        atom_refs = {
            key: AtomRef.from_dataset_statistics(dataset_statistics, weigher_key=key).to(device)
            for key in self.fit_keys
        }
        scalar_atom_refs = {
            key: AtomRef.from_dataset_statistics(
                dataset_statistics, basis_info=self.basis_info, weigher_key=key, scalar_only=True
            ).to(device)
            for key in self.fit_keys
        }

        atom_ref_energy_diff_mean = {
            key: dict(energy_minus_atom_ref=0.0, energy_minus_scalar_atom_ref=0.0)
            for key in self.fit_keys
        }
        atom_ref_energy_diff_std = {
            key: dict(energy_minus_atom_ref=0.0, energy_minus_scalar_atom_ref=0.0)
            for key in self.fit_keys
        }
        atom_ref_energy_diff_abs_max = {
            key: dict(energy_minus_atom_ref=0.0, energy_minus_scalar_atom_ref=0.0)
            for key in self.fit_keys
        }

        for i, sample in enumerate(
            tqdm(loader, desc="calculating statistics: energies after AtomRef")
        ):
            if self.n_batches is not None and i >= self.n_batches:
                break
            assert isinstance(sample, OFData)  # to make pycharm happy
            sample = sample.to(device)
            for key in self.fit_keys:
                per_sample_weights = self.weigher_dict[key].get_weights(sample).to(dtype)
                for atom_ref, result_name in (
                    (atom_refs[key], "energy_minus_atom_ref"),
                    (scalar_atom_refs[key], "energy_minus_scalar_atom_ref"),
                ):
                    atom_ref_energy = atom_ref.sample_forward(sample)
                    atom_ref_energy_diff_std[key][result_name] += torch.dot(
                        (sample.energy_label - atom_ref_energy) ** 2, per_sample_weights
                    )
                    atom_ref_energy_diff_mean[key][result_name] += torch.dot(
                        sample.energy_label - atom_ref_energy, per_sample_weights
                    )
                    atom_ref_energy_diff_abs_max[key][result_name] = max(
                        atom_ref_energy_diff_abs_max[key][result_name],
                        float(
                            torch.max(
                                torch.abs(sample.energy_label - atom_ref_energy)
                                * per_sample_weights
                            ).item()
                        ),
                    )

        atom_ref_energy_diff_mean = {
            key: {result_name: result / sample_count[key] for result_name, result in value.items()}
            for key, value in atom_ref_energy_diff_mean.items()
        }
        atom_ref_energy_diff_std = {
            key: {
                result_name: (result / sample_count[key]).sqrt()
                for result_name, result in value.items()
            }
            for key, value in atom_ref_energy_diff_std.items()
        }

        energy_after_atom_ref_dataset_statistics_dict = dict(
            mean=atom_ref_energy_diff_mean,
            std=atom_ref_energy_diff_std,
            abs_max=atom_ref_energy_diff_abs_max,
        )

        logger.info(f"Saving energy_diff after AtomRef statistics to {path}")
        add_statistics_dict(energy_after_atom_ref_dataset_statistics_dict)

        return dataset_statistics
