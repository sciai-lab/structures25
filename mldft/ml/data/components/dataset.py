"""Dataset class for machine learning."""

from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import zarr
from omegaconf import ListConfig
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import MasterTransformation
from mldft.ml.data.components.of_data import OFData
from mldft.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class OFDataset(Dataset):
    """Dataset of OF-DFT data.

    Each scf iteration of each molecule is a sample.
    """

    data_class = OFData

    def __init__(
        self,
        paths: List[Path],
        basis_info: BasisInfo,
        transforms: MasterTransformation = None,
        limit_scf_iterations: int | list[int] | None = None,
        keep_initial_guess: bool = True,
        num_scf_iterations_per_path: List[int] | None = None,
        cache_in_memory: bool = False,
        **of_data_kwargs,
    ) -> None:
        """
        Args:
            paths: List of paths to .zarr files.
            basis_info: Basis information for all samples.
            transforms: List of transforms to apply to each sample.
            limit_scf_iterations: Which scf iterations to use. When passing an int s>0 we use all scf iterations larger
                or equal to s. If s<0 we use the s last iterations and when passing a list of ints, we use only the scf
                iterations in the list.
            keep_initial_guess: Whether to keep the initial guess in the dataset when filtering
                with limit_scf_iterations. This is useful if you want to train the energy model on the later scf
                iterations but also want to train the initial guess model on the minao label. Setting this to False
                will not remove the initial guess, limit_scf_iterations=1 and keep_initial_guess=False will remove them.
            num_scf_iterations_per_path: List of number of scf iterations per path. If None, the number of scf
                iterations is read from the paths directly which can take some time.
            of_data_kwargs: Keyword arguments passed to :meth:`OFData.from_file`.
            cache_in_memory: Whether to cache the dataset in memory. This is useful if the dataset
                fits into memory and the dataset is used multiple times, especially if expensive transforms
                are used. Warning: Do not enable if non-deterministic transforms are used.
        """
        super().__init__()
        # For now, always use all scf iterations.
        # Later, we can add a parameter to use only a subset, e.g. some slice
        assert isinstance(
            basis_info, BasisInfo
        ), f"basis_info must be of type BasisInfo, but is {type(basis_info)}"
        assert len(paths) > 0, "Paths must not be empty"
        self.paths = paths
        self.basis_info = basis_info
        self.of_data_kwargs = of_data_kwargs
        self.transforms = transforms

        # Find out how many samples (scf iterations) there are in total, and construct a list of indices that maps a
        # path index to the first sample index belonging to that path.
        if num_scf_iterations_per_path is None:
            num_scf_iterations_per_path = []
            for path in tqdm(
                self.paths, desc="Creating Dataset, counting scf iterations per path"
            ):
                # This works for zip or file system storage types
                with zarr.open(path, mode="r") as root:
                    if "n_scf_steps" in root["of_labels"]:
                        num_scf_iterations_per_path.append(root["of_labels/n_scf_steps"][()])
                    else:
                        raise KeyError(f" Key n_scf_steps not found in {path}")
        self.limit_scf_iterations = limit_scf_iterations
        self.scf_iterations_per_path = self.configure_scf_iterations(
            num_scf_iterations_per_path, keep_initial_guess
        )
        num_scf_iterations_per_path = [len(scfs) for scfs in self.scf_iterations_per_path]
        self.path_indices = np.cumsum([0] + num_scf_iterations_per_path)
        self.n_samples = int(self.path_indices[-1])

        self.cache_in_memory = cache_in_memory
        if cache_in_memory:
            self.getitem = lru_cache(maxsize=None)(self.getitem)

    def configure_scf_iterations(
        self, num_scf_iterations_per_path: list[int], keep_initial_guess: bool = True
    ) -> list[np.ndarray]:
        """Determine for each path how many scf iterations are there, depending on scf_iterations.

        If scf_iterations>0 we use all scf iterations larger or equal to s, this could delete some geometries but only for large numbers.
        For larger numbers rather use the negative integer functionality. If scf_iterations<0 we use the s last iterations.
        When passing a list of ints, we use only the scf iterations in the list. This could also delete some geometries.

        Args:
            num_scf_iterations_per_path: List of number of scf iterations per path in the whole non-filtered dataset.
            keep_initial_guess: Whether to keep the initial guess in the dataset when filtering. Setting this to False
                will not remove the initial guess, limit_scf_iterations=1 and keep_initial_guess=False will remove them.

        Returns:
            scf_iterations_per_path: List of arrays containing the indices of the scf iterations to use for each path.
        """
        if self.limit_scf_iterations is None:
            if not keep_initial_guess:
                log.warning(
                    "keep_initial_guess is False but limit_scf_iterations is None, every label will be kept."
                )
            return [np.arange(scf_iterations) for scf_iterations in num_scf_iterations_per_path]
        scf_iterations_per_path = []
        if isinstance(self.limit_scf_iterations, int):
            # positive integer case
            if self.limit_scf_iterations >= 0:
                i = 0
                while i < len(self.paths):
                    # If keep initial guess we append an empty array and add 0 later
                    if (
                        num_scf_iterations_per_path[i] < self.limit_scf_iterations
                        and not keep_initial_guess
                    ):
                        num_scf_iterations_per_path.pop(i)
                        log.warning(
                            f"Path {self.paths[i]} has less than {self.limit_scf_iterations} scf iterations. Removing it from dataset."
                        )
                        self.paths.pop(i)
                    else:
                        scf_iterations_per_path.append(
                            np.arange(self.limit_scf_iterations, num_scf_iterations_per_path[i])
                        )
                        i += 1
            else:
                # negative integer case
                for num_scf_iterations in num_scf_iterations_per_path:
                    scf_iterations_per_path.append(
                        np.arange(
                            max(0, num_scf_iterations + self.limit_scf_iterations),
                            num_scf_iterations,
                        )
                    )
        elif isinstance(self.limit_scf_iterations, (list, ListConfig)):
            # list of integers case
            scf_set = set(self.limit_scf_iterations)
            i = 0
            while i < len(self.paths):
                existing_scf_iterations = set(range(num_scf_iterations_per_path[i]))
                negative_existing_scf_iterations = set(
                    range(-1, -num_scf_iterations_per_path[i] - 1, -1)
                )
                existing_scf_iterations.update(negative_existing_scf_iterations)
                used_scf_iterations = existing_scf_iterations.intersection(scf_set)
                # If keep initial guess we append an empty array and add 0 later
                if len(used_scf_iterations) == 0 and not keep_initial_guess:
                    num_scf_iterations_per_path.pop(i)
                    log.warning(
                        f"Path {self.paths[i]} has no scf iterations in {self.limit_scf_iterations}. Removing it from dataset."
                    )
                    self.paths.pop(i)
                else:
                    scf_iterations_per_path.append(np.fromiter(used_scf_iterations, int))
                    i += 1
        else:
            raise ValueError("scf_iterations must be int, list or None.")
        if keep_initial_guess:
            # add zero, save way to do it correctly
            for i in range(len(scf_iterations_per_path)):
                if 0 not in scf_iterations_per_path[i]:
                    scf_iterations_per_path[i] = np.insert(scf_iterations_per_path[i], 0, 0)
        return scf_iterations_per_path

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs) -> "OFDataset":
        """Initialize the dataset from a directory containing .zarr files.

        Args:
            directory: Path to the directory.
            **kwargs: Keyword arguments passed to the constructor.
        """
        directory = Path(directory)
        paths = sorted(directory.glob("*.zarr*"))
        if len(paths) == 0:
            raise FileNotFoundError(f"No .zarr files found in {directory}")
        return cls(paths=paths, **kwargs)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, item: int) -> OFData:
        """Returns the sample at the given index.

        Args:
            item: Index of the sample.
        """
        return self.getitem(item)

    def getitem(self, item: int) -> OFData:
        """Returns the sample at the given index.

        Note: This method is cached if cache_in_memory is True. As :meth:`__getitem__` cannot be overwritten at
        runtime, this method is used instead.

        Args:
            item: Index of the sample.

        Returns:
            sample: :class:`~mldft.ml.data.components.of_data.OFData` object.
        """
        geometry_idx = np.searchsorted(self.path_indices, item, side="right") - 1
        path = self.paths[geometry_idx]
        # numpy ints somehow don't batch correctly so we use python ints
        scf_iteration = int(
            self.scf_iterations_per_path[geometry_idx][item - self.path_indices[geometry_idx]]
        )

        sample = self.data_class.from_file(
            path=path,
            scf_iteration=scf_iteration,
            basis_info=self.basis_info,
            **self.of_data_kwargs,
        )

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
