"""Arbitrary dataset of molecules in .xyz format.

This dataset is itended for small case tests of molecules that are not part of specific datasets.
"""

from functools import lru_cache

import numpy as np
from loguru import logger

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.utils.molecules import read_xyz_file


class MiscXYZ(DataGenDataset):
    """Class for the arbitrary molecules.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "MiscXYZ",
        num_processes: int = 1,
    ):
        """Initialize the MiscXYZ dataset.

        Args:
            raw_data_dir: Path to the raw data directory.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the directory containing the labels.
            filename: The filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.
            external_potential_modification: configuration for external potential modification.

        Raises:
            AssertionError: If the subset is not in the list of available subsets.
        """
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        self.filename = filename.split(".")[0]
        self.num_molecules = self.get_num_molecules()

    def download(self) -> None:
        """This is just a stub as this kind of dataset can not be downloaded."""
        logger.info("No download required/possible for MiscXYZ dataset.")

    def get_num_molecules(self) -> int:
        """Get the number of molecules in the dataset.

        Returns:
            int: Number of molecules in the dataset.
        """
        return len(list(self.raw_data_dir.glob("*.xyz")))

    @lru_cache(maxsize=1)
    def get_all_atomic_numbers(self) -> np.ndarray:
        """Get all atomic numbers present in the dataset.

        Iterates over all molecules in the dataset and collects all atomic numbers.

        Returns:
            np.ndarray: Array of atomic numbers present in the dataset.
        """
        all_atomic_numbers = set()
        for id in self.get_ids():
            charges, _ = self.load_charges_and_positions(id)
            all_atomic_numbers.update(charges)

        return np.array(sorted(all_atomic_numbers))

    def load_charges_and_positions(self, id: int) -> tuple[list, list]:
        """Load nuclear charges and positions for the given molecule indices from the .xyz files.
        Args:
            ids: Array of indices of the molecules to compute.

        Returns:
            np.ndarray: Array of atomic numbers (A).
            np.ndarray: Array of atomic positions (A, 3).
        """
        # We iterate over this list of files often, but it's still negligible compared to the kohn-sham time
        file_name = list(self.raw_data_dir.glob(f"*_{id:06}.xyz"))[0]
        charges, positions = read_xyz_file(file_name)
        return charges, positions

    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules in the dataset.

        Returns:
            np.ndarray: Array of indices of the molecules in the dataset.
        """
        return np.sort(
            np.array(
                [int(f.stem.split("_")[1]) for f in self.raw_data_dir.glob("*.xyz") if f.is_file()]
            )
        )
