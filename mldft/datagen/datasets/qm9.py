"""QM9 dataset.

Contains 129,133 molecules from the QM9 dataset. The ids of the molecules are given by the index of
the xyz file.
"""

import multiprocessing
from pathlib import Path

import numpy as np
from loguru import logger

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.utils.download import download_file, extract_tar
from mldft.utils.molecules import read_xyz_file


class QM9(DataGenDataset):
    """Class for the QM9 dataset.

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
        name: str = "QM9",
        num_processes: int = 1,
    ):
        """Initialize the QM9 dataset.

        Args:
            raw_data_dir: Path to the raw data directory.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the directory containing the labels.
            filename: The filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.

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
        """Download the raw data."""
        logger.info("Downloading QM9")
        downloaded_file = download_file(
            "https://ndownloader.figshare.com/files/3195389",
            self.raw_data_dir,
            filename="dsgdb9nsd.xyz.tar.bz2",
        )
        logger.info("Extracting QM9")
        extract_tar(downloaded_file, self.raw_data_dir, mode="r:bz2")
        downloaded_file.unlink()
        self.convert_xyz_files()

    def convert_xyz_files(self) -> None:
        """Convert the xyz files from QM9 to have the format 1e-6 instead of 1*^-6 which can't be
        read by pyscf."""
        logger.info(f"Converting xyz files to {self.raw_data_dir}")
        convert_folder_sorted_parallel(self.raw_data_dir, self.raw_data_dir, self.num_processes)

    def get_num_molecules(self) -> int:
        """Get the number of molecules in the dataset.

        Returns:
            int: Number of molecules in the dataset.
        """
        return len(list(self.raw_data_dir.glob("*.xyz")))

    def get_all_atomic_numbers(self) -> np.ndarray:
        return np.array([1, 6, 7, 8, 9])

    def load_charges_and_positions(self, id: int) -> tuple[list, list]:
        """Load nuclear charges and positions for the given molecule indices from the .xyz files.
        Args:
            ids: Array of indices of the molecules to compute.

        Returns:
            np.ndarray: Array of atomic numbers (A).
            np.ndarray: Array of atomic positions (A, 3).
        """
        # We iterate over this list of files often, but it's still negligible compared to the kohn-sham time
        file_name = self.raw_data_dir / f"dsgdb9nsd_{id:06}.xyz"
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


def convert_string_format(xyz_file_path: Path, out_folder: Path) -> None:
    """Convert the xyz files from QM9 to have the format 1e-6 instead of 1*^-6 which can't be read
    by pyscf.

    Args:
        xyz_file_path: Path to the xyz file
        out_folder: Path to the output folder
    """
    with open(xyz_file_path) as xyz_file:
        xyz_content = xyz_file.read()
    xyz_content = xyz_content.replace("*^", "e")
    with open(out_folder / xyz_file_path.name, "w") as xyz_file:
        xyz_file.write(xyz_content)


def convert_folder_sorted_parallel(in_folder: Path, out_folder: Path, num_processes: int) -> None:
    """Apply the conversion function to all xyz files in the folder in parallel.

    Args:
        in_folder: Path to the input folder
        out_folder: Path to the output folder
        num_processes: Number of processes to use
    """
    # Get the xyz files in the folder, sort and combine with the output folder
    out_folder.mkdir(exist_ok=True)
    out_folder.chmod(0o770)
    file_out_folder = [
        (xyz_file_path, out_folder) for xyz_file_path in sorted(in_folder.glob("*.xyz"))
    ]
    # Multiprocess for faster conversion
    logger.info(f"Using {num_processes} processes")
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(convert_string_format, file_out_folder)


class QM9Test(QM9):
    def download(self) -> None:
        downloaded_file = download_file(
            "https://figshare.com/ndownloader/files/3195398",
            self.raw_data_dir,
            filename="dsC7O2H10nsd.xyz.tar.bz2",
        )
        logger.info("Extracting QM9")
        extract_tar(downloaded_file, self.raw_data_dir, mode="r:bz2")
        downloaded_file.unlink()
        self.convert_xyz_files()

    def load_charges_and_positions(self, id: int) -> tuple[list, list]:
        """Load nuclear charges and positions for the given molecule indices from the .xyz files.
        Args:
            ids: Array of indices of the molecules to compute.

        Returns:
            np.ndarray: Array of atomic numbers (A).
            np.ndarray: Array of atomic positions (A, 3).
        """
        # We iterate over this list of files often, but it's still negligible compared to the kohn-sham time
        file_name = self.raw_data_dir / f"dsC7O2H10nsd_{id:04}.xyz"
        charges, positions = read_xyz_file(file_name)
        return charges, positions
