import numpy as np
from pyscf import gto

from mldft.datagen.datasets.dataset import DataGenDataset


class SmallDataset(DataGenDataset):
    """Class for small dataset.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the directory containing the raw data.
        kohn_sham_data_dir: Path to the directory containing the Kohn-Sham data.
        num_processes: Number of processes to use for the computation.
        num_molecules: Number of molecules in the dataset.
        molecules: List of molecules in the dataset.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str,
        molecules: list[gto.M],
        num_processes: int = 1,
    ):
        """Define molecules and initialize using parent class.

        Args:
            raw_data_dir: Path to the directory containing the raw data.
            kohn_sham_data_dir: Path to the directory containing the Kohn-Sham data.
            label_dir: Path to the directory containing the labels.
            filename: The filename to use for the output files.
            name: Name of the dataset, used as the folder name.
            molecules: List of molecules in the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.
        """

        self.molecules = molecules

        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        self.num_molecules = self.get_num_molecules()

    def download(self) -> None:
        """Create the raw data directory and save the molecules as npz files."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.chmod(0o770)

        for i, mol in enumerate(self.molecules):
            positions = mol.atom_coords(unit="angstrom")
            atomic_numbers = mol.atom_charges()

            # Save data into separate npz files
            np.savez(
                self.raw_data_dir / f"mol_{int(i):07}.npz",
                positions=positions,
                atomic_numbers=atomic_numbers,
            )

    def get_num_molecules(self) -> int:
        """Get the number of molecules in the dataset."""
        return len(self.molecules)

    def load_charges_and_positions(self, id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load nuclear charges and positions for the given molecule indices.

        Args:
            id: Index of the molecules to compute.
        """
        data = np.load(self.raw_data_dir / f"mol_{id:07}.npz")
        charges = data["atomic_numbers"]
        positions = data["positions"]
        return charges, positions

    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules in the dataset."""
        return np.arange(self.get_num_molecules())

    def get_all_atomic_numbers(self) -> np.ndarray:
        """Get all atomic numbers in the dataset."""
        return np.unique(np.concatenate([mol.atom_charges() for mol in self.molecules]))
