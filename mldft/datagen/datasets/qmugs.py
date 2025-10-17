"""QMUGS datasets.

Multiple possible subsets of the QMUGS dataset are defined.
"""

import os
from pathlib import Path

import numpy as np
import polars
from loguru import logger
from rdkit import Chem
from tqdm import tqdm

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.utils.download import download_file, extract_tar
from mldft.utils.molecules import load_charges_and_positions_sdf


def get_bin_from_num_atoms(num_heavy_atoms: int):
    """Get the bin of the molecule based on the number of heavy atoms.

    Under 10 will be mapped to -1, 10-15 to 0, 16-20 to 1 and so on.

    Args:
        num_heavy_atoms: Number of heavy atoms in the molecule.

    Returns:
        int: The bin of the molecule.
    """
    if num_heavy_atoms > 15:
        return (num_heavy_atoms - 1) // 5 - 2
    elif num_heavy_atoms > 9:
        return 0
    return -1


def extract_elements_from_smiles(smiles: str) -> set[int]:
    # Extract elements from the SMILES string considering potential stereochemistry annotations
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return {atom.GetAtomicNum() for atom in mol.GetAtoms()}


class QMUGS(DataGenDataset):
    """QMUGS dataset.

    This includes the whole dataset with 2 million molecules. The ids of the molecules are given by
    3 * chembl_id + conf_id, where conf_id is 0, 1 or 2.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "QMUGS",
        num_processes: int = 1,
        allowed_atomic_numbers: tuple[int] = (1, 6, 7, 8, 9),
    ):
        """Initialize the QMUGS dataset."""
        self.allowed_atomic_numbers = np.array(allowed_atomic_numbers)
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        self.filename = filename
        structure_info_path = self.raw_data_dir.parent / "structure_info.csv"
        if structure_info_path.exists():
            self.structure_info = polars.read_csv(structure_info_path)
        else:
            # This is not in download, to enable to update the csv file creation without having to download again
            csv_file = self.raw_data_dir.parent / "summary.csv"
            # Read the csv file with information on how many heavy atoms are inside each molecule
            logger.info("Reading the csv file to filter molecules.")
            structure_info = polars.read_csv(
                csv_file, columns=["chembl_id", "conf_id", "smiles", "atoms", "heavy_atoms"]
            )
            # Assign a unique id to each molecule and write it in the id column, it is given by 3 * chembl_id + conf_id
            # First add the individual integer columns
            structure_info = structure_info.with_columns(
                polars.col("chembl_id").str.slice(6).cast(polars.Int64).alias("chembl_int")
            )
            structure_info = structure_info.with_columns(
                polars.col("conf_id").str.slice(5).cast(polars.Int64).alias("conf_int")
            )
            # Finally create the combined id and sort
            structure_info = structure_info.with_columns(
                (3 * polars.col("chembl_int") + polars.col("conf_int")).alias("id")
            ).sort("id")
            logger.info("Filtering out molecules containing not supported elements.")
            mask = []
            atomic_numbers = []
            all_atomic_numbers = set()
            allowed_atomic_numbers = set(self.allowed_atomic_numbers)
            for row in tqdm(
                structure_info.iter_rows(),
                total=len(structure_info),
                desc="Scanning Data",
                dynamic_ncols=True,
            ):
                unique_atomic_numbers = extract_elements_from_smiles(row[2])
                not_allowed = any(
                    number not in allowed_atomic_numbers for number in unique_atomic_numbers
                )
                mask.append(not not_allowed)
                all_atomic_numbers.update(
                    unique_atomic_numbers.intersection(allowed_atomic_numbers)
                )
                atomic_numbers.append((",").join(map(str, unique_atomic_numbers)))
            if len(all_atomic_numbers - allowed_atomic_numbers) > 0:
                logger.warning(
                    f"Found unsupported elements: {all_atomic_numbers - allowed_atomic_numbers}"
                )
            structure_info = structure_info.with_columns(
                polars.Series("atomic_numbers", atomic_numbers)
            )
            structure_info = structure_info.filter(mask)
            # Bin by number of heavy atoms
            structure_info = structure_info.with_columns(
                polars.col("heavy_atoms")
                .map_elements(get_bin_from_num_atoms, return_dtype=int)
                .alias("bin")
            )
            structure_info.write_csv(self.raw_data_dir.parent / "structure_info.csv")
            self.structure_info = structure_info
        self.num_molecules = self.get_num_molecules()

    def download(self) -> None:
        """Download the raw data."""
        logger.info("Downloading QMUGS")
        structure_file = download_file(
            "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files"
            "=structures.tar.gz",
            self.raw_data_dir.parent,
            "structures.tar.gz",
        )
        logger.info("Extracting QMUGS, this can take a while.")
        extract_tar(structure_file, self.raw_data_dir.parent)
        structure_file.unlink()
        os.rename(self.raw_data_dir.parent / "structures", self.raw_data_dir)
        logger.info("Downloading the csv file with additional information.")
        download_file(
            "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files"
            "=summary.csv",
            self.raw_data_dir.parent,
            "summary.csv",
        )

    def get_num_molecules(self):
        """Get the number of molecules in the dataset."""
        return len(self.structure_info)

    @staticmethod
    def id_to_chembl_conf_id(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert the ids to chembl_id and conf_id.

        Args:
            ids: Array of indices of the molecules to compute.
        Returns:
            np.ndarray: Array of chembl_ids.
            np.ndarray: Array of conf_ids.
        """
        chembl_ids = ids // 3
        conf_ids = ids % 3
        return chembl_ids, conf_ids

    def load_charges_and_positions(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Load nuclear charges and positions for the given molecule indices.

        Args:
            ids: Array of indices of the molecules to compute.

        Returns:
            list: List of arrays of atomic numbers (N) (A).
            list: List of arrays of atomic positions (N) (A, 3).
        """
        # We iterate over this list of files often, but it's still negligible compared to the kohn-sham time
        chembl_id, conf_id = self.id_to_chembl_conf_id(ids)
        file_name = self.raw_data_dir / f"CHEMBL{chembl_id}" / f"conf_{conf_id:02d}.sdf"
        charges, positions = load_charges_and_positions_sdf(file_name)
        return charges, positions

    def get_all_atomic_numbers(self):
        """Get all atomic numbers in the dataset."""
        return self.allowed_atomic_numbers

    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules in the dataset.

        Returns:
            np.ndarray: Array of indices of the molecules in the dataset.
        """
        return self.structure_info["id"].to_numpy()


class QMUGSLargeBins(QMUGS):
    """A subset of QMUGS containing molecules larger than 15 heavy atoms.

    50 molecules from each bin of heavy atoms are randomly (but deterministically with seed 1)
    sampled.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "QMUGS",
        num_processes: int = 1,
        use_original_ids: bool = True,
    ):
        """Initialize the QMUGS dataset.

        Args:
            raw_data_dir: Path to the raw data directory.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the label directory.
            filename: Filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.
        """
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        # Remove bins smaller than 16 heavy atoms
        self.structure_info = self.structure_info.filter(polars.col("bin") >= 1)

        if use_original_ids:
            ids = np.loadtxt(Path(__file__).parent / "QMUGSLargeBins_Ids.txt")
            self.structure_info = self.structure_info.filter(polars.col("id").is_in(ids))
        else:
            # Sample 50 molecules from each bin
            self.structure_info = self.structure_info.group_by("bin").map_groups(
                lambda df: df.sample(50, seed=1)
            )
        # Sort again for reproducibility
        self.structure_info = self.structure_info.sort("id")
        self.num_molecules = self.get_num_molecules()
        # Write in ID file or check if it exists
        id_file = self.kohn_sham_data_dir.parent / "QMUGSLargeBins.csv"
        if id_file.exists():
            logger.info(f"ID file {id_file.name} already exists, checking if its the same.")
            assert polars.read_csv(id_file).equals(
                self.structure_info.drop("atomic_numbers")
            ), "The ID file already exists, but it is not the same as the current dataset."
        else:
            self.structure_info.drop("atomic_numbers").write_csv(id_file)


class QMUGSBin(QMUGS):
    """A subset of QMUGS containing molecules from a specific bin of heavy atoms."""

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "QMUGS",
        num_processes: int = 1,
        bin: int = 0,
    ):
        """Initialize the QMUGS dataset.

        Args:
            raw_data_dir: Path to the raw data directory.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the label directory.
            filename: Filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.
            bin: The bin of heavy atoms to include.
        """
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        # Remove bins smaller than 16 heavy atoms and smaller than 10 heavy atoms
        self.structure_info = self.structure_info.filter(polars.col("bin") == bin)
        # Sort again for reproducibility
        self.structure_info = self.structure_info.sort(["conf_int", "chembl_int"])
        # Write in ID file or check if it exists
        id_file = self.raw_data_dir.parent / f"QMUGSBin{bin}.csv"
        if id_file.exists():
            logger.info(f"ID file {id_file.name} already exists, checking if its the same.")
            assert polars.read_csv(id_file).equals(self.structure_info.drop("atomic_numbers"))
        else:
            self.structure_info.drop("atomic_numbers").write_csv(id_file)
