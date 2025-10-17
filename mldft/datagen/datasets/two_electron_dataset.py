from pyscf import gto

from mldft.datagen.datasets.small_dataset import SmallDataset


class TwoElectronDataset(SmallDataset):
    """Class for the two-electron dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str,
        num_processes: int = 1,
    ):
        """Define molecules and initialize using parent class."""

        self.molecules = [
            # Hydrogen (H2)
            gto.M(atom="H 0 0 0; H 0 0 0.74", unit="angstrom"),
            # Helium (He)
            gto.M(atom="He 0 0 0", unit="angstrom"),
        ]

        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
            molecules=self.molecules,
        )
