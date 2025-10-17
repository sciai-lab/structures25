import numpy as np
import pytest

dataset = [
    "qm9_dataset",
    "misc_dataset",
]
dataset_for_filenametest = [data + "_for_filenametest" for data in dataset[:1]]


@pytest.mark.slow
class TestDatasets:
    @pytest.mark.parametrize("dataset", dataset)
    def test_files_downloaded(self, dataset, request):
        """Test whether the files have been downloaded."""
        self.dataset = request.getfixturevalue(dataset)
        assert any(self.dataset.raw_data_dir.iterdir()), "Raw data directory is empty."

    @pytest.mark.parametrize("dataset", dataset)
    def test_get_num_molecules(self, dataset, request):
        """Test the get_num_molecules function."""
        self.dataset = request.getfixturevalue(dataset)
        num_molecules = self.dataset.get_num_molecules()
        assert num_molecules > 0, "Number of molecules should be greater than 0."

    @pytest.mark.parametrize("dataset", dataset)
    def test_load_charges_and_positions(self, dataset, request):
        """Test the load_charges_and_positions function."""
        self.dataset = request.getfixturevalue(dataset)
        ids = self.dataset.get_ids()
        charges, positions = self.dataset.load_charges_and_positions(ids[0])
        assert len(charges) == len(positions), "Number of charges and positions should be equal."
        assert all(position.shape[0] == 3 for position in positions), "Positions are not 3D."

    @pytest.mark.parametrize("dataset", dataset)
    def test_get_ids(self, dataset, request):
        """Test the get_ids function."""
        self.dataset = request.getfixturevalue(dataset)
        ids = self.dataset.get_ids()
        assert isinstance(ids, np.ndarray), "Returned value should be a NumPy array."
        assert len(ids) > 0, "Number of IDs should be greater than 0."

    @pytest.mark.parametrize("dataset", dataset_for_filenametest)
    def test_get_all_chk_files_from_id(self, dataset, request):
        """Test the get_ids function."""
        self.dataset = request.getfixturevalue(dataset)

        files = []
        for molecule_id in range(1, 5):
            files += self.dataset.get_all_chk_files_from_ids([molecule_id])

        assert len(files) == 5, "Number of files should be 5."

        files = self.dataset.get_all_chk_files_from_ids([0, 1, 2, 3, 4])
        assert len(files) == 5, "Number of files should be 5."
