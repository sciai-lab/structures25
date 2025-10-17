from pathlib import Path
from tempfile import TemporaryDirectory

from mldft.datagen.datasets.qm9 import convert_string_format


def test_convert_string_format():
    """Test the conversion of the string format in a temporary file."""
    # Create a temporary directory to work in
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Create a temporary XYZ file with test data
        xyz_data = """H -6.8424*^-6 2.345 0.123
                     O 1.2345e-5 3.456 0.789"""
        temp_xyz_file = temp_dir_path / "test.xyz"
        temp_xyz_file.write_text(xyz_data)

        # Call your function to convert the temporary XYZ file
        convert_string_format(temp_xyz_file, temp_dir_path)

        # Verify the content of the converted file
        converted_file = temp_dir_path / "test.xyz"
        converted_content = converted_file.read_text()

        # Assert that the content has been properly converted
        assert "-6.8424e-6" in converted_content
        assert "1.2345e-5" in converted_content
