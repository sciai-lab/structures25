import tarfile
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from tqdm import tqdm


def download_file(url: str, folder: Path, filename: Optional[str] = None) -> Path:
    """Download a file from a URL.

    Args:
        url: URL of the file to download.
        folder: Path to the directory where the file will be downloaded.
        filename: Optional name of the file.
    """
    # Get the file name
    if filename is None:
        filename = url.split("/")[-1]
    file_path = folder / filename

    # create the folder if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o770)

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the total size of the file
        total_size = int(response.headers.get("content-length", 0))

        # Create a progress bar
        progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, leave=False)

        # Open the file and write the content
        with open(file_path, "wb") as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)

        # Close the progress bar
        progress_bar.close()

        # Check if the total size matches the size of the downloaded file
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Something went wrong while downloading the file")

    else:
        raise RuntimeError(f"The request was not successful. Status code: {response.status_code}")

    return file_path


def extract_tar(file_path: Path, extract_path: Path = None, mode="r:gz") -> None:
    """Extract files from a tar file.

    Args:
        file_path: Path to the tar file.
        extract_path: Path to the directory where the tar file will be extracted.
        mode: Mode of the tar file.
    """
    # If no extract path is given, extract in the same directory as the tar file
    if extract_path is None:
        extract_path = file_path.parent

    # Create the extract path if it doesn't exist
    extract_path.mkdir(parents=True, exist_ok=True, mode=0o770)

    # Open the tar file and extract it
    with tarfile.open(file_path, mode=mode) as tar:
        tar.extractall(path=extract_path)

    logger.info(f"File extracted at: {extract_path}")
