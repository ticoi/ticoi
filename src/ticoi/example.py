"""Utility functions to download and find example data."""

import os
import shutil
import tarfile
import tempfile
import urllib

# Define the location of the data in the example directory
_EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))

# Absolute filepaths to the example files.
_FILEPATHS_DATA = {
    "ITS_LIVE_Lowell_Lower": os.path.join(_EXAMPLES_DIRECTORY, "Lowell", "ITS_LIVE_Lowell_Lower_test.nc"),
    "Argentiere_example_interp": os.path.join(_EXAMPLES_DIRECTORY, "Argentiere", "Argentiere_example_interp.nc"),
    "Argentiere_static": os.path.join(_EXAMPLES_DIRECTORY, "Argentiere", "Argentiere_static.gpkg"),
    "Argentiere_iceflow": os.path.join(_EXAMPLES_DIRECTORY, "Argentiere", "Argentiere_iceflow.gpkg"),
}

available = list(_FILEPATHS_DATA.keys())


def download_examples(overwrite: bool = False) -> None:
    """
    Fetch the example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    if not overwrite and all(map(os.path.isfile, list(_FILEPATHS_DATA.values()))):
        print("Datasets exist")
        return

    # Static commit hash to be bumped every time it needs to be.
    commit = "3121f37e8de767cb7ea21cbd93b4dd59a81b1ced"
    # The URL from which to download the repository
    url = f"https://github.com/ticoi/ticoi_data/tarball/main#commit={commit}"

    # Create a temporary directory to extract the tarball in.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "data.tar.gz")

        response = urllib.request.urlopen(url)
        # If the response was right, download the tarball to the temporary directory
        if response.getcode() == 200:
            with open(tar_path, "wb") as outfile:
                outfile.write(response.read())
        else:
            raise ValueError(f"Example data fetch gave non-200 response: {response.status_code}")

        # Extract the tarball
        with tarfile.open(tar_path) as tar:
            try:
                tar.extractall(tmp_dir, filter="tar")
            except TypeError:  # For compatibility with different versions of python: The filter argument, which was added in Python 3.10.12, specifies how members are modified or rejected before extraction.
                tar.extractall(tmp_dir)

        # Find the first directory in the temp_dir (should only be one) and construct the example data dir paths.
        for dir_name in ["Argentiere", "Lowell"]:
            tmp_dir_name = os.path.join(
                tmp_dir,
                [dirname for dirname in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, dirname))][0],
                dir_name,
            )

            # Copy the temporary extracted data to the example directory.
            shutil.copytree(tmp_dir_name, os.path.join(_EXAMPLES_DIRECTORY, dir_name))


def get_path(name: str) -> str:
    """
    Get path of example data. List of available files can be found in "examples.available".

    :param name: Name of test data.
    :return:
    """

    if name in list(_FILEPATHS_DATA.keys()):
        download_examples()
        return _FILEPATHS_DATA[name]
    else:
        raise ValueError('Data name should be one of "' + '" , "'.join(list(_FILEPATHS_DATA.keys())) + '".')
