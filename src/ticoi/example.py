"""Utility functions to download and find example data.

Authors : Laurane Charrier, Lei Guo, Nathan Lioret
The package is based on the methodological developments published in:
- Charrier, L., Dehecq, A., Guo, L., Brun, F., Millan, R., Lioret, N., ... & Halas, P. (2025). TICOI: an operational
  Python package to generate regular glacier velocity time series. EGUsphere, 2025, 1-40.

- Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an
  optimal temporal sampling from displacement observation networks. IEEE Transactions on Geoscience and Remote Sensing,
  60, 1-10.
"""

import os
import shutil
import tarfile
import tempfile
import urllib
from filelock import FileLock

# Define the location of the data in the example directory
_EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))

# Absolute filepaths to the example files.
_FILEPATHS_DATA = {
    "ITS_LIVE_Lowell_Lower": os.path.join(_EXAMPLES_DIRECTORY, "Lowell", "ITS_LIVE_Lowell_Lower_test.nc"),
    "IGE_S2_Argentiere": os.path.join(_EXAMPLES_DIRECTORY, "Argentiere", "Alps_Mont-Blanc_Argentiere_S2.nc"),
    "IGE_Pleiades_Argentiere": os.path.join(
        _EXAMPLES_DIRECTORY, "Argentiere", "Alps_Mont-Blanc_Argentiere_Pleiades.nc"
    ),
    "Argentiere_example_interp": os.path.join(_EXAMPLES_DIRECTORY, "Argentiere", "Argentiere_example_interp.nc"),
    "Argentiere_static": os.path.join(_EXAMPLES_DIRECTORY, "Argentiere", "Argentiere_static.gpkg"),
    "Argentiere_iceflow": os.path.join(_EXAMPLES_DIRECTORY, "Argentiere", "Argentiere_iceflow.gpkg"),
}

available = list(_FILEPATHS_DATA.keys())


def download_examples(overwrite: bool = False) -> None:
    """
    Fetch example files safely across OSes and parallel test runs.
    Removes stale directories and uses a file lock to prevent race conditions.
    """
    os.makedirs(_EXAMPLES_DIRECTORY, exist_ok=True)
    lock_path = os.path.join(_EXAMPLES_DIRECTORY, ".example_data.lock")

    with FileLock(lock_path):
        # Skip download if all files exist and overwrite is False
        if not overwrite and all(map(os.path.isfile, _FILEPATHS_DATA.values())):
            print("Datasets exist")
            return

        # Static commit hash to be bumped when data changes
        commit = "3121f37e8de767cb7ea21cbd93b4dd59a81b1ced"
        url = f"https://github.com/ticoi/ticoi_data/tarball/main#commit={commit}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = os.path.join(tmp_dir, "data.tar.gz")

            # Download the tarball
            response = urllib.request.urlopen(url)
            if response.getcode() != 200:
                raise ValueError(f"Example data fetch gave non-200 response: {response.status}")

            with open(tar_path, "wb") as outfile:
                outfile.write(response.read())

            # Extract the tarball
            try:
                with tarfile.open(tar_path) as tar:
                    tar.extractall(tmp_dir, filter="tar")
            except TypeError:
                with tarfile.open(tar_path) as tar:
                    tar.extractall(tmp_dir)

            # Identify the top-level extracted directory
            top_level = next(d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d)))

            for dir_name in ["Argentiere", "Lowell"]:
                src = os.path.join(tmp_dir, top_level, dir_name)
                dst = os.path.join(_EXAMPLES_DIRECTORY, dir_name)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        print("Example datasets downloaded successfully.")


def get_path(name: str, overwrite: bool = False) -> str:
    """
    Get path of example data. List of available files can be found in "examples.available".

    :param name: Name of test data.
    :return:
    """

    if name in list(_FILEPATHS_DATA.keys()):
        download_examples(overwrite=overwrite)
        return _FILEPATHS_DATA[name]
    else:
        raise ValueError('Data name should be one of "' + '" , "'.join(list(_FILEPATHS_DATA.keys())) + '".')
