"""Utility functions to download and find example data."""

import os

# Define the location of the data in the example directory
_EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))

# Absolute filepaths to the example files.
_FILEPATHS_DATA = {"ITS_LIVE_Lowell_Lower": os.path.join(_EXAMPLES_DIRECTORY, "ITS_LIVE_Lowell_Lower_test.nc")}

available = list(_FILEPATHS_DATA.keys())


def get_path(name: str) -> str:
    """
    Get path of example data. List of available files can be found in "examples.available".

    :param name: Name of test data.
    :return:
    """
    if name in list(_FILEPATHS_DATA.keys()):
        return _FILEPATHS_DATA[name]
    else:
        raise ValueError('Data name should be one of "' + '" , "'.join(list(_FILEPATHS_DATA.keys())) + '".')
