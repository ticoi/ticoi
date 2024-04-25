import pytest

import ticoi.cube_data_classxr
from ticoi.cube_data_classxr import cube_data_class
import xarray as xr
import os

class Testclass_cube_data_xr:
    @pytest.fixture(autouse=True)
    def cube(self):
        return cube_data_class()

    @pytest.fixture
    def base_filepath(self):
        # This just sets up the base path to where your files are stored
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))

    @pytest.mark.parametrize("filename", [
        "ITS_LIVE_Lowell_Lower_test.nc",
        "c_x18620_y08085_2016-2022_crop_GPS_Lower.nc"
    ])

    def test_load(self, cube, base_filepath, filename):
        filepath_nc = os.path.join(base_filepath, filename)
        print(filepath_nc)
        cube.load(filepath=filepath_nc, verbose=False)
        assert isinstance(cube, cube_data_class), "cube should be an instance of cube_data_class"
        assert isinstance(cube.ds, xr.Dataset), "cube.ds should be an xarray dataset"

        # Check if all required variables are in the dataset
        required_variables = {'vx', 'vy', 'mid_date', 'x', 'y', 'date1', 'date2', 'errorx', 'errory', 'sensor',
                              'source'}
        dataset_variables = set(cube.ds.variables.keys())  # Ensure you're comparing against the keys
        missing_variables = required_variables - dataset_variables
        assert not missing_variables, f"Dataset is missing variables: {missing_variables}"

        # Check if the order of dimension is correct
        expected_dims = ("x", "y", "mid_date")
        assert tuple(cube.ds['vx'].dims) == expected_dims, f"Expected global dimensions {expected_dims}, but got {tuple(cube.ds.dims)}"