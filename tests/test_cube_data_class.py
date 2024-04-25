import pytest

import ticoi.cube_data_classxr
from ticoi.cube_data_classxr import cube_data_class
import xarray as xr
import os
import pytest

import pytest
import os
from ticoi.cube_data_classxr import cube_data_class  # Assuming cube_data_class is defined in your_module

class Testclass_cube_data_xr:
    @pytest.fixture
    def base_filepath(self):
        """Returns the absolute path to the test data directory."""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))

    @pytest.fixture
    def filepath(self, base_filepath, request):
        """Dynamically get the filename from the test parameters."""
        filename = request.param  # Access the parameterized filename
        return os.path.join(base_filepath, filename)

    @pytest.fixture
    def cube_data_class_instance(self, filepath):
        """Loads the file into an instance of cube_data_class and returns it."""
        cube = cube_data_class()  # Create an instance of cube_data_class
        cube.load(filepath=filepath, verbose=False)  # Load data
        return cube

    @pytest.mark.parametrize("filepath", [
        "ITS_LIVE_Lowell_Lower_test.nc",
        "c_x18620_y08085_2016-2022_crop_GPS_Lower.nc"
    ], indirect=["filepath"])  # Note that indirect should specify which parameters are to be treated indirectly

    def test_load(self, cube_data_class_instance):
        """Tests that the cube_data_class_instance is properly loaded and contains expected data."""
        assert isinstance(cube_data_class_instance, cube_data_class), "Should be an instance of cube_data_class"
        assert isinstance(cube_data_class_instance.ds, xr.Dataset), "Should be an xarray dataset"

        required_variables = {'vx', 'vy', 'mid_date', 'x', 'y', 'date1', 'date2', 'errorx', 'errory', 'sensor', 'source','temporal_baseline'}
        assert required_variables.issubset(set(cube_data_class_instance.ds.variables)), "Dataset is missing variables"
        expected_dims = ("x", "y", "mid_date")
        assert tuple(cube_data_class_instance.ds['vx'].dims) == expected_dims, "Dimension order incorrect"

    def test_load_pixel(self,cube_data_class_instance):
        pix = cube_data_class_instance.load_pixel()