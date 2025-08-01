import os

import numpy as np
import pytest
import xarray as xr

from ticoi.cube_data_classxr import (
    CubeDataClass,  # Assuming cube_data_class is defined in your_module
)
from ticoi.example import get_path


class Testclass_cube_data_xr:
    @pytest.fixture
    def base_filepath(self):
        """Returns the absolute path to the test data directory."""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))

    @pytest.fixture
    def filepath(self, base_filepath, request):
        """Dynamically get the filename from the test parameters."""
        filename = request.param  # Access the parameterized filename
        return get_path(filename)

    @pytest.fixture
    def cube_data_class_instance(self, filepath):
        """Loads the file into an instance of cube_data_class and returns it."""
        cube = CubeDataClass()  # Create an instance of cube_data_class
        cube.load(filepath=filepath, verbose=False)  # Load data
        return cube

    # to do the test for several parameters, the function test can be decorated with pytest.mark.parametrize
    @pytest.mark.parametrize(
        "filepath", ["ITS_LIVE_Lowell_Lower"], indirect=["filepath"]
    )  # Note that indirect should specify which parameters are to be treated indirectly
    def test_load(self, cube_data_class_instance):
        """Tests that the cube_data_class_instance is properly loaded and contains expected data."""
        assert isinstance(cube_data_class_instance, CubeDataClass), "Should be an instance of cube_data_class"
        assert isinstance(cube_data_class_instance.ds, xr.Dataset), "Should be an xarray dataset"

        required_variables = {
            "vx",
            "vy",
            "mid_date",
            "x",
            "y",
            "date1",
            "date2",
            "errorx",
            "errory",
            "sensor",
            "source",
            "temporal_baseline",
        }
        assert required_variables.issubset(set(cube_data_class_instance.ds.variables)), "Dataset is missing variables"
        expected_dims = ("x", "y", "mid_date")
        assert tuple(cube_data_class_instance.ds["vx"].dims) == expected_dims, "Dimension order incorrect"

    # Test load_pixel for the cube from IGE, for different pixel coordinates, either in pixels or in EPSG:4326
    @pytest.mark.parametrize(
        "filepath", ["ITS_LIVE_Lowell_Lower"], indirect=["filepath"]
    )  # indirect mean that this parameter should be handled by a fixture that can interpret these values
    @pytest.mark.parametrize(
        "x, y, expected",
        [
            (1, 2, np.array([36.0, -52.0, 112.59999847, 149.1000061, 16.0]).astype("float32")),
            (-138.18069, 60.29076, np.array([59.0, -6.0, 112.59999847, 149.1000061, 16.0]).astype("float32")),
        ],
    )
    def test_load_pixel(self, cube_data_class_instance, x, y, expected):
        data, mean, dates_range = cube_data_class_instance.load_pixel(x, y)
        assert len(data) == 2, "Data is not a list of two elements"
        assert data[0].shape[1] == 2, "data_dates is not an array with two columns"
        assert str(data[0][0, 0].dtype) == "datetime64[D]" or str(data[0][0, 0].dtype) == "datetime64[s]", (
            "data_dates is not an array with two columns"
        )
        assert data[1].shape[1] == 5
        actual = data[1][0, :]
        np.testing.assert_array_almost_equal(actual, expected, decimal=1)
