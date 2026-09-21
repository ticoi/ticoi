import os

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

import ticoi.cube_data_classxr as cube_data_module
from ticoi.cube_data_classxr import (
    CubeDataClass,  # Assuming cube_data_class is defined in your_module
    _unique_valid_dates,
)
from ticoi.example import get_path


def test_unique_valid_dates_loads_each_lazy_array_once():
    calls = []

    def load_dates(name, values):
        calls.append(name)
        return values

    date1_values = np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[ns]")
    date2_values = np.array(["2020-01-02", "2020-01-03", "2020-01-04"], dtype="datetime64[ns]")
    date1 = xr.DataArray(
        da.from_delayed(delayed(load_dates)("date1", date1_values), shape=(3,), dtype="datetime64[ns]")
    )
    date2 = xr.DataArray(
        da.from_delayed(delayed(load_dates)("date2", date2_values), shape=(3,), dtype="datetime64[ns]")
    )

    actual = _unique_valid_dates(date1, date2)

    expected = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]").astype("datetime64[ns]")
    np.testing.assert_array_equal(actual, expected)
    assert calls == ["date1", "date2"]


def test_load_pixel_materializes_lazy_variables_in_one_compute(monkeypatch):
    cube = CubeDataClass()
    n = 4
    dates = np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]")
    shape = (n, 1, 1)
    cube.ds = xr.Dataset(
        {
            "date1": ("mid_date", da.from_array(dates, chunks=n)),
            "date2": ("mid_date", da.from_array(dates + 1, chunks=n)),
            "vx": (("mid_date", "y", "x"), da.ones(shape, chunks=shape)),
            "vy": (("mid_date", "y", "x"), da.ones(shape, chunks=shape)),
            "errorx": (("mid_date", "y", "x"), da.ones(shape, chunks=shape)),
            "errory": (("mid_date", "y", "x"), da.ones(shape, chunks=shape)),
            "temporal_baseline": ("mid_date", da.ones(n, chunks=n)),
        },
        coords={"mid_date": dates, "x": [0], "y": [0]},
        attrs={"proj4": "EPSG:3413"},
    )
    calls = 0
    original = xr.Dataset.compute

    def counted_compute(self, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, **kwargs)

    monkeypatch.setattr(xr.Dataset, "compute", counted_compute)
    data, _, _ = cube.load_pixel(0, 0, proj="EPSG:3413")

    np.testing.assert_array_equal(data[0], np.column_stack((dates, dates + 1)))
    assert calls == 1


def test_filter_expands_baseline_selection_when_initial_threshold_has_no_observations(monkeypatch):
    date1 = np.array(["2020-01-01"] * 3 + ["2020-01-02"] * 3 + ["2020-01-03"] * 3, dtype="datetime64[ns]")
    date2 = np.array(["2020-01-02"] * 3 + ["2020-01-03"] * 3 + ["2020-01-01"] * 3, dtype="datetime64[ns]")
    mid_date = date1 + (date2 - date1) // 2
    cube = CubeDataClass()
    cube.ds = xr.Dataset(
        {
            "date1": ("mid_date", date1),
            "date2": ("mid_date", date2),
            "vx": (("mid_date", "y", "x"), np.ones((9, 1, 1))),
            "vy": (("mid_date", "y", "x"), np.ones((9, 1, 1))),
            "temporal_baseline": ("mid_date", np.full(9, 200.0)),
        },
        coords={"mid_date": mid_date, "x": [0], "y": [0]},
        attrs={"proj4": "EPSG:3413"},
    )
    selected_counts = []

    def record_selected_observations(array, dates, t_out, **kwargs):
        selected_counts.append(len(dates))
        return np.zeros((len(t_out), array.shape[1], array.shape[2]), dtype=array.dtype)

    monkeypatch.setattr(cube_data_module, "numpy_smooth_wrapper", record_selected_observations)

    result, _ = cube.filter_cube_before_inversion(select_baseline=100, smooth_method="gaussian")

    assert selected_counts == [9, 9]
    assert result.sizes["mid_date"] == 2


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
        "filepath", ["ITS_LIVE_Lowell_Lower", "IGE_S2_Argentiere", "IGE_Pleiades_Argentiere"], indirect=["filepath"]
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
            "sensor",
            "source",
            "temporal_baseline",
        }
        assert required_variables.issubset(set(cube_data_class_instance.ds.variables)), "Dataset is missing variables"
        expected_dims = ("mid_date", "y", "x")
        assert tuple(cube_data_class_instance.ds["vx"].dims) == expected_dims, "Dimension order incorrect"

    # Test load_pixel for the cube from IGE, for different pixel coordinates, either in pixels or in EPSG:4326
    @pytest.mark.parametrize(
        "filepath", ["ITS_LIVE_Lowell_Lower"], indirect=["filepath"]
    )  # indirect mean that this parameter should be handled by a fixture that can interpret these values
    @pytest.mark.parametrize(
        "x, y, expected",
        [
            (1, 2, np.array([-5.0, -43.0, 112.6, 149.1, 15.0]).astype("float32")),
            (-138.18069, 60.29076, np.array([59.0, -6.0, 112.59999847, 149.1000061, 15.0]).astype("float32")),
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

        cube_data_class_instance.ds.load()
        loaded_data, loaded_mean, loaded_dates_range = cube_data_class_instance.load_pixel(x, y)
        np.testing.assert_array_equal(loaded_data[0], data[0])
        np.testing.assert_array_equal(loaded_data[1], data[1])
        assert loaded_mean == mean
        assert loaded_dates_range == dates_range
