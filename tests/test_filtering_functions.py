import dask.array as da
import numpy as np
import pytest
import xarray as xr

from ticoi.filtering_functions import dask_filt_warpper, dask_smooth_wrapper, numpy_smooth_wrapper


@pytest.mark.parametrize(
    "method",
    [
        "median_angle",
        "vvc_angle",
        "vvc_angle_mzscore",
        "z_score",
        "mz_score",
        "iqr",
        "magnitude",
        "median_magnitude",
        "error",
    ],
)
def test_numpy_filter_matches_single_chunk_dask(method):
    rng = np.random.default_rng(11)
    vx = rng.normal(20, 5, (30, 3, 3)).astype("float32")
    vy = rng.normal(10, 3, (30, 3, 3)).astype("float32")
    vx[::7, 0, 0] = np.nan
    vy[::9, 2, 2] = np.nan
    coords = {"mid_date": np.arange(30), "y": np.arange(3), "x": np.arange(3)}
    vx_numpy = xr.DataArray(vx, dims=("mid_date", "y", "x"), coords=coords)
    vy_numpy = xr.DataArray(vy, dims=("mid_date", "y", "x"), coords=coords)
    data = xr.Dataset({"vx": vx_numpy, "vy": vy_numpy, "errorx": abs(vx_numpy), "errory": abs(vy_numpy)})

    expected = dask_filt_warpper(data.chunk(data.sizes), filt_method=method, axis=0)
    actual = dask_filt_warpper(data, filt_method=method, axis=0)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("method, threshold", [("vvc_angle", 45), ("flow_angle", 45), ("error", 10), ("iqr", 1.5)])
def test_dataset_filters_remove_outlier_without_masking_valid_velocities(method, threshold, chunked):
    from ticoi.cube_data_classxr import CubeDataClass

    dims = ("mid_date", "y", "x")
    vx = np.broadcast_to(np.array([20.0, 21.0, 22.0, -20.0])[:, None, None], (4, 2, 2)).copy()
    errors = np.broadcast_to(np.array([1.0, 1.0, 1.0, 200.0])[:, None, None], vx.shape).copy()
    cube = CubeDataClass()
    cube.ds = xr.Dataset(
        {"vx": (dims, vx), "vy": (dims, np.ones_like(vx)), "errorx": (dims, errors), "errory": (dims, errors)},
        coords={"mid_date": np.arange(4), "y": [0, 1], "x": [0, 1]},
    )
    if chunked:
        cube.ds = cube.ds.chunk({"mid_date": -1, "x": 1, "y": 1})
    direction = xr.Dataset({"direction": (("y", "x"), np.full((2, 2), 90.0))}, coords={"y": [0, 1], "x": [0, 1]})

    cube.delete_outliers({method: threshold}, direction=direction)

    expected = vx.copy()
    expected[-1] = np.nan
    np.testing.assert_allclose(cube.ds["vx"].values, expected)
    np.testing.assert_array_equal(np.isnan(cube.ds["vy"].values), np.isnan(expected))


def test_numpy_smoothing_matches_dask_exactly():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(40, 3, 3)).astype("float32")
    values[::11, 1, 1] = np.nan
    dates = np.datetime64("2020-01-01") + np.arange(40) * np.timedelta64(12, "D")
    # Include duplicate dates to verify identical random-number consumption.
    dates[10] = dates[9]
    dates = xr.DataArray(dates, dims="mid_date")
    t_out = dates.values[:-1] + np.diff(dates.values) // 2

    np.random.seed(1234)
    expected = dask_smooth_wrapper(
        da.from_array(values, chunks=values.shape), dates, t_out, t_win=11, order=3, axis=0
    ).compute()
    np.random.seed(1234)
    actual = numpy_smooth_wrapper(values, dates, t_out, t_win=11, order=3, axis=0)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_duplicate_date_jitter_is_reproducible_with_random_state():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(20, 2, 2)).astype("float32")
    dates = np.datetime64("2020-01-01") + np.arange(20) * np.timedelta64(12, "D")
    dates[8] = dates[7]
    dates = xr.DataArray(dates, dims="mid_date")
    t_out = dates.values[:-1] + np.diff(dates.values) // 2

    first = numpy_smooth_wrapper(values, dates, t_out, t_win=11, order=3, axis=0, random_state=42)
    np.random.uniform(size=100)
    second = numpy_smooth_wrapper(values, dates, t_out, t_win=11, order=3, axis=0, random_state=42)

    np.testing.assert_array_equal(second, first)


# import pytest
# from ticoi.filtering_functions import numpy_ewma_vectorized,ewma_smooth, gaussian_smooth
#
#
# class TestNumpyEWMA:
#     @pytest.mark.parametrize(
#         "series, halflife, expected",
#         [
#             (
#                     np.array([1, 2, 3, 4, 5], dtype=np.float64),
#                     3,
#                     np.array([1.0, 1.57955865, 2.31556391, 3.15139048, 4.04307395], dtype=np.float64),
#             ),
#             (
#                     np.array([5, 3, 6, 2, 8], dtype=np.float64),
#                     5,
#                     np.array([5.0, 4.44544421, 5.21563779, 3.88290174, 5.82557934], dtype=np.float64),
#             ),
#         ],
#     )
#     # def test_numpy_ewma_vectorized(self, series, halflife, expected):
#     #     result = numpy_ewma_vectorized(series, halflife)
#     #     np.testing.assert_almost_equal(
#     #         result,
#     #         expected,
#     #         decimal=5,
#     #         err_msg="EWMA calculation does not match expected values",
#     #     )
#     #
#     # def test_numpy_ewma_empty_array(self):
#     #     series = np.array([], dtype=np.float64)
#     #     result = numpy_ewma_vectorized(series, halflife=3)
#     #     assert result.size == 0, "EWMA calculation for empty array should return an empty array"
#
#     def test_numpy_ewma_single_element(self):
#         series = np.array([42], dtype=np.float64)
#         result = numpy_ewma_vectorized(series, halflife=5)
#         assert result[0] == 42, "EWMA for single element should return the element itself"
#
#     def test_numpy_ewma_negative_values(self):
#         series = np.array([-1, -3, -5, -7], dtype=np.float64)
#         result = numpy_ewma_vectorized(series, halflife=2)
#         assert result[-1] < 0, "EWMA for negative values should return negative numbers"
#
#     def test_numpy_ewma_constant_series(self):
#         series = np.array([4, 4, 4, 4, 4], dtype=np.float64)
#         result = numpy_ewma_vectorized(series, halflife=10)
#         np.testing.assert_almost_equal(
#             result,
#             series,
#             decimal=5,
#             err_msg="EWMA for constant series should equal the series values",
#         )
#
# class TestEWMA:
#     @pytest.mark.parametrize(
#         "series, t_obs, t_interp, t_out, t_win, expected",
#         [
#             # Test with a simple series and valid parameters
#             (
#                     np.array([1, 2, 3, 4, 5]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     90,
#                     np.array([1, 1.6, 2.504, 3.5016, 4.50012]),
#             ),
#             # Test with a series containing NaNs
#             (
#                     np.array([np.nan, 2, np.nan, 4, 5]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     90,
#                     np.array([2, 2.8, 3.64, 4.432, 5]),
#             ),
#             # Test with t_out outside the range of t_interp
#             (
#                     np.array([1, 2, 3, 4, 5]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([-1, 5]),
#                     90,
#                     np.array([0, 0]),  # Edge case: zeros for out-of-bounds
#             ),
#             # Test with a custom smoothing window
#             (
#                     np.array([1, 2, 3, 4, 5]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     10,
#                     np.array([1, 1.81818182, 2.65909091, 3.51525424, 4.38386868]),
#             ),
#         ],
#     )
#     # def test_ewma_values(self, series, t_obs, t_interp, t_out, t_win, expected):
#     #     result = ewma_smooth(series, t_obs, t_interp, t_out, t_win)
#     #     assert np.allclose(result, expected, equal_nan=True), "Smoothed values do not match expected output"
#
#     def test_ewma_empty_series(self):
#         # Test with an empty series
#         series = np.array([])
#         t_obs = np.array([])
#         t_interp = np.array([])
#         t_out = np.array([])
#         result = ewma_smooth(series, t_obs, t_interp, t_out, t_win=90)
#         assert result.size == 0, "Expected an empty output for an empty input"
#
#     def test_ewma_all_nan_series(self):
#         # Series with all NaNs
#         series = np.array([np.nan, np.nan, np.nan])
#         t_obs = np.array([1, 2, 3])
#         t_interp = np.array([1, 2, 3])
#         t_out = np.array([1, 2, 3])
#         result = ewma_smooth(series, t_obs, t_interp, t_out, t_win=90)
#         assert np.all(result == 0), "Expected all zeros for a series of NaNs"
#
#     # def test_ewma_mismatched_lengths(self):
#     #     # Test with mismatched lengths for input arrays
#     #     series = np.array([1, 2, 3])
#     #     t_obs = np.array([0, 1])  # Mismatched length
#     #     t_interp = np.array([0, 1, 2])
#     #     t_out = np.array([0, 1, 2])
#     #
#     #     with pytest.raises(ValueError):
#     #         ewma_smooth(series, t_obs, t_interp, t_out, t_win=90)
#
#     # def test_ewma_invalid_t_out(self):
#     #     # Test with invalid t_out values (e.g., non-integer or negative indices)
#     #     series = np.array([1, 2, 3])
#     #     t_obs = np.array([0, 1, 2])
#     #     t_interp = np.array([0, 1, 2])
#     #     t_out = np.array([-1, "invalid"])  # Invalid t_out values
#     #
#     #     with pytest.raises(TypeError):
#     #         ewma_smooth(series, t_obs, t_interp, t_out, t_win=90)
#
#
#
# class TestGaussianSmooth:
#
#     @pytest.mark.parametrize(
#         "series, t_obs, t_interp, t_out, t_win, sigma, order, expected",
#         [
#             (
#                     np.array([1, 2, 3, 4, 5]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     np.array([0, 1, 2, 3, 4]),
#                     90,
#                     3,
#                     3,
#                     np.array([1, 2, 3, 4, 5]),
#             ),
#         ],
#     )
#     # def test_gaussian_smooth_basic(
#     #         self, series, t_obs, t_interp, t_out, t_win, sigma, order, expected
#     # ):
#     #     result = gaussian_smooth(series, t_obs, t_interp, t_out, t_win, sigma, order)
#     #     assert np.allclose(result, expected)
#
#     def test_gaussian_smooth_empty_series(self):
#         series = np.array([])
#         t_obs = np.array([0, 1, 2, 3, 4])
#         t_interp = np.array([0, 1, 2, 3, 4])
#         t_out = np.array([0, 1, 2, 3, 4])
#         result = gaussian_smooth(series, t_obs, t_interp, t_out)
#         assert np.all(result == np.zeros(len(t_out)))
#
#     def test_gaussian_smooth_nan_series(self):
#         series = np.array([1, np.nan, 3, np.nan, 5])
#         t_obs = np.array([0, 1, 2, 3, 4])
#         t_interp = np.array([0, 1, 2, 3, 4])
#         t_out = np.array([0, 1, 2, 3, 4])
#         result = gaussian_smooth(series, t_obs, t_interp, t_out)
#         assert not np.isnan(result).any()
#
#     def test_gaussian_smooth_out_of_bounds(self):
#         series = np.array([1, 2, 3, 4, 5])
#         t_obs = np.array([0, 1, 2, 3, 4])
#         t_interp = np.array([-1, 0, 1, 2, 5])
#         t_out = np.array([0, 1, 2, 3, 4])
#         result = gaussian_smooth(series, t_obs, t_interp, t_out)
#         assert len(result) == len(t_out)
#
#     def test_gaussian_smooth_custom_parameters(self):
#         series = np.array([10, 20, 30, 40, 50])
#         t_obs = np.array([0, 2, 4, 6, 8])
#         t_interp = np.linspace(0, 8, 10)
#         t_out = np.array([1, 3, 5, 7])
#         result = gaussian_smooth(series, t_obs, t_interp, t_out, t_win=5, sigma=1)
#         assert len(result) == len(t_out)
