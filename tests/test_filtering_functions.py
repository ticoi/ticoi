# import numpy as np
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
