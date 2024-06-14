"""
Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter
from scipy.signal import savgol_filter

# %% ======================================================================== #
#                             TEMPORAL SMOOTHING                              #
# =========================================================================%% #

def numpy_ewma_vectorized(series: np.ndarray, halflife: int = 30) -> np.ndarray:
    """
    Calculate the exponentially weighted moving average of a series using vectorized operations.

    :param series: Input series for which the EWMA needs to be calculated
    :param halflife: Halflife parameter for the EWMA calculation (default is 30)

    :return: The exponentially weighted moving average of the input series
    """

    alpha = 1 - np.exp(-np.log(2) / halflife)
    alpha_rev = 1 - alpha
    n = series.shape[0]
    pows = alpha_rev ** (np.arange(n + 1))
    scale_arr = 1 / pows[:-1]
    offset = series[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)
    mult = series * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def ewma_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:
    """
    Calculates an exponentially weighted moving average (EWMA) of a series at specific time points.

    :param series: Input series to be smoothed
    :param t_obs: Time points of the observed series
    :param t_interp: Time points to interpolate the series at
    :param t_out: Time points to return the smoothed series at
    :param t_win: Smoothing window size (default is 90)
    :param halflife: Exponential decay factor (default is 90)

    :return: The smoothed series at the specified time points
    """

    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    try:
        series_interp = np.interp(t_interp, t_obs, series)
        series_smooth = numpy_ewma_vectorized(series_interp, halflife=t_win)
    except:  # If there is only nan
        return np.zeros(len(t_out))
    return series_smooth[t_out]


def gaussian_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:
    """
    Perform Gaussian smoothing on a time series data.

    :param series: Input time series data
    :param t_obs: Time observations corresponding to the input data
    :param t_interp: Time points for interpolation
    :param t_out: Time points for the output
    :param t_win: Smoothing window size (default is 90)
    :param sigma: Standard deviation for Gaussian kernel (default is 3)
    :param order: Order of the smoothing function (default is 3)

    :return:The smoothed time series data at the specified output time points
    """

    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    try:
        # noinspection PyTypeChecker
        # series = median_filter(series, size=5, mode='reflect', axes=0)
        series_interp = np.interp(t_interp, t_obs, series)
        series_smooth = gaussian_filter1d(series_interp, sigma, mode="reflect", truncate=4.0, radius=t_win)
        return series_smooth[t_out]
    except:
        return np.zeros(len(t_out))


def median_smooth(series: np.ndarray, t_obs: np.ndarray, t_interp: np.ndarray, t_out: np.ndarray, t_win: int = 90,
                  sigma: int = 3, order: int | None = 3) -> np.ndarray:
    """
    Calculate a smoothed series using median filtering.

    :param series: The input series to be smoothed
    :param t_obs: The time observations corresponding to the input series
    :param t_interp: The time values for interpolation
    :param t_out: The time values for the output series
    :param t_win: Smoothing window size (default is 90)

    :return:The smoothed series corresponding to the output time values t_out
    """

    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    try:
        series_interp = np.interp(t_interp, t_obs, series)
        # noinspection PyTypeChecker
        series_smooth = median_filter(series_interp, size=t_win, mode='reflect', axes=0)
    except:
        return np.zeros(len(t_out))

    return series_smooth[t_out]


def savgol_smooth(series: np.ndarray, t_obs: np.ndarray, t_interp: np.ndarray, t_out: np.ndarray, t_win: int = 90,
                  sigma: int = 3, order: int | None = 3) -> np.ndarray:
    """
    Perform Savitzky-Golay smoothing on a time series.

    :param series: Input time series to be smoothed
    :param t_obs: Observed time points corresponding to the input series
    :param t_interp: Time points for interpolation
    :param t_out: Time points to extract the smoothed values for
    :param t_win: Smoothing window size (default is 90)
    :param order: Order of the polynomial used in the smoothing (default is 3)

    :return: The smoothed time series at the specified output time points
    """

    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    try:
        series_interp = np.interp(t_interp, t_obs, series)
        series_smooth = savgol_filter(series_interp, window_length=t_win, polyorder=order, axis=-1)
    except:
        return np.zeros(len(t_out))
    return series_smooth[t_out]


def dask_smooth(dask_array: np.ndarray, t_obs: np.ndarray, t_interp: np.ndarray, t_out: np.ndarray,
                filt_func: str = gaussian_smooth, t_win: int = 90, sigma: int = 3, order: int = 3,
                axis: int = 2) -> da.array:
    """
    Apply smoothing to the input Dask array along the specified axis using the specified method.

    :param dask_array: Input Dask array to be smoothed.
    :param t_obs: Array of observation times corresponding to the input dask_array.
    :param t_interp: Array of times at which to interpolate the data.
    :param t_out: Array of times at which to output the smoothed data.
    :param filt_func: Smoothing method to be used ("gaussian", "emwa", "median", "savgol") (default is "gaussian")
    :param t_win: Smoothing window size (default is 90)
    :param sigma: Standard deviation for Gaussian smoothing (default is 3)
    :param order : Order of the smoothing function (default is 3)
    :param axis: Axis along which to apply the smoothing.

    :return: A Dask array containing the smoothed data.
    """

    # TODO : using scipy.interpolate instead of np.interp to do it for one chunk?
    # But it could be slow and memory intensive

    return da.from_array(np.apply_along_axis(filt_func, axis, dask_array, t_obs=t_obs,
                                             t_interp=t_interp, t_out=t_out, t_win=t_win, sigma=sigma, order=order))


def dask_smooth_wrapper(dask_array: da.array, dates: xr.DataArray, t_out: np.ndarray, smooth_method: str = "gaussian",
                        t_win: int = 90, sigma: int = 3, order: int = 3, axis: int = 2):
    """
    A function that wraps a Dask array to apply a smoothing function.

    :param dask_array: Dask array to be smoothed
    :param dates: Array of the central dates of the data
    :param t_out: Output timestamps for the smoothed array
    :param smooth_method: Smoothing method to be used ("gaussian", "emwa", "median", "savgol") (default is "gaussian")
    :param t_win: Smoothing window size (default is 90)
    :param sigma: Standard deviation for Gaussian smoothing (default is 3)
    :param order: Order of the smoothing function (default is 3)
    :param axis: Axis along which smoothing is applied (default is 2)

    :return: Smoothed dask array with specified parameters.
    """

    # Conversion of the mid_date of the observations into numerical values
    # It corresponds to the difference between each mid_date and the minimal date, in days
    t_obs = (dates.data - dates.data.min()).astype("timedelta64[D]").astype("float64")

    if t_out.dtype == "datetime64[ns]":  # Convert ns to days
        t_out = (t_out - dates.data.min()).astype("timedelta64[D]").astype("int")
    if t_out.min() < 0:
        t_obs = t_obs - t_out.min()  # Ensure the output time points are within the range of interpolated points
        t_out = t_out - t_out.min()

    # Some mid_date could be exactly the same, this will raise error latter
    # Therefore we add very small values to it
    while np.unique(t_obs).size < t_obs.size:
        t_obs += np.random.uniform(
            low=0.01, high=0.09, size=t_obs.shape
        )  # Add a small value to make it unique, in case of non-monotonic time point
    t_obs.sort()

    t_interp = np.arange(
        0, int(max(t_obs.max(), t_out.max()) + 1), 1
    )  # Time stamps for interpolated velocity, here every day

    # Apply a kernel on the observations to get a time series with a temporal sampling specified by t_interp
    filt_func = {"gaussian": gaussian_smooth, "ewma": ewma_smooth, "median": median_smooth, "savgol": savgol_smooth}

    da_smooth = dask_array.map_blocks(
        dask_smooth,
        filt_func=filt_func[smooth_method],
        t_obs=t_obs,
        t_interp=t_interp,
        t_out=t_out,
        t_win=t_win,
        sigma=sigma,
        order=order,
        axis=axis,
        dtype=dask_array.dtype,
    )

    return da_smooth


def z_score_filt(obs: da.array, z_thres: int = 3, axis: int = 2):
    """

    :param obs: cube data to filter
    :param z_thres: threshold to remove observations, if the absolute zscore is higher than this threshold (default is 3)
    :param axis: axis on which to perform the zscore computation
    :return: boolean mask
    """

    mean = np.nanmean(obs, axis=axis, keepdims=True)
    std_dev = np.nanstd(obs, axis=axis, keepdims=True)

    z_scores = (obs - mean) / std_dev
    inlier_flag = np.abs(z_scores) < z_thres

    return inlier_flag


def NVVC_angle_filt(
    obs_cpx: np.array, vvc_thres: float = 0.1, angle_thres: int = 45, z_thres: int = 3, axis: int = 2
) -> (np.array):
    """
    Combine angle filter and zscore
    If the VVC is lower than a given threshold, outliers are filtered out according to the zscore, else to the median angle filter,
    i.e. pixels are filtered out if the angle with the observation is angle_thres away from the median vector
    :param obs_cpx: cube data to filter
    :param vvc_thres: threshold to combine zscore and median_angle filter
    :param angle_thres:  threshold to remove observations, remove the observation if it is angle_thres away from the median vector
    :param z_thres: threshold to remove observations, if the absolute zscore is higher than this threshold (default is 3)
    :param axis: axis on which to perform the zscore computation
    :return: boolean mask
    """

    vx, vy = np.real(obs_cpx), np.imag(obs_cpx)
    vx_mean = np.nanmedian(vx, axis=axis, keepdims=True)
    vy_mean = np.nanmedian(vy, axis=axis, keepdims=True)
    mean_magnitude = np.hypot(vx_mean, vy_mean)  # compute the averaged norm of the observations

    velo_magnitude = np.hypot(vx, vy)  # compute the norm of each observations
    x_component = np.nansum(vx / velo_magnitude, axis=axis)
    y_component = np.nansum(vy / velo_magnitude, axis=axis)

    nz = velo_magnitude.shape[axis]
    VVC = (
        np.hypot(x_component, y_component) / nz
    )  # velocity coherence as defined in   Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    # ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
    VVC = np.expand_dims(VVC, axis=axis)

    vvc_cond = VVC > vvc_thres

    dot_product = vx_mean * vx + vy_mean * vy

    angle_filter = dot_product / (mean_magnitude * velo_magnitude) > np.cos(angle_thres * np.pi / 180)

    inlier_flag = np.where(vvc_cond, angle_filter, z_score_filt(velo_magnitude, z_thres=z_thres, axis=axis))

    return inlier_flag


def topo_angle_filt(
    obs_cpx: xr.DataArray,
    slope: xr.DataArray,
    aspect: xr.DataArray,
    angle_thres: int = 45,
    z_thres: int = 3,
    axis: int = 2,
) -> da.array:
    """
    Remove the observations if it is angle_thres away from the topographic gradient
    :param obs_cpx: cube data to filter
    :param slope: slope data
    :param aspect: aspect data
    :param angle_thres: threshold to remove observations, remove the observation if it is angle_thres away from the median vector
    :param axis: axis on which to perform the zscore computation
    :return: boolean mask
    """

    vx, vy = np.real(obs_cpx), np.imag(obs_cpx)
    velo_magnitude = np.hypot(vx, vy)  # compute the norm of each observations

    angle_rad = np.arctan2(vx, vy)

    flow_direction = (np.rad2deg(angle_rad) + 360) % 360

    aspect_diff = np.abs((flow_direction - aspect + 180) % 360 - 180)

    aspect_filter = aspect_diff < angle_thres
    # aspect_filter = np.where(aspect_cond, True, z_score_filt(velo_magnitude, z_thres=z_thres, axis=axis))

    slope_cond = slope > 3
    slope_filter = np.where(slope_cond, True, z_score_filt(velo_magnitude, z_thres=z_thres, axis=axis))

    inlier_flag = np.logical_and(slope_filter, aspect_filter.data)

    return xr.DataArray(inlier_flag, dims=obs_cpx.dims, coords=obs_cpx.coords)


def median_angle_filt(obs_cpx: np.array, angle_thres: int = 45, axis: int = 2):
    """
    Remove the observation if it is angle_thres away from the median vector
    :param obs_cpx: cube data to filter
    :param angle_thres: threshold to remove observations, remove the observation if it is angle_thres away from the median vector
    :param axis: axis on which to perform the zscore computation
    :return: boolean mask
    """

    vx, vy = np.real(obs_cpx), np.imag(obs_cpx)

    vx_mean = np.nanmedian(vx, axis=axis, keepdims=True)
    vy_mean = np.nanmedian(vy, axis=axis, keepdims=True)

    mean_magnitude = np.hypot(vx_mean, vy_mean)
    velo_magnitude = np.hypot(vx, vy)

    dot_product = vx_mean * vx + vy_mean * vy
    angle_filter = dot_product / (mean_magnitude * velo_magnitude) > np.cos(angle_thres * np.pi / 180)

    bis_cond = mean_magnitude > 10
    inlier_flag = np.where(bis_cond, angle_filter, z_score_filt(velo_magnitude, z_thres=3, axis=axis))

    return inlier_flag


def dask_filt_warpper(
    da_vx: xr.DataArray,
    da_vy: xr.DataArray,
    filt_method: str = "median_angle",
    vvc_thres: float = 0.3,
    angle_thres: int = 30,
    z_thres: int = 3,
    magnitude_thres: int = 1000,
    error_thres: int = 100,
    slope: xr.Dataset = None,
    aspect: xr.Dataset = None,
    axis: int = 2,
):
    """

    :param da_vx: vx observations
    :param da_vy: vy observations
    :param filt_method: filtering method
    :param vvc_thres: threshold to combine zscore and median_angle filter
    :param angle_thres: threshold to remove observations, remove the observation if it is angle_thres away from the median vector
    :param z_thres: threshold to remove observations, if the absolute zscore is higher than this threshold (default is 3)
    :param magnitude_thres: threshold to remove observations, if the magnitude is higher than this threshold (default is 1000)
    :param error_thres: threshold to remove observations, if the magnitude is higher than this threshold (default is 100)
    :param axis: axis on which to perform the zscore computation (default is 2)
    :return:
    """

    if filt_method == "median_angle":  # delete according to a treshold in angle between observations and median vector
        obs_arr = da_vx.data + 1j * da_vy.data
        inlier_mask = obs_arr.map_blocks(median_angle_filt, angle_thres=angle_thres, axis=axis, dtype=obs_arr.dtype)

    elif filt_method == "vvc_angle":  # combination between z_score and median_angle
        obs_arr = da_vx.data + 1j * da_vy.data
        inlier_mask = obs_arr.map_blocks(
            NVVC_angle_filt, vvc_thres=vvc_thres, angle_thres=angle_thres, axis=axis, dtype=obs_arr.dtype
        )

    elif filt_method == "z_score":  # threshold according to the zscore
        inlier_mask_vx = da_vx.data.map_blocks(z_score_filt, z_thres=z_thres, axis=axis, dtype=da_vx.dtype)
        inlier_mask_vy = da_vy.data.map_blocks(z_score_filt, z_thres=z_thres, axis=axis, dtype=da_vy.dtype)
        inlier_mask = np.logical_and(inlier_mask_vx, inlier_mask_vy)

    elif filt_method == "magnitude":  # delete according to a threshold  in magnitude
        obs_arr = np.hypot(da_vx.data, da_vy.data)
        inlier_mask = obs_arr.map_blocks(lambda x: x < magnitude_thres, dtype=obs_arr.dtype)

    elif filt_method == "error":  # delete according to a threshold  in error
        inlier_mask_vx = da_vx.data.map_blocks(lambda x: x < error_thres, dtype=da_vx.dtype)
        inlier_mask_vy = da_vy.data.map_blocks(lambda x: x < error_thres, dtype=da_vy.dtype)
        inlier_mask = np.logical_and(inlier_mask_vx, inlier_mask_vy)

    elif filt_method == "topo_angle":
        obs_arr = da_vx + 1j * da_vy
        _, slope_expanded, aspect_expanded = xr.broadcast(obs_arr, slope["slope"], aspect["aspect"])
        slope_expanded, aspect_expanded = slope_expanded.chunk(obs_arr.chunks), aspect_expanded.chunk(obs_arr.chunks)
        # template = obs_arr['aspect'].transpose('x', 'y').chunk(obs_arr.chunks[0:2])
        inlier_mask = xr.map_blocks(
            topo_angle_filt,
            obs_arr,
            args=(slope_expanded, aspect_expanded),
            template=obs_arr,
            kwargs={"angle_thres": angle_thres, "z_thres": z_thres, "axis": axis},
        )
    else:
        raise ValueError(
            f"Filtering method should be either 'median_angle', 'vvc_angle', 'topo_angle', 'z_score', 'magnitude' or 'error'."
        )

    return inlier_mask.compute()


# %%
