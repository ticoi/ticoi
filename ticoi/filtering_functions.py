import numpy as np
import dask.array as da
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
import xarray as xr


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


def ewma_smooth(series: np.ndarray, t_obs: np.ndarray, t_interp: np.ndarray, t_out: np.ndarray, t_win: int = 90,
                sigma: int = 3, order: int | None = 3) -> np.ndarray:
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


def gaussian_smooth(series: np.ndarray, t_obs: np.ndarray, t_interp: np.ndarray, t_out: np.ndarray, t_win: int = 90,
                    sigma: int = 3, order: int | None = 3) -> np.ndarray:
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
        series = median_filter(series, size=5, mode='reflect', axes=0)
        series_interp = np.interp(t_interp, t_obs, series)
        series_smooth = gaussian_filter1d(series_interp, sigma, mode='reflect', truncate=4.0,
                                          radius=t_win)
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
    t_obs = (
        (dates.data - dates.data.min())
        .astype("timedelta64[D]")
        .astype("float64")
    )

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
    filt_func = {
        'gaussian': gaussian_smooth,
        'ewma': ewma_smooth,
        'median': median_smooth,
        'savgol': savgol_smooth
    }

    da_smooth = dask_array.map_blocks(dask_smooth, filt_func=filt_func[smooth_method], t_obs=t_obs,
                                      t_interp=t_interp, t_out=t_out, t_win=t_win, sigma=sigma,
                                      order=order, axis=axis, dtype=dask_array.dtype)

    return da_smooth

def z_score_filt(obs, z_thres=3, axis=2):
    
    mean = np.nanmean(obs, axis=axis, keepdims=True)
    std_dev = np.nanstd(obs, axis=axis, keepdims=True)

    z_scores = (obs - mean) / std_dev
    inlier_flag = np.abs(z_scores) < z_thres

    return inlier_flag

def median_angle_filt_np(vx, vy, angle_thres=45):
    
    vx_mean = np.nanmedian(vx, axis=2, keepdims=True)
    vy_mean = np.nanmedian(vy, axis=2, keepdims=True)
    
    mean_magnitude = np.hypot(vx_mean, vy_mean)
    velo_magnitude = np.hypot(vx, vy)
    
    # Check if magnitudes are greater than a threshold (tolerance) to avoid division by zero
    bis_cond = (velo_magnitude > 1e-6)
    
    dot_product = np.where(bis_cond, (vx_mean * vx + vy_mean * vy), np.nan)
    inlier_flag = np.where(bis_cond, (dot_product / (mean_magnitude * velo_magnitude) > np.cos(angle_thres * np.pi / 180)), False)
    
    return inlier_flag

def median_angle_filt(obs_cpx, angle_thres=45, axis=2):
    
    vx, vy = np.real(obs_cpx), np.imag(obs_cpx)
    
    vx_mean = np.nanmedian(vx, axis=axis, keepdims=True)
    vy_mean = np.nanmedian(vy, axis=axis, keepdims=True)
    
    mean_magnitude = np.hypot(vx_mean, vy_mean)
    velo_magnitude = np.hypot(vx, vy)
    
    # Check if magnitudes are greater than a threshold (tolerance) to avoid division by zero
    bis_cond = (velo_magnitude > 1e-6)
    
    dot_product = np.where(bis_cond, (vx_mean * vx + vy_mean * vy), np.nan)
    inlier_flag = np.where(bis_cond, (dot_product / (mean_magnitude * velo_magnitude) > np.cos(angle_thres * np.pi / 180)), False)
    
    return inlier_flag


def dask_filt_warpper(da_vx, da_vy, filt_method="median_angle", angle_thres=45, z_thres=3, magnitude_thres=1000, error_thres=100, axis=2):
    
    if filt_method == 'median_angle':
        obs_arr = da_vx.data + 1j * da_vy.data
        inlier_mask = obs_arr.map_blocks(median_angle_filt, angle_thres=angle_thres, axis=axis, dtype=obs_arr.dtype)
        
    elif filt_method == 'z_score':
        inlier_mask_vx = da_vx.data.map_blocks(z_score_filt, z_thres=z_thres, axis=axis, dtype=da_vx.dtype)
        inlier_mask_vy = da_vy.data.map_blocks(z_score_filt, z_thres=z_thres, axis=axis, dtype=da_vy.dtype)
        inlier_mask = np.logical_and(inlier_mask_vx, inlier_mask_vy)
        
    elif filt_method == 'magnitude':
        obs_arr = np.hypot(da_vx.data, da_vy.data)
        inlier_mask = obs_arr.map_blocks(lambda x: x < magnitude_thres, dtype=obs_arr.dtype)
        
    elif filt_method == 'error':
        inlier_mask_vx = da_vx.data.map_blocks(lambda x: x < error_thres, dtype=da_vx.dtype)
        inlier_mask_vy = da_vy.data.map_blocks(lambda x: x < error_thres, dtype=da_vy.dtype)
        inlier_mask = np.logical_and(inlier_mask_vx, inlier_mask_vy)
        
    else:
        raise ValueError(f"Filtering method should be either 'median_angle', 'z_score', 'magnitude' or 'error'.")
    
    return inlier_mask.compute()

