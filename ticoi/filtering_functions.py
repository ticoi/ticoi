import numpy as np
import dask.array as da
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


# %% ======================================================================== #
#                             TEMPORAL SMOOTHING                              #
# =========================================================================%% #

def numpy_ewma_vectorized(series, halflife=30):
    """
    Calculate the exponentially weighted moving average of a series using vectorized operations.

    Parameters:\n
    :param series: Input series for which the EWMA needs to be calculated
    :param halflife: Halflife parameter for the EWMA calculation (default is 30)

    Returns:\n
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


def ewma_smooth(series, t_obs, t_interp, t_out, t_win=90, sigma=None, order=None):
    """
    Calculates an exponentially weighted moving average (EWMA) of a series at specific time points.

    Parameters:\n
    :param series: Input series to be smoothed
    :param t_obs: Time points of the observed series
    :param t_interp: Time points to interpolate the series at
    :param t_out: Time points to return the smoothed series at
    :param t_win: Smoothing window size (default is 90)
    :param halflife: Exponential decay factor (default is 90)

    Returns:\n
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


def gaussian_smooth(series, t_obs, t_interp, t_out, t_win=90, sigma=3, order=3):
    """
    Perform Gaussian smoothing on a time series data.

    Parameters:\n
    :param series: Input time series data
    :param t_obs: Time observations corresponding to the input data
    :param t_interp: Time points for interpolation
    :param t_out: Time points for the output
    :param t_win: Smoothing window size (default is 90)
    :param sigma: Standard deviation for Gaussian kernel (default is 3)
    :param order: Order of the smoothing function (default is 3)

    Returns:\n
    :return:The smoothed time series data at the specified output time points
    """

    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    try:
        series = median_filter(series, size=5, mode='reflect', axes=0)
        series_interp = np.interp(t_interp, t_obs, series)
        series_smooth = gaussian_filter1d(series_interp, sigma, mode='reflect', truncate=4.0,
                                          radius=t_win)
        return series_smooth[t_out]
    except:
        return np.zeros(len(t_out))


def median_smooth(series, t_obs, t_interp, t_out, t_win=90, sigma=None, order=None):
    """
    Calculate a smoothed series using median filtering.

    Parameters:\n
    :param series: The input series to be smoothed
    :param t_obs: The time observations corresponding to the input series
    :param t_interp: The time values for interpolation
    :param t_out: The time values for the output series
    :param t_win: Smoothing window size (default is 90)

    Returns:\n
    :return:The smoothed series corresponding to the output time values t_out
    """

    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    try:
        series_interp = np.interp(t_interp, t_obs, series)
        series_smooth = median_filter(series_interp, size=t_win, mode='reflect', axes=0)
    except:
        return np.zeros(len(t_out))

    return series_smooth[t_out]


def savgol_smooth(series, t_obs, t_interp, t_out, t_win=90, sigma=None, order=3):
    """
    Perform Savitzky-Golay smoothing on a time series.

    Parameters:\n
    :param series: Input time series to be smoothed
    :param t_obs: Observed time points corresponding to the input series
    :param t_interp: Time points for interpolation
    :param t_out: Time points to extract the smoothed values for
    :param t_win: Smoothing window size (default is 90)
    :param order: Order of the polynomial used in the smoothing (default is 3)

    Returns:\n
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


def dask_smooth(dask_array, t_obs, t_interp, t_out, filt_func=gaussian_smooth, t_win=90, sigma=3, order=3, axis=2):
    """
    Apply smoothing to the input Dask array along the specified axis using the specified method.

    Parameters:\n
    :param dask_array: Input Dask array to be smoothed.
    :param t_obs: Array of observation times corresponding to the input dask_array.
    :param t_interp: Array of times at which to interpolate the data.
    :param t_out: Array of times at which to output the smoothed data.
    :param method: Smoothing method to be used ("gaussian", "emwa", "median", "savgol") (default is "gaussian")
    :param t_win: Smoothing window size (default is 90)
    :param sigma: Standard deviation for Gaussian smoothing (default is 3)
    :param order : Order of the smoothing function (default is 3)
    :param axis: Axis along which to apply the smoothing.

    Returns:\n
    :return: A Dask array containing the smoothed data.
    """

    # TODO : using scipy.interpolate instead of np.interp to do it for one chunk?
    # But it could be slow and memory intensive

    return da.from_array(np.apply_along_axis(filt_func, axis, dask_array, t_obs=t_obs,
                                             t_interp=t_interp, t_out=t_out, t_win=t_win, sigma=sigma, order=order))


def dask_smooth_wrapper(dask_array, dates, t_out, smooth_method="gaussian", t_win=90, sigma=3, order=3, axis=2):
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
