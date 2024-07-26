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
from scipy.stats import median_abs_deviation
import pwlf
from scipy import interpolate
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

import numpy as np


def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed

from statsmodels.nonparametric.smoothers_lowess import lowess
def lowess_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:
    
    try:
        frac = 180 / len(t_interp)
        
        not_nan = ~np.isnan(series)
        series, t_obs = series[not_nan], t_obs[not_nan]
        # dy = np.gradient(series, t_obs)
        # smoothed_dy = lowess(dy, t_obs, frac=frac, return_sorted=False)
        # y_smooth = np.zeros_like(t_obs)
        # y_smooth[0] = series[0]
        
        # # 对于每个点，使用梯形规则累加积分
        # for i in range(1, len(t_obs)):
        #     delta_x = t_obs[i] - t_obs[i - 1]
        #     # 使用梯度（变化率）直接更新Y值
        #     y_smooth[i] = y_smooth[i - 1] + smoothed_dy[i - 1] * delta_x
        
        # # 插值到指定的输出时间点
        # series_interp = np.interp(t_interp, t_obs, y_smooth)
        # return series_interp[t_out]
        
        series_smooth = lowess(series, t_obs, frac=frac, return_sorted=False)
        series_interp = np.interp(t_interp, t_obs, series_smooth)
        return series_interp[t_out]
    except:
        return np.zeros(len(t_out))

from sklearn.decomposition import FastICA
import numpy as np

def ica_denoise(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:
    """
    Perform ICA denoising on a time series data.

    :param series: Input time series data
    :param t_obs: Time observations corresponding to the input data
    :param t_interp: Time points for interpolation
    :param t_out: Time points for the output
    :param n_components: Number of ICA components to use (default is 1)

    :return: The denoised time series data at the specified output time points
    """

    # Remove NaN values
    valid_indices = ~np.isnan(series)
    t_obs = t_obs[valid_indices]
    series = series[valid_indices]

    try:
        # Interpolate series to uniform time points
        # series_interp = np.interp(t_interp, t_obs, series)

        # # Reshape for ICA
        # series_interp_reshaped = series_interp.reshape(-1, 1)

        # Apply ICA
        ica = FastICA(n_components=2)
        data_ica = ica.fit_transform(np.column_stack((series, t_obs)))

        # Inverse transform to get denoised series
        data_denoised = ica.inverse_transform(data_ica)

        # Flatten the denoised series
        # series_denoised_flat = series_denoised.flatten()

        # Interpolate to get values at the desired output time points
        denoised_output = np.interp(t_interp, t_obs, data_denoised[:,0])
        # denoised_output = np.interp(t_out, t_interp, series_denoised_flat)

        return denoised_output[t_out]
    except Exception as e:
        return np.zeros(len(t_out))

def pwlf_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:
    
    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    
    try:
        my_pwlf = pwlf.PiecewiseLinFit(t_obs, series)
        res = my_pwlf.fitfast(6, pop=3)
        series_pwlf = my_pwlf.predict(t_obs)
        residual = series - series_pwlf
        residual_filt = mz_score_filt(residual, t_obs, axis=0)
        series_smooth = series_pwlf + residual_filt
        series_interp = np.interp(t_interp, t_obs, series_smooth)
        return series_interp[t_out]
    except:
        return np.zeros(len(t_out))
        

from scipy.interpolate import make_lsq_spline, BSpline

def bspline_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:
    
    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    
    try:
        order = 3
        n_knots = len(t_interp) // 90
        knots = np.linspace(t_obs.min(), t_obs.max(), n_knots - order + 1)
        knots = np.concatenate(([t_obs.min()] * order, knots, [t_obs.max()] * order))
        
        bspline = make_lsq_spline(t_obs, series, knots, order)
        series_smooth = bspline(t_out)
        return series_smooth
    except:
        return np.zeros(len(t_out))

def smoothing_cubic_spline_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:

    """
    Calculate a smoothed series using median filtering.

    :param series: The input series to be smoothed
    :param t_obs: The time observations corresponding to the input series
    :param t_interp: The time values for interpolation
    :param t_out: The time values for the output series
    :param t_win: Smoothing window size (default is 90)

    :return:The smoothed series corresponding to the output time values t_out
    """
    from scipy.interpolate import splrep, BSpline
    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    try:
        # #Without outliers detetection option 1
        # smoothing_spline = splrep(t_obs, series, s=len(t_obs)*100)
        # series_interp = BSpline(*smoothing_spline,extrapolate=False)(t_out)

        # #Without outliers detetection option 2, this one seems worst
        # from scipy.interpolate import make_smoothing_spline
        # smoothing_spline = make_smoothing_spline(t_obs, series)
        # series_interp = smoothing_spline(t_out)

        #With outliers detetection
        smoothing_spline = splrep(t_obs, series, s=len(t_obs)*100)
        series_interp = BSpline(*smoothing_spline,extrapolate=False)(t_obs)
        t_obs_filtered = t_obs[abs(series_interp - series) / series_interp * 100 < 50]
        series_filtered = series[abs(series_interp - series) / series_interp * 100 < 50]
        smoothing_spline = splrep(t_obs_filtered, series_filtered, s=len(t_obs_filtered) * 100)
        series_interp = BSpline(*smoothing_spline,extrapolate=False)(t_out)

        # noinspection PyTypeChecker
        # series_smooth = median_filter(series_interp, size=t_win, mode="reflect", axes=0)
    except:
        print('Error in smoothing spline')
        return np.zeros(len(t_out))

    return series_interp

def ridge_regression_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:

    """
    Calculate a smoothed series using median filtering.

    :param series: The input series to be smoothed
    :param t_obs: The time observations corresponding to the input series
    :param t_interp: The time values for interpolation
    :param t_out: The time values for the output series
    :param t_win: Smoothing window size (default is 90)

    :return:The smoothed series corresponding to the output time values t_out
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    try:
        #search for the best parameters
        param_grid = {
    'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100],
    'gamma': np.logspace(-2, 2, 5)  # [0.01, 0.1, 1, 10, 100]
}
        krr = KernelRidge(kernel='rbf')

        grid_search = GridSearchCV(krr, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(t_obs.reshape(-1, 1), series.reshape(-1, 1))

        # Extract the best parameters
        best_alpha = grid_search.best_params_['alpha']
        best_gamma = grid_search.best_params_['gamma']

        print(best_alpha,best_gamma)
        krr = KernelRidge(kernel='rbf', alpha=best_alpha, gamma=best_gamma)
        # Fit the model
        krr.fit(t_obs.reshape(-1, 1), series.reshape(-1, 1))
        series_interp = krr.predict(t_out.reshape(-1, 1))

    except:
        print('Error in smoothing spline')
        return np.zeros(len(t_out))

    return series_interp.ravel()

def median_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:

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
        series_smooth = median_filter(series_interp, size=t_win, mode="reflect", axes=0)
    except:
        return np.zeros(len(t_out))

    return series_smooth[t_out]

def savgol_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:

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

def savgol_non_uniform_smooth(
    series: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    t_win: int = 90,
    sigma: int = 3,
    order: int | None = 3,
) -> np.ndarray:

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
        series_smooth = non_uniform_savgol(t_obs, series, window=9, polynom=3)
        # series_smooth = np.interp(t_interp, t_obs, series_smooth)
        series_interp = np.interp(t_interp, t_obs, series_smooth)
        # series_smooth = savgol_filter(series_interp, window_length=9, polyorder=order, axis=-1)
    # except:
    #     return np.zeros(len(t_out))
    # return series_smooth[t_out]
    except:
        return np.zeros(len(t_out))
    return series_interp[t_out]


def dask_smooth(
    dask_array: np.ndarray,
    t_obs: np.ndarray,
    t_interp: np.ndarray,
    t_out: np.ndarray,
    filt_func: str = gaussian_smooth,
    t_win: int = 90,
    sigma: int = 3,
    order: int = 3,
    axis: int = 2,
) -> da.array:

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

    return da.from_array(
        np.apply_along_axis(
            filt_func,
            axis,
            dask_array,
            t_obs=t_obs,
            t_interp=t_interp,
            t_out=t_out,
            t_win=t_win,
            sigma=sigma,
            order=order,
        )
    )


def dask_smooth_wrapper(
    dask_array: da.array,
    dates: xr.DataArray,
    t_out: np.ndarray,
    smooth_method: str = "gaussian",
    t_win: int = 90,
    sigma: int = 3,
    order: int = 3,
    axis: int = 2,
):

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
    filt_func = {"gaussian": gaussian_smooth, "ewma": ewma_smooth, "median": median_smooth, "savgol": savgol_smooth, 
                 "ICA": ica_denoise, "lowess": lowess_smooth, "pwlf": pwlf_smooth, "bspline": bspline_smooth, "savgol_non_uniform": savgol_non_uniform_smooth,'smoothing_spline':smoothing_cubic_spline_smooth,'ridge_regression':ridge_regression_smooth}

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


def z_score_filt(obs: da.array, z_thres: int = 2, axis: int = 2):

    """
    Remove the observations if it is 3 time the standard deviation from the average of observations over this pixel
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

def mz_score_filt(obs: da.array, z_thres: int = 3.5, axis: int = 2):

    """
    Remove the observations if it is 3 time the MAD from the median of observations over this pixel
    :param obs: cube data to filter
    :param z_thres: threshold to remove observations, if the absolute zscore is higher than this threshold (default is 3)
    :param axis: axis on which to perform the zscore computation
    :return: boolean mask
    """

    med = np.nanmedian(obs, axis=axis, keepdims=True)
    mad = np.nanmedian(abs(obs-med), axis=axis, keepdims=True)

    # mad = median_abs_deviation(obs, axis=axis)

    z_scores = 0.6745*(obs - med) / mad
    inlier_flag = np.abs(z_scores) < z_thres

    return inlier_flag

def NVVC_angle_filt(
    obs_cpx: np.array, vvc_thres: float = 0.1, angle_thres: int = 45, z_thres: int = 2, axis: int = 2
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


def NVVC_angle_mzscore_filt(
    obs_cpx: np.array, vvc_thres: float = 0.1, angle_thres: int = 45, mz_thres: int = 3.5, axis: int = 2
) -> (np.array):

    """
    Combine angle filter and zscore
    If the VVC is lower than a given threshold, outliers are filtered out according to the zscore, else to the median angle filter,
    i.e. pixels are filtered out if the angle with the observation is angle_thres away from the median vector
    :param obs_cpx: cube data to filter
    :param vvc_thres: threshold to combine zscore and median_angle filter
    :param angle_thres:  threshold to remove observations, remove the observation if it is angle_thres away from the median vector
    :param mz_thres: threshold to remove observations, if the absolute zscore is higher than this threshold (default is 3)
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

    inlier_flag = np.where(vvc_cond, angle_filter, mz_score_filt(velo_magnitude, mz_thres=mz_thres, axis=axis))

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
    Combine this filter based on the aspect with a filter based on the zscore only if the slope is lower than 3
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

    # combine a filter based on the aspect and a filter based on the zscore only if the slope is lower than 3
    slope_cond = slope > 3
    slope_filter = np.where(slope_cond, True, mz_score_filt(velo_magnitude, z_thres=z_thres, axis=axis))

    inlier_flag = np.logical_and(slope_filter, aspect_filter.data)

    return xr.DataArray(inlier_flag, dims=obs_cpx.dims, coords=obs_cpx.coords)

def flow_angle_filt(
    obs_cpx: xr.DataArray,
    direction: xr.DataArray,
    angle_thres: int = 45,
    z_thres: int = 3,
    axis: int = 2,
) -> da.array:
    """
    Remove the observations if it is angle_thres away from the given flow direction
    Combine this filter based on the aspect with a filter based on the zscore only if the 4/5 of the observations slower than 5 m/y
    :param obs_cpx: cube data to filter
    :param direction: given flow direction
    :param angle_thres: threshold to remove observations, remove the observation if it is angle_thres away from the median vector
    :param axis: axis on which to perform the zscore computation
    :return: boolean mask
    """
    vx, vy = np.real(obs_cpx), np.imag(obs_cpx)
    velo_magnitude = np.hypot(vx, vy)  # compute the norm of each observations

    angle_rad = np.arctan2(vx, vy)

    flow_direction = (np.rad2deg(angle_rad) + 360) % 360
    
    direction_diff = np.abs((flow_direction - direction + 180) % 360 - 180)
    
    angle_filter = direction_diff < angle_thres
    
    # if 1/5 of the observations larger than 5 m/y, then consider it as moving area
    # valid_and_greater_than_10 = (~np.isnan(velo_magnitude)) & (velo_magnitude > 5)
    # bis_ratio = np.sum(valid_and_greater_than_10, axis=2) / np.sum(~np.isnan(velo_magnitude), axis=2) 
    # bis_cond = bis_ratio.values[:, :, np.newaxis] > 0.2

    # mag_filter = np.where(bis_cond , True, z_score_filt(velo_magnitude, z_thres=z_thres, axis=axis))
    # angle_filter[np.expand_dims(np.isnan(direction), axis=2)] = True
    angle_filter = angle_filter.where(~np.isnan(direction), True) # change the stable area to true incase of all invalid data
    mag_filter = np.where(~np.isnan(direction) , True, z_score_filt(velo_magnitude, z_thres=z_thres, axis=axis))
    inlier_flag = np.logical_and(mag_filter, angle_filter.data)

    return xr.DataArray(inlier_flag, dims=obs_cpx.dims, coords=obs_cpx.coords)


def median_magnitude_filt(obs_cpx: np.array, median_magnitude_thres: int = 3, axis: int = 2):

    """
    Remove the observation if it median_magnitude_thres times bigger than the mean velocity at pixel, or if it is
    1/median_magnitude_thres times smaller than the mean velocity at pixel

    :param obs_cpx: [np array] --- Cube data to filter (complex where the real part is vx and the imaginary part is vy)
    :param median_magnitude_thres: [int] [default is 3] --- Position of the threshold relatively to the mean velocity at pixel
    :param axis: [int] [default is 2] --- Axis on which the threshold should be applied (default is the time axis)

    :return inlier_flag: [np array] --- Boolean mask of the size of vx (and vy)
    """

    vv = np.abs(obs_cpx)
    mean_magnitude = np.nanmedian(vv, axis=axis, keepdims=True)

    inlier_flag = np.where(
        (vv > mean_magnitude / median_magnitude_thres) & (vv < mean_magnitude * median_magnitude_thres), True, False
    )

    return inlier_flag


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
    angle_thres: int = 45,
    z_thres: int = 2,
    mz_thres=3.5,
    magnitude_thres: int = 1000,
    median_magnitude_thres=3,
    error_thres: int = 100,
    slope: xr.Dataset = None,
    aspect: xr.Dataset = None,
    direction: xr.Dataset = None,
    axis: int = 2,
):
    """

    :param da_vx: vx observations
    :param da_vy: vy observations
    :param filt_method: filtering method
    :param vvc_thres: threshold to combine zscore and median_angle filter
    :param angle_thres: threshold to remove observations, remove the observation if it is angle_thres away from the median vector
    :param z_thres: threshold to remove observations, if the absolute zscore is higher than this threshold (default is 2)
    :param mz_thres: threshold to remove observations, if the absolute mzscore is higher than this threshold (default is 3.5)
    :param magnitude_thres: threshold to remove observations, if the magnitude is higher than this threshold (default is 1000)
    :param error_thres: threshold to remove observations, if the magnitude is higher than this threshold (default is 100)
    :param axis: axis on which to perform the zscore computation (default is 2)
    :return:
    """

    if filt_method == "median_angle":  # delete according to a threshold in angle between observations and median vector
        obs_arr = da_vx.data + 1j * da_vy.data
        inlier_mask = obs_arr.map_blocks(median_angle_filt, angle_thres=angle_thres, axis=axis, dtype=obs_arr.dtype)

    elif filt_method == "vvc_angle":  # combination between z_score and median_angle
        obs_arr = da_vx.data + 1j * da_vy.data
        inlier_mask = obs_arr.map_blocks(
            NVVC_angle_filt, vvc_thres=vvc_thres, angle_thres=angle_thres, axis=axis, dtype=obs_arr.dtype
        )

    elif filt_method == "vvc_angle_mzscore":  # combination between z_score and median_angle
        obs_arr = da_vx.data + 1j * da_vy.data
        inlier_mask = obs_arr.map_blocks(
            NVVC_angle_mzscore_filt,
            vvc_thres=vvc_thres,
            angle_thres=angle_thres,
            mz_thres=mz_thres,
            axis=axis,
            dtype=obs_arr.dtype,
        )

    elif filt_method == "z_score":  # threshold according to the zscore
        inlier_mask_vx = da_vx.data.map_blocks(z_score_filt, z_thres=z_thres, axis=axis, dtype=da_vx.dtype)
        inlier_mask_vy = da_vy.data.map_blocks(z_score_filt, z_thres=z_thres, axis=axis, dtype=da_vy.dtype)
        inlier_mask = np.logical_and(inlier_mask_vx, inlier_mask_vy)

    elif filt_method == "mz_score":  # threshold according to the zscore
        inlier_mask_vx = da_vx.data.map_blocks(mz_score_filt, z_thres=z_thres, axis=axis, dtype=da_vx.dtype)
        inlier_mask_vy = da_vy.data.map_blocks(mz_score_filt, z_thres=z_thres, axis=axis, dtype=da_vy.dtype)
        inlier_mask = np.logical_and(inlier_mask_vx, inlier_mask_vy)


    elif filt_method == "magnitude":  # delete according to a threshold in magnitude
        obs_arr = np.hypot(da_vx.data, da_vy.data)
        inlier_mask = obs_arr.map_blocks(lambda x: x < magnitude_thres, dtype=obs_arr.dtype)

    elif (
        filt_method == "median_magnitude"
    ):  # the threshold in magnitude is computed relatively to the median of the data
        obs_arr = da_vx.data + 1j * da_vy.data
        inlier_mask = obs_arr.map_blocks(
            median_magnitude_filt, median_magnitude_thres=median_magnitude_thres, axis=axis, dtype=obs_arr.dtype
        )

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
    elif filt_method == "flow_angle":
        obs_arr = da_vx + 1j * da_vy
        _, direction_expanded = xr.broadcast(obs_arr, direction["direction"])
        direction_expanded = direction_expanded.chunk(obs_arr.chunks)
        inlier_mask = xr.map_blocks(
            flow_angle_filt,
            obs_arr,
            args=(direction_expanded,),
            template=obs_arr,
            kwargs={"angle_thres": angle_thres, "z_thres": z_thres, "axis": axis},
        )
    else:
        raise ValueError(
            f"Filtering method should be either 'median_angle', 'vvc_angle', 'topo_angle', 'z_score', 'magnitude', 'median_magnitude' or 'error'."
        )

    return inlier_mask.compute()
