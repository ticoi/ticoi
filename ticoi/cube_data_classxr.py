'''
Class object to store and manipulate velocity observation data

Author : Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import numpy as np
import os
from ticoi.mjd2date import mjd2date  # /ST_RELEASE/UTILITIES/PYTHON/mjd2date.py
import xarray as xr
import pandas as pd
import dask
from pyproj import Proj, Transformer, CRS
import rasterio.enums
from datetime import date
from ticoi.secondary_functions import reconstruct_Common_Ref
import itertools
import warnings
import time
import dask.array as da
from dask.diagnostics import ProgressBar
from ticoi.secondary_functions import Construction_dates_range_np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

def determine_optimal_chuck_size(
        ds, variable_name="vx", x_dim="x", y_dim="y", verbose=True
):
    """
    A function to determine the optimal chunk size for a given time series array based on its size.
    
    Parameters:
    - ds: xarray Dataset containing the time series array
    - variable_name: Name of the variable containing the time series array (default is "vx")
    - x_dim: Name of the x dimension in the array (default is "x")
    - y_dim: Name of the y dimension in the array (default is "y")
    - verbose: Boolean flag to control verbosity of output (default is True)
    
    Returns:
    - tc: Chunk size along the time dimension
    - yc: Chunk size along the y dimension
    - xc: Chunk size along the x dimension
    """
    
    if verbose:
        print("Dask chunk size:")
    ## set chunk size to 5 MB if single time series array < 1 MB in size
    ## else increase to max of 1 GB chunk sizes.
    time_series_array_size = (
        ds[variable_name]
        .sel(
            {
                x_dim: ds[variable_name][x_dim].values[0],
                y_dim: ds[variable_name][y_dim].values[0],
            }
        )
        .nbytes
    )
    MB = 1048576
    if time_series_array_size < 1e6:
        chunk_size_limit = 50 * MB
    elif time_series_array_size < 1e7:
        chunk_size_limit = 100 * MB
    elif time_series_array_size < 1e8:
        chunk_size_limit = 200 * MB
    else:
        chunk_size_limit = 1000 * MB
    arr = ds[variable_name].data.rechunk(
        {0: -1, 1: "auto", 2: "auto"}, block_size_limit=chunk_size_limit, balance=True
    )
    tc, yc, xc = arr.chunks[0][0], arr.chunks[1][0], arr.chunks[2][0]
    chunksize = ds[variable_name][:tc, :yc, :xc].nbytes / 1e6
    if verbose:
        print("Chunk shape:", "(" + ",".join([str(x) for x in [tc, yc, xc]]) + ")")
        print(
            "Chunk size:",
            ds[variable_name][:tc, :yc, :xc].nbytes,
            "(" + str(round(chunksize, 1)) + "MB)",
        )
    return tc, yc, xc

def numpy_ewma_vectorized(series, halflife=30):
    """
    Calculate the exponentially weighted moving average of a series using vectorized operations.

    Parameters:
    series (np.array): The input series for which the EWMA needs to be calculated.
    halflife (int): The halflife parameter for the EWMA calculation. Default is 30.

    Returns:
    np.array: The exponentially weighted moving average of the input series.
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

    Parameters:
    - series: the input series to be smoothed
    - t_obs: the time points of the observed series
    - t_interp: the time points to interpolate the series at
    - t_out: the time points to return the smoothed series at
    - halflife: the exponential decay factor (default is 90)

    Returns:
    - The smoothed series at the specified time points
    """
    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    series_interp = np.interp(t_interp, t_obs, series)
    series_smooth = numpy_ewma_vectorized(series_interp, halflife=t_win)
    return series_smooth[t_out]


def gaussian_smooth(series, t_obs, t_interp, t_out, t_win=90, sigma=3, order=None):
    """
    Perform Gaussian smoothing on a time series data.

    Parameters:
    - series: The input time series data.
    - t_obs: The time observations corresponding to the input data.
    - t_interp: The time points for interpolation.
    - t_out: The time points for the output.
    - sigma: Standard deviation for Gaussian kernel (default is 3).
    - radius: The radius for smoothing (default is 90).

    Returns:
    - Smoothed time series data at the specified output time points.
    """
    
    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    series = median_filter(series, size=5, mode='reflect', axes=0)
    series_interp = np.interp(t_interp, t_obs, series)
    series_smooth = gaussian_filter1d(series_interp, sigma, mode='reflect', truncate=4.0,
                                                    radius=t_win)
    return series_smooth[t_out]

def median_smooth(series, t_obs, t_interp, t_out, t_win=90, sigma=None, order=None):
    """
    Calculate a smoothed series using median filtering.

    Parameters:
    - series: The input series to be smoothed.
    - t_obs: The time observations corresponding to the input series.
    - t_interp: The time values for interpolation.
    - t_out: The time values for the output series.
    - size: The window size for the median filter (default is 90).

    Returns:
    - The smoothed series corresponding to the output time values t_out.
    """
    
    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    series_interp = np.interp(t_interp, t_obs, series)
    series_smooth = median_filter(series_interp, size=t_win, mode='reflect', axes=0)
    return series_smooth[t_out]

def savgol_smooth(series, t_obs, t_interp, t_out, t_win=90, sigma=None, order=3):
    """
    Perform Savitzky-Golay smoothing on a time series.

    Parameters:
    - series: the input time series to be smoothed
    - t_obs: the observed time points corresponding to the input series
    - t_interp: the time points for interpolation
    - t_out: the time points to extract the smoothed values for
    - window_length: the length of the smoothing window (default is 90)
    - order: the order of the polynomial used in the smoothing (default is 3)

    Returns:
    - The smoothed time series at the specified output time points
    """
    t_obs = t_obs[~np.isnan(series)]
    series = series[~np.isnan(series)]
    series_interp = np.interp(t_interp, t_obs, series)
    series_smooth = savgol_filter(series_interp, window_length=t_win, polyorder=order, axis=-1)
    return series_smooth[t_out]

def dask_smooth(dask_array, t_obs, t_interp, t_out, filt_func=gaussian_smooth, t_win=90, sigma=3, order=3, axis=2):
    """
    Apply smoothing to the input Dask array along the specified axis using the specified method.
    
    Parameters:
    - dask_array: The input Dask array to be smoothed.
    - t_obs: The array of observation times corresponding to the input dask_array.
    - t_interp: The array of times at which to interpolate the data.
    - t_out: The array of times at which to output the smoothed data.
    - method: Smoothing method ("gaussian", "emwa", "median", "savgol"). Default is "gaussian". 
    - t_win: The time window size for smoothing. Default is 90. 
    - sigma: The standard deviation for Gaussian smoothing. Default is 3.
    - order: The order of the Savitzky-Golay filter for "savgol" method. Default is 5.
    - axis: The axis along which to apply the smoothing.

    Returns:
    - A Dask array containing the smoothed data.
    """
    # TODO : using scipy.interpolate instead of np.interp to do it for one chunk?  
    # But it could be slow and memory intensive
    
    return da.from_array(np.apply_along_axis(filt_func, axis, dask_array, t_obs=t_obs,
                                             t_interp=t_interp, t_out=t_out, t_win=t_win, sigma=sigma, order=order))


# TODO: find a more elegant way to handle the smoothing with different method
#       now the code is a bit of complicated and hard to read (too many lines)
#       But I can not find better way to do it currently...
def dask_smooth_wrapper(dask_array, dates, t_out, method="gaussian", t_win=90, sigma=3, order=90, axis=2):
    """
    A function that wraps a Dask array to apply a smoothing function. 
    Parameters:
        - dask_array: Dask array to be smoothed.
        - dates: Array of dates corresponding to the data.
        - t_out: Output timestamps for the smoothed array.
        - method: Method of smoothing (default is "gaussian").
        - t_win: Window size for smoothing (default is 90).
        - sigma: Standard deviation for Gaussian smoothing (default is 3).
        - order: Order of the smoothing function (default is 90).
        - axis: Axis along which smoothing is applied (default is 2).
    Returns:
        - Smoothed Dask array with specified parameters.
    """
    # conversion of the mid_date of the observations into numerical values
    # it corresponds the difference between each mid_date in the minimal date, in days
    t_obs = (
        (dates.data - dates.data.min())
        .astype("timedelta64[D]")
        .astype("float64")
    )

    if t_out.dtype == "datetime64[ns]":  #convert ns to days
        t_out = (t_out - dates.data.min()).astype("timedelta64[D]").astype("int")
    if t_out.min() < 0:
        t_obs = t_obs - t_out.min() #ensure the output time points are within the range of interpolated points
        t_out = t_out - t_out.min()
        
    # some mid_date could be exactly the same, this will raise error latter
    # therefore we add very small values to it
    while np.unique(t_obs).size < t_obs.size:
        t_obs += np.random.uniform(
            low=0.01, high=0.09, size=t_obs.shape
        )  # add a small value to make it unique, in case of non-monotonic time point
    t_obs.sort()

    t_interp = np.arange(
        0, int(max(t_obs.max(), t_out.max()) + 1), 1
    )  # time stamps for interpolated velocity, here every day

    #apply a kernel on the observations to get a time series with a temporal sampling specified by t_interp
    # NOTE: dask can not handle if..else... inside the map_blocks function
    if method == "gaussian":
        filt_func = gaussian_smooth
    elif method == "ewma":
        filt_func = ewma_smooth
    elif method == "median":
        filt_func = median_smooth
    elif method == "savgol":
        filt_func = savgol_smooth
    
    da_smooth = dask_array.map_blocks(dask_smooth, filt_func=filt_func, t_obs=t_obs, t_interp=t_interp, t_out=t_out,
                                       t_win=t_win, sigma=sigma, order=order,
                                       axis=axis, dtype=dask_array.dtype)
    
    return da_smooth

class cube_data_class:

    def __init__(self):
        self.filedir = ''
        self.filename = ''
        self.nx = 250
        self.ny = 250
        self.nz = 0
        self.author = ''
        self.ds = xr.Dataset({})

    def subset(self, proj, subset):
        """
        Crop according to 4 coordinates
        :param proj: EPSG system of the coordinates given in subset
        :param subset: list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :return: nothing, crop self.ds without the need of returning it
        """
        if CRS(self.ds.proj4) != CRS(proj):
            transformer = Transformer.from_crs(CRS(proj),
                                               CRS(self.ds.proj4))  # convert the coordinates from proj to self.ds.proj4
            lon1, lat1 = transformer.transform(subset[2], subset[1])
            lon2, lat2 = transformer.transform(subset[3], subset[1])
            lon3, lat3 = transformer.transform(subset[2], subset[1])
            lon4, lat4 = transformer.transform(subset[3], subset[0])
            self.ds = self.ds.sel(x=slice(np.min([lon1, lon2, lon3, lon4]), np.max([lon1, lon2, lon3, lon4])),
                                  y=slice(np.max([lat1, lat2, lat3, lat4]), np.min([lat1, lat2, lat3, lat4])))
            del lon1, lon2, lon3, lon4, lat1, lat2, lat3, lat4
        else:
            self.ds = self.ds.sel(x=slice(np.min([subset[0], subset[1]]), np.max([subset[0], subset[1]])),
                                  y=slice(np.max([subset[2], subset[3]]), np.min([subset[2], subset[3]])))

    def buffer(self, proj, buffer):
        """
        Crop the dataset around a given pixel, according to a given buffer
        :param proj: EPSG system of the coordinates given in subset
        :param buffer:  a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in m)
        :return: nothing, crop self.ds without the need of returning it
        """
        if CRS(self.ds.proj4) != CRS(proj):
            transformer = Transformer.from_crs(CRS(proj),
                                               CRS(self.ds.proj4))  # convert the coordinates from proj to self.ds.proj4
            i1, j1 = transformer.transform(buffer[1] + buffer[2],
                                           buffer[0] - buffer[2])
            i2, j2 = transformer.transform(buffer[1] - buffer[2],
                                           buffer[0] + buffer[2])
            i3, j3 = transformer.transform(buffer[1] + buffer[2],
                                           buffer[0] + buffer[2])
            i4, j4 = transformer.transform(buffer[1] - buffer[2],
                                           buffer[0] - buffer[2])
            self.ds = self.ds.sel(x=slice(np.min([i1, i2, i3, i4]), np.max([i1, i2, i3, i4])),
                                  y=slice(np.max([j1, j2, j3, j4]), np.min([j1, j2, j3, j4])))
            del i3, i4, j3, j4
        else:
            i1, j1 = buffer[0] - buffer[2], buffer[1] + buffer[2]
            i2, j2 = buffer[0] + buffer[2], buffer[1] - buffer[2]
            self.ds = self.ds.sel(x=slice(np.min([i1, i2]), np.max([i1, i2])),
                                  y=slice(np.max([j1, j2]), np.min([j1, j2])))
        del i1, i2, j1, j2, buffer

    # ====== = ====== LOAD DATASET ====== = ======
    def load_itslive(self, filepath, conf=False, pick_date=None, subset=None,
                     pick_sensor=None, pick_temp_bas=None, buffer=None,
                     verbose=False, proj='EPSG:4326'):  # {{{
        """
        Load a cube dataset written by ITS_LIVE
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """
        if verbose:
            print(filepath)

        self.filedir = os.path.dirname(filepath)  # path were is stored the netcdf file
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        self.ds = self.ds.assign_attrs({'proj4': self.ds['mapping'].proj4text})
        self.author = self.ds.author.split(', a NASA')[0]
        self.source = self.ds.url

        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)

        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        if pick_date is not None:
            self.ds = self.ds.where(((self.ds['acquisition_date_img1'] >= np.datetime64(pick_date[0])) & (
                    self.ds['acquisition_date_img2'] <= np.datetime64(pick_date[1]))).compute(), drop=True)

        self.nx = self.ds['x'].sizes['x']
        self.ny = self.ds['y'].sizes['y']
        self.nz = self.ds['mid_date'].sizes['mid_date']

        if conf:
            minconfx = np.nanmin(self.ds['vx_error'].values[:])
            maxconfx = np.nanmax(self.ds['vx_error'].values[:])
            minconfy = np.nanmin(self.ds['vy_error'].values[:])
            maxconfy = np.nanmax(self.ds['vy_error'].values[:])

        date1 = np.array([np.datetime64(date_str, 'ns') for date_str in self.ds['acquisition_date_img1'].values])
        date2 = np.array([np.datetime64(date_str, 'ns') for date_str in self.ds['acquisition_date_img2'].values])
        sensor = np.core.defchararray.add(np.char.strip(self.ds['mission_img1'].values.astype(str), '�'),
                                          np.char.strip(self.ds['satellite_img1'].values.astype(str), '�')
                                          ).astype(
            'U10')  # np.char.strip is used to remove the null character ('�') from each elemen and np.core.defchararray.add to concatenate array of different types
        sensor[sensor == 'L7'] = 'Landsat-7'
        sensor[sensor == 'L8'] = 'Landsat-8'
        sensor[sensor == 'L9'] = 'Landsat-9'
        sensor[np.isin(sensor, ['S1A', 'S1B'])] = 'Sentinel-1'
        sensor[np.isin(sensor, ['S2A', 'S2B'])] = 'Sentinel-2'

        if conf:  # normalize the error between 0 and 1, and convert error in confidence
            errorx = 1 - (self.ds['vx_error'].values - minconfx) / (maxconfx - minconfx)
            errory = 1 - (self.ds['vy_error'].values - minconfy) / (maxconfy - minconfy)
        else:
            errorx = self.ds['vx_error'].values
            errory = self.ds['vy_error'].values

        # Drop variables not in the specified list
        variables_to_keep = ['vx', 'vy', 'mid_date', 'x', 'y']
        self.ds = self.ds.drop_vars([var for var in self.ds.variables if var not in variables_to_keep])
        # Drop attributes not in the specified list
        attributes_to_keep = ['date_created', 'mapping', 'author', 'proj4']
        self.ds.attrs = {attr: self.ds.attrs[attr] for attr in attributes_to_keep if attr in self.ds.attrs}

        # self.ds = self.ds.unify_chunks()  # to avoid error ValueError: Object has inconsistent chunks along dimension mid_date. This can be fixed by calling unify_chunks().
        # Create new variable and chunk them
        self.ds['sensor'] = xr.DataArray(sensor, dims='mid_date').chunk(chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['date1'] = xr.DataArray(date1, dims='mid_date').chunk(chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['date2'] = xr.DataArray(date2, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['source'] = xr.DataArray(['ITS_LIVE'] * self.nz, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['errorx'] = xr.DataArray(
            errorx,
            dims=['mid_date'],
            coords={'mid_date': self.ds.mid_date}).chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['errory'] = xr.DataArray(
            errory,
            dims=['mid_date'],
            coords={'mid_date': self.ds.mid_date}).chunk(
            chunks=self.ds.chunks['mid_date'])

        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds['sensor'].isin(pick_sensor))
        if pick_temp_bas is not None:
            temp = ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))
            self.ds = self.ds.where(((pick_temp_bas[0] < temp) & (temp < pick_temp_bas[1])).compute(), drop=True)
            del temp
        self.ds = self.ds.unify_chunks()

    def load_millan(self, filepath, conf=False, pick_date=None, subset=None,
                    pick_sensor=None, pick_temp_bas=None, buffer=None,
                    verbose=False, proj='EPSG:4326'):
        """
        Load a cube dataset written by R. Millan et al.
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """

        if verbose:
            print(filepath)
        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        self.author = 'IGE'  # name of the author
        self.source = self.ds.source
        del filepath

        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)

        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        # Uniformization of the name and format of the time coordinate
        self.ds = self.ds.rename({'z': 'mid_date'})
        date1 = [mjd2date(date_str) for date_str in self.ds['date1'].values]  # convertion in date
        date2 = [mjd2date(date_str) for date_str in self.ds['date2'].values]
        self.ds = self.ds.unify_chunks()
        self.ds['date1'] = xr.DataArray(np.array(date1).astype('datetime64[ns]'), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['date2'] = xr.DataArray(np.array(date2).astype('datetime64[ns]'), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        del date1, date2

        if pick_date is not None:  # Temporal subset between two dates
            self.ds = self.ds.where(
                ((self.ds['date1'] >= np.datetime64(pick_date[0])) & (
                        self.ds['date2'] <= np.datetime64(pick_date[1]))).compute(),
                drop=True)
        del pick_date

        self.ds = self.ds.assign_coords(
            mid_date=np.array(self.ds['date1'] + (self.ds['date2'] - self.ds['date1']) // 2))

        self.nx = self.ds['x'].sizes['x']
        self.ny = self.ds['y'].sizes['y']
        self.nz = self.ds['mid_date'].sizes['mid_date']

        if conf and 'confx' not in self.ds.data_vars:  # convert the errors into confidence indicators between 0 and 1
            minconfx = np.nanmin(self.ds['error_vx'].values[:])
            maxconfx = np.nanmax(self.ds['error_vx'].values[:])
            minconfy = np.nanmin(self.ds['error_vy'].values[:])
            maxconfy = np.nanmax(self.ds['error_vy'].values[:])
            errorx = 1 - (self.ds['error_vx'].values - minconfx) / (maxconfx - minconfx)
            errory = 1 - (self.ds['error_vy'].values - minconfy) / (maxconfy - minconfy)
        else:
            errorx = self.ds['error_vx'].values[:]
            errory = self.ds['error_vy'].values[:]

        # Homogenize sensors names
        sensor = np.char.strip(self.ds['sensor'].values.astype(str),
                               '�')  # np.char.strip is used to remove the null character ('�') from each element
        sensor[np.isin(sensor, ['S1'])] = 'Sentinel-1'
        sensor[np.isin(sensor, ['S2'])] = 'Sentinel-2'
        sensor[np.isin(sensor, ['landsat-8', 'L8', 'L8. '])] = 'Landsat-8'

        # Drop variables not in the specified list
        self.ds = self.ds.drop_vars(
            [var for var in self.ds.variables if var not in ['vx', 'vy', 'mid_date', 'x', 'y', 'date1', 'date2']])
        self.ds = self.ds.transpose('mid_date', 'y', 'x')

        # Store the variable in xarray dataset
        self.ds['sensor'] = xr.DataArray(sensor, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        del sensor
        self.ds = self.ds.unify_chunks()
        self.ds['source'] = xr.DataArray(['IGE'] * self.nz, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['errorx'] = xr.DataArray(np.tile(errorx[:, np.newaxis, np.newaxis], (1, self.ny, self.nx)),
                                         dims=['mid_date', 'y', 'x'],
                                         coords={'mid_date': self.ds.mid_date, 'y': self.ds.y, 'x': self.ds.x}).chunk(
            chunks=self.ds.chunks)
        self.ds = self.ds.unify_chunks()
        self.ds['errory'] = xr.DataArray(np.tile(errory[:, np.newaxis, np.newaxis], (1, self.ny, self.nx)),
                                         dims=['mid_date', 'y', 'x'],
                                         coords={'mid_date': self.ds.mid_date, 'y': self.ds.y, 'x': self.ds.x}).chunk(
            chunks=self.ds.chunks)
        del errorx, errory

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds['sensor'].isin(pick_sensor))
        if pick_temp_bas is not None:
            self.ds = self.ds.sel(
                mid_date=(pick_temp_bas[0] < ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))) & (
                        ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D')) < pick_temp_bas[1]))
        self.ds = self.ds.unify_chunks()

    def load_ducasse(self, filepath, conf=False, pick_date=None, subset=None,
                     pick_sensor=None, pick_temp_bas=None, buffer=None,
                     verbose=False, proj='EPSG:4326'):
        """
        Load a cube dataset written by E. Ducasse et al.
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """

        if verbose:
            print(filepath)
        self.ds = self.ds.chunk({'x': 125, 'y': 125, 'time': 2000})  # set chunk
        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        # self.author = self.ds.author  # name of the author
        self.author = 'IGE'  # name of the author
        del filepath

        # Spatial subset
        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)

        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        # Uniformization of the name and format of the time coordinate
        self.ds = self.ds.rename({'time': 'mid_date'})
        date1 = [date_str.split(' ')[0] for date_str in self.ds['mid_date'].values]
        date2 = [date_str.split(' ')[1] for date_str in self.ds['mid_date'].values]
        self.ds['date1'] = xr.DataArray(np.array(date1).astype('datetime64[ns]'), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['date2'] = xr.DataArray(np.array(date2).astype('datetime64[ns]'), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        del date1, date2

        if pick_date is not None:  # Temporal subset between two dates
            self.ds = self.ds.where(
                ((self.ds['date1'] >= np.datetime64(pick_date[0])) & (
                        self.ds['date2'] <= np.datetime64(pick_date[1]))).compute(),
                drop=True)
        del pick_date

        self.ds = self.ds.assign_coords(
            mid_date=np.array(self.ds['date1'] + (self.ds['date2'] - self.ds['date1']) // 2))

        self.nx = self.ds['x'].sizes['x']
        self.ny = self.ds['y'].sizes['y']
        self.nz = self.ds['mid_date'].sizes['mid_date']

        # Drop variables not in the specified list
        variables_to_keep = ['vx', 'vy', 'mid_date', 'x', 'y', 'date1', 'date2']
        self.ds = self.ds.drop_vars([var for var in self.ds.variables if var not in variables_to_keep])
        self.ds = self.ds.transpose('mid_date', 'y', 'x')

        # Store the variable in xarray dataset
        self.ds['sensor'] = xr.DataArray(['Pleiades'] * len(self.ds['mid_date']), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['source'] = xr.DataArray(['IGE'] * self.nz, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['vy'] = -self.ds['vy']

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds['sensor'].isin(pick_sensor))
        if pick_temp_bas is not None:
            self.ds = self.ds.sel(
                mid_date=(pick_temp_bas[0] < ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))) & (
                        ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D')) < pick_temp_bas[1]))

    def load_charrier(self, filepath, conf=False, pick_date=None, subset=None,
                      pick_sensor=None, pick_temp_bas=None, buffer=None,
                      verbose=False, proj='EPSG:4326'):
        """
        Load a cube dataset written by L.Charrier et al.
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """

        if verbose:
            print(filepath)
        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        if self.ds.author == 'J. Mouginot, R.Millan, A.Derkacheva_aligned':
            self.author = 'IGE'  # name of the author
        else:
            self.author = self.ds.author

        self.source = self.ds.source
        del filepath

        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)

        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        if pick_date is not None:  # Temporal subset between two dates
            self.ds = self.ds.where(
                ((self.ds['date1'] >= np.datetime64(pick_date[0])) & (
                        self.ds['date2'] <= np.datetime64(pick_date[1]))).compute(),
                drop=True)
        del pick_date

        self.nx = self.ds['x'].sizes['x']
        self.ny = self.ds['y'].sizes['y']
        self.nz = self.ds['mid_date'].sizes['mid_date']

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds['sensor'].isin(pick_sensor))

        if pick_temp_bas is not None:
            self.ds = self.ds.sel(
                mid_date=(pick_temp_bas[0] < ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))) & (
                        ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D')) < pick_temp_bas[1]))

        if conf and 'confx' not in self.ds.data_vars:  # convert the errors into confidence indicators between 0 and 1
            minconfx = np.nanmin(self.ds['errorx'].values[:])
            maxconfx = np.nanmax(self.ds['errorx'].values[:])
            minconfy = np.nanmin(self.ds['errory'].values[:])
            maxconfy = np.nanmax(self.ds['errory'].values[:])
            errorx = 1 - (self.ds['errorx'].values - minconfx) / (maxconfx - minconfx)
            errory = 1 - (self.ds['errory'].values - minconfy) / (maxconfy - minconfy)
            self.ds['errorx'] = xr.DataArray(errorx,
                                             dims=['mid_date', 'y', 'x'],
                                             coords={'mid_date': self.ds.mid_date, 'y': self.ds.y,
                                                     'x': self.ds.x}).chunk(
                chunks=self.ds.chunks)
            self.ds['errory'] = xr.DataArray(errory,
                                             dims=['mid_date', 'y', 'x'],
                                             coords={'mid_date': self.ds.mid_date, 'y': self.ds.y,
                                                     'x': self.ds.x}).chunk(
                chunks=self.ds.chunks)

        # For cube writen with write_result_TICOI
        if 'source' not in self.ds.variables:
            self.ds['source'] = xr.DataArray([self.ds.author] * self.nz, dims='mid_date').chunk(
                chunks=self.ds.chunks['mid_date'])
        if 'sensor' not in self.ds.variables:
            self.ds['sensor'] = xr.DataArray([self.ds.sensor] * self.nz, dims='mid_date').chunk(
                chunks=self.ds.chunks['mid_date'])

    def load(self, filepath=None, conf=False, pick_date=None, subset=None,
             pick_sensor=None, pick_temp_bas=None, buffer=None, proj=None, chunks={},
             verbose=True):
        """
        Load a cube dataset which could be in format netcdf or zarr
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param chunks: dictionary with the size of chunks for each dimension, if chunks=-1 loads the dataset with dask using a single chunk for all arrays. chunks={} loads the dataset with dask using engine preferred chunks if exposed by the backend, otherwise with a single chunk for all arrays, chunks='auto' will use dask auto chunking taking into account the engine preferred chunks.
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """
        self.__init__()
        with dask.config.set(
                **{"array.slicing.split_large_chunks": False}
        ):  # To avoid creating the large chunks
            if filepath.split(".")[-1] == "nc":
                try:
                    self.ds = xr.open_dataset(filepath, engine="netcdf4", chunks=chunks)
                    if chunks == {}:
                        tc, yc, xc = determine_optimal_chuck_size(
                            self.ds,
                            variable_name="vx",
                            x_dim="x",
                            y_dim="y",
                            verbose=True,
                        )
                        self.ds = self.ds.chunk({"mid_date": tc, "x": xc, "y": yc})
                except (
                        NotImplementedError
                ):  # Can not use auto rechunking with object dtype. We are unable to estimate the size in bytes of object data
                    self.ds = xr.open_dataset(
                        filepath, engine="netcdf4", chunks={}
                    )  # set no chunks
                    if chunks == {}:
                        tc, yc, xc = determine_optimal_chuck_size(
                            self.ds,
                            variable_name="vx",
                            x_dim="x",
                            y_dim="y",
                            verbose=True,
                        )
                        self.ds = self.ds.chunk({"mid_date": tc, "x": xc, "y": yc})

            elif filepath.split(".")[-1] == "zarr":
                if chunks == {}:
                    chunks = "auto"  # change the default value to auto

                self.ds = xr.open_dataset(
                    filepath,
                    decode_timedelta=False,
                    engine="zarr",
                    consolidated=True,
                    chunks=chunks,
                )

        if verbose:
            print("file open")
        if (
                "Author" in self.ds.attrs
        ):  # uniformization of the attribute Author to author
            self.ds.attrs["author"] = self.ds.attrs.pop("Author")

        dico_load = {
            "ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)": self.load_itslive,
            "J. Mouginot, R.Millan, A.Derkacheva": self.load_millan,
            "J. Mouginot, R.Millan, A.Derkacheva_aligned": self.load_charrier,
            "L. Charrier, L. Guo": self.load_charrier,
            "L. Charrier": self.load_charrier,
            "E. Ducasse": self.load_ducasse,
            "S. Leinss, L. Charrier": self.load_charrier,
        }
        dico_load[self.ds.author](
            filepath,
            pick_date=pick_date,
            subset=subset,
            conf=conf,
            pick_sensor=pick_sensor,
            pick_temp_bas=pick_temp_bas,
            buffer=buffer,
            proj=proj,
        )
        # reorder the coordinates to keep the consistency
        self.ds = self.ds.copy().sortby("mid_date").transpose("x", "y", "mid_date")
        if verbose:
            print(self.ds.author)

    # ====== = ====== CONVERT CUBES DATA TO LIST OR ARRAY ====== = ======
    def sensor_(self):
        return self.ds['sensor'].values.tolist()

    def source_(self):
        return self.ds['source'].values.tolist()

    def temp_base_(self, list=True, format='float'):
        if format == 'D':
            temp = (self.ds['date2'] - self.ds['date1'])
        elif format == 'float':
            # temp = (self.ds['date2'].values-self.ds['date1'].values).astype('timedelta64[D]'))/ np.timedelta64(1, 'D')
            temp = ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))
        else:
            raise NameError('Please enter format as float or D')
        if list:
            return temp.values.tolist()
        else:
            return temp.values

    def date1_(self):
        return np.asarray(self.ds['date1']).astype('datetime64[D]')

    def date2_(self):
        return np.asarray(self.ds['date2']).astype('datetime64[D]')

    def datec_(self):
        return (self.date1_() + self.temp_base_(list=False, format='D') // 2).astype('datetime64[D]')

    def vv_(self):
        return np.sqrt(self.ds['vx'] ** 2 + self.ds['vy'] ** 2)

    # ====== = ====== PROCESS ON PIXEL BASIS  ====== = ======

    def load_pixel(self, i, j, unit=365, regu=1, coef=1, flags=None, solver='LSMR', interp='nearest', merged=None, proj='EPSG:4326',
                   visual=False, rolling_mean=None, verbose=False):
        '''
        Load data over one pixel
        :param i: int, x coordinate
        :param j: int, y coordinate
        :param unit: int, 365 in the unit is 'm/y' and 1 if the unit is 'm/d'
        :param regu: int or string : regularisation of the solver
        :param solver: string, 'LMSR', 'LMSMR_ini', 'LS_bounded', etc : solver mode
        :param interp: string, 'linear' or 'nearest'
        :param merged: None, or cube_data_class : cube_object to merge with self : the pd dataframe will contain values from self and merged
        :param proj: string EPSG of i,j projection
        :param velo_or_disp: string, 'disp' or 'vel' : if 'disp' return displacements, if 'vel' return velocities
        :param visual: bool, do you want to visualize the results
        :param verbose: bool, do you want to plot some text
        :return: pd dataframe
        '''

        # variables to keep
        var_to_keep = (
            ["date1", "date2", "vx", "vy", "errorx", "errory", "temporal_baseline"]
            if not visual
            else ["date1", "date2", "vx", "vy", "errorx", "errory", "temporal_baseline", "sensor", "source"]
        )

        # coordinates conversion
        # Conversion 165 µs ± 1.98 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
        if proj == 'int':
            data = self.ds.isel(x=i, y=j)[var_to_keep]
        else:
            # Convert coordinates if needed
            if proj == 'EPSG:4326':
                myProj = Proj(self.ds.proj4)
                i, j = myProj(i, j)
                if verbose: print(f'Converted to projection {self.ds.proj4}: {i, j}')
            else:
                if CRS(self.ds.proj4) != CRS(proj):
                    transformer = Transformer.from_crs(CRS(proj), CRS(self.ds.proj4))
                    i, j = transformer.transform(i, j)
                    if verbose: print(f'Converted to projection {self.ds.proj4}: {i, j}')
            # Interpolate only necessary variables and drop NaN values
            if interp == 'nearest':
                data = self.ds.sel(x=i, y=j, method='nearest')[var_to_keep].dropna(
                    dim='mid_date')  # 74.3 ms ± 1.33 ms per loop (mean ± std. dev. of 7 runs, 10 loops each
            else:
                data = self.ds.interp(x=i, y=j, method=interp)[var_to_keep].dropna(
                    dim='mid_date')  # 282 ms ± 12.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        if flags is not None:
            if isinstance(regu, dict) and isinstance(coef, dict):
                flag = np.round(flags['flags'].sel(x=i, y=j, method='nearest').values)
                regu = regu[flag]
                coef = coef[flag]
                # print(f'pixel:{i} {j} , flag: {flag}, assigned with regu {regu} and coef {coef}')
            else:
                raise ValueError("regu must be a dict if assign_flag is True!")
            
        data_dates = data[['date1', 'date2']].to_array().values.T
        if rolling_mean is None and (
                solver == 'LSMR_ini' or regu == '1accelnotnull' or regu == 'directionxy'):
            mean = np.array(
                [data['vx'].mean(),
                 data['vy'].mean()])
            dates_range = None
        else:  # 1.51 ms ± 12.6 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
            # Load rolling mean for the given pixel, only on the dates available
            dates_range = Construction_dates_range_np(
                data_dates)  # 652 µs ± 3.24 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
            mean = \
            rolling_mean.sel(mid_date=dates_range[:-1] + np.diff(dates_range) // 2, x=i, y=j, method='nearest')[
                ['vx_filt', 'vy_filt']]
            mean = [mean[i].values / unit for i in
                    ['vx_filt', 'vy_filt']]  # convert it to m/day
            
        if visual:
            data_str = data[['sensor', 'source']].to_array().values.T
            data_values = data.drop_vars(['date1', 'date2', 'sensor', 'source']).to_array().values.T
            data = [data_dates, data_values, data_str]
        else:
            data_values = data.drop_vars(['date1', 'date2']).to_array().values.T
            data = [data_dates, data_values]

        # TODO add merge case
        # TODO move this part into the inversion, and calculate the mean, dates_range for whole cube?
        if flags is not None:
            return data, mean, dates_range, regu, coef
        else:
            return data, mean, dates_range

    def coord2pix(self, x, y):
        '''Convert a point in coordinates to a point in pixels'''
        try:
            i = int(np.where(np.round(self.ds['x']).astype('int') == round(x))[0])
            j = int(np.where(np.round(self.ds['y']).astype('int') == round(y))[0])
        except:
            print('exact correspondance between x,y and i,j does not exist')
            i = int(
                (round(x) - round(np.min(self.ds['x'].values))) / (
                        self.ds['x'][1] - self.ds['x'][0]))  # [1,250] -> python index mode [0:249]
            j = int((round(y) - round(np.max(self.ds['y'].values))) / (
                    self.ds['y'][1] - self.ds['y'][0]))  # y et j varient en sens inverse
        return i, j

    # ====== = ====== PROCESS ON CUBE ====== = ======

    def preData_np(
            self,
            i=None,
            j=None,
            smooth_method="gaussian",
            s_win=3,
            t_win=90,
            sigma=3,
            order=3,
            unit=365,
            delete_outliers=None,
            flags=None,
            regu=1,
            proj="EPSG:4326",
            velo_or_disp="velo",
            verbose=False,
    ):
        """
        Preprocess data to be processed on cube
        :param i: int, x-coordinate of the considered pixel
        :param j: int, y-coordinate of the considered pixel
        :param smooth_method: string, ("gaussian", "median", "emwa", "savgol")
        :param s_win: int, size of the spatial window
        :param t_win: int, size of the temporal window, required for 
        :param sigma: int, size of the gaussian window
        :param order: int, order of the savgol smoothing
        :param unit: int, 365 of the unit is m/y 1 if the unit is m/d
        :param delete_outliers: None or int, if int delete all velocities which a quality indicator higher than delete_outliers
        :param regu: int or string : regularisation of the solver
        :param proj: string EPSG of i,j projection
        :param velo_or_disp: string, 'disp' or 'vel' to indicate the type of the observations
        :param verbose: bool, do you want to plot some text
        :return:
        """

        def loop_rolling(da_arr, mid_dates, date_range, smooth_method="gaussian", s_win=3, t_win=90, sigma=3, order=3, time_axis=2,verbose=False):
            """
            A function to calculate spatial mean, resample data, and calculate exponential smoothed velocity.

            Parameters:
            - array: input dask.array data
            - dates: time labels for input array, in datetime format, should have same length as array
            - s_win: window size for spatial average (default is 3)
            - t_win: time window size for ewma smoothing (default is 90)
            - sigma: standard deviation for gaussian filter (default is 3)
            - radius: radius for gaussian filter (default is 90)
            - time_axis: optional parameter for time axis (default is 2)

            Returns:
            - dask array with exponential smoothed velocity
            """

            from dask.array.lib.stride_tricks import sliding_window_view

            # calculate the mean of the velocity over the spatial window
            if verbose: start = time.time()
            # chunk size : ((10, 2), (20, 4), (61366,))
            spatial_mean = da.nanmean(sliding_window_view(da_arr.data, (s_win, s_win), axis=(0, 1)), axis=(-1, -2))
            spatial_mean = da.pad(
                spatial_mean,
                ((s_win // 2, s_win // 2), (s_win // 2, s_win // 2), (0, 0)),
                mode="edge",
            )
            # chunk size of spatial mean becomes after the pading: ((1, 9, 1, 1), (1, 20, 2, 1), (61366,))

            date_out = date_range[:-1] + np.diff(date_range) // 2

            """
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(1, 1, figsize=(12, 6))
            mid_date = cube['mid_date'].values
            ax.scatter(mid_date, series, marker='_', s=15, color='gray')
            ax.scatter(date_out, gaussian_filt, marker='v', s=15, color='blue')
            ax.scatter(date_out, median_filt, marker='^', s=15, color='green')
            ax.scatter(date_out, savgol_filt, marker='p', s=15, color='orange')
            ax.scatter(date_out, ewm_filt, marker='o', s=15, color='purple')
            ax.legend(['Observed', 'Gaussian', 'Median', 'SavGol', 'EWMA'], loc='upper left')
            f.savefig('compasion_different_smoother.png')
            """
            
            with ProgressBar():
                ewm_smooth = dask_smooth_wrapper(spatial_mean, mid_dates, t_out=date_out, method=smooth_method,
                                                 sigma=sigma, t_win=t_win, order=order, axis=time_axis).compute()

            if verbose: print(f'Smoothing observations took {round((time.time() - start), 1)} s')

            return ewm_smooth.compute(), np.unique(date_out)
        
        
        def loop_rolling2(da_arr, mid_dates, date_range, smooth_method="gaussian", s_win=3, t_win=90, sigma=3, order=3, baseline=None, time_axis=2,verbose=False):
            """
            A function to calculate spatial mean, resample data, and calculate exponential smoothed velocity.

            Parameters:
            - array: input dask.array data
            - dates: time labels for input array, in datetime format, should have same length as array
            - s_win: window size for spatial average (default is 3)
            - t_win: time window size for ewma smoothing (default is 90)
            - sigma: standard deviation for gaussian filter (default is 3)
            - radius: radius for gaussian filter (default is 90)
            - time_axis: optional parameter for time axis (default is 2)

            Returns:
            - dask array with exponential smoothed velocity
            """

            from dask.array.lib.stride_tricks import sliding_window_view

            date_out = date_range[:-1] + np.diff(date_range) // 2
            if verbose: start = time.time()
            
            if baseline is not None:
                baseline = baseline.compute()
                idx = np.where(baseline < 700 )
                mid_dates = mid_dates.isel(mid_date=idx[0])
                da_arr = da_arr.isel(mid_date=idx[0])
            
            with ProgressBar():
                ewm_smooth = dask_smooth_wrapper(da_arr.data, mid_dates, t_out=date_out, method=method, 
                                                 sigma=sigma, t_win=t_win, order=order, axis=time_axis).compute()

            if verbose: print(f'Smoothing observations took {round((time.time() - start), 1)} s')
            
            spatial_mean = da.nanmean(sliding_window_view(ewm_smooth, (s_win, s_win), axis=(0, 1)), axis=(-1, -2))
            spatial_mean = da.pad(
                spatial_mean,
                ((s_win // 2, s_win // 2), (s_win // 2, s_win // 2), (0, 0)),
                mode="edge",
            )
            
            # chunk size of spatial mean becomes after the pading: ((1, 9, 1, 1), (1, 20, 2, 1), (61366,))
            
            # ewm_smooth1 = ewm_smooth.compute()
            # smoothed = ewm_smooth1[145, 159, :]
            # spatial_mean1 = spatial_mean.compute()
            # filtered = spatial_mean1[145, 159, :]
            # import matplotlib.pyplot as plt
            # f, ax = plt.subplots(1, 1, figsize=(12, 6))
            # mid_date = cube['mid_date'].values
            # ax.scatter(mid_date, series, marker='o', s=15, color='gray')
            # ax.scatter(date_out, smoothed, marker='v', s=15, color='blue')
            # ax.scatter(date_out, filtered, marker='^', s=15, color='red')
            # ax.legend(['Observed', 'rolling', 'spatial'], loc='upper left')
            # f.savefig('comparison between original and smoothed 2.png')

            """
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(1, 1, figsize=(12, 6))
            mid_date = cube['mid_date'].values
            ax.scatter(mid_date, series, marker='_', s=15, color='gray')
            ax.scatter(date_out, gaussian_filt, marker='v', s=15, color='blue')
            ax.scatter(date_out, median_filt, marker='^', s=15, color='green')
            ax.scatter(date_out, savgol_filt, marker='p', s=15, color='orange')
            ax.scatter(date_out, ewm_filt, marker='o', s=15, color='purple')
            ax.legend(['Observed', 'Gaussian', 'Median', 'SavGol', 'EWMA'], loc='upper left')
            f.savefig('compasion_different_smoother.png')
            """
            


            return spatial_mean.compute(), np.unique(date_out)

              
        if i is not None and j is not None:
            if verbose: print("Clipping dataset to individual pixel: (x, y) = ({},{})".format(i, j))
            buffer = (s_win + 2) * (self.ds["x"][1] - self.ds["x"][0])
            self.buffer(proj, [i, j, buffer])
            self.ds = self.ds.unify_chunks()

        # convert all the dimension of the cube
        # TODO: do we need that?
        if CRS(self.ds.proj4) != CRS(proj):
            transformer = Transformer.from_crs(
                CRS(proj), CRS(self.ds.proj4)
            )  # convert the coordinates from proj to self.ds.proj4
            self.ds = self.ds.assign_coords(
                x=transformer.transform_x(self.ds.x, self.ds.y),
                y=transformer.transform_y(self.ds.x, self.ds.y),
            )
            if verbose:
                print(
                    "transform to projection: {proj} to {proj2}".format(
                        proj=proj, proj2=self.ds.proj4
                    )
                )
        # reorder the coordinates to keep the consistency
        # TODO need to put transpose in load function
        # DONE
        # cube = self.ds.copy().sortby("mid_date").transpose("x", "y", "mid_date")
        cube = self.ds.copy()
        cube["temporal_baseline"] = xr.DataArray((cube["date2"] - cube["date1"]).dt.days.values, dims='mid_date')
        # change the meaning of the velo_or_disp to avoid confusing
        # the rolling smooth should be carried on velocity, while we need displacement during inversion
        if velo_or_disp == "disp":  # to provide displacement values
            
            cube["vx"] = cube["vx"] / cube["temporal_baseline"] * unit
            cube["vy"] = cube["vy"] / cube["temporal_baseline"] * unit

        # TODO outlier removal, needs to complete
        if flags is not None:
            if isinstance(regu, dict):
                regu = list(regu.values())
            else:
                raise ValueError("regu must be a dict if assign_flag is True!")
        else:
            regu = list(regu)
        
        if delete_outliers == "median_angle":
            vx_mean = cube["vx"].mean(dim=['mid_date'])
            vy_mean = cube["vy"].mean(dim=['mid_date'])

            mean_magnitude = np.sqrt(vx_mean ** 2 + vy_mean ** 2)
            cube_magnitude = np.sqrt(cube["vx"] ** 2 + cube["vy"] ** 2)

            # Check if magnitudes are greater than a threshold (tolerance) to avoid division by zero
            tolerance = 1e-6
            valid_magnitudes = (cube_magnitude > tolerance).compute()

            # Calculate the dot product of mean velocity vector and individual velocity vectors
            cube_bis = cube.where(valid_magnitudes, drop=True)
            cube_magnitude = np.sqrt(cube_bis["vx"] ** 2 + cube_bis["vy"] ** 2)
            dot_product = (vx_mean * cube_bis["vx"] + vy_mean * cube_bis["vy"])

            # Calculate the angle condition
            angle_condition = (dot_product / (mean_magnitude * cube_magnitude) > np.sqrt(2) / 2).compute()

            # Apply the angle condition to filter the cube
            cube = cube_bis.where(angle_condition, drop=True)

            del angle, angle_condition,cube_bis
        elif isinstance(delete_outliers, int):
            cube = cube.where(
                (cube["errorx"] < delete_outliers)
                & (cube["errory"] < delete_outliers)
            )

        if ("1accelnotnull" in regu or "directionxy" in regu
        ):
            date_range = np.sort(np.unique(np.concatenate((cube['date1'].values, cube['date2'].values), axis=0)))
            if verbose:start = time.time()
            vx_filtered, dates_uniq = loop_rolling2(
                cube["vx"],
                cube["mid_date"],
                date_range,
                smooth_method=smooth_method,
                s_win=s_win,
                t_win=t_win,
                sigma=sigma,
                order=order,
                time_axis=2,
            )
            vy_filtered, dates_uniq = loop_rolling2(
                cube["vy"],
                cube["mid_date"],
                date_range,
                smooth_method=smooth_method,
                s_win=s_win,
                t_win=t_win,
                sigma=sigma,
                order=order,
                time_axis=2,
            )

            # the time dimension of the smoothed velocity observations is different from the original,
            # which is because of the possible dublicate mid_date of different image pairs...
            obs_filt = xr.Dataset(
                data_vars=dict(
                    vx_filt=(["x", "y", "mid_date"], vx_filtered),
                    vy_filt=(["x", "y", "mid_date"], vy_filtered),
                ),
                coords=dict(
                    x=(["x"], self.ds.x.data),
                    y=(["y"], self.ds.y.data),
                    mid_date=dates_uniq,
                ),
                attrs=dict(
                    description="Smoothed velocity observations",
                    units="m/y",
                    projection=self.ds.proj4,
                ),
            )
            obs_filt.load()
            del vx_filtered, vy_filtered

            if verbose:print(
                "Calculating smoothing mean of the observations completed in {:.2f} seconds".format(
                    time.time() - start
                )
            )
        else:
            obs_filt = None

        # unify the observations to displacement
        # to provide displacement values during inversion
        cube["vx"] = cube["vx"] * cube["temporal_baseline"] / unit
        cube["vy"] = cube["vy"] * cube["temporal_baseline"] / unit

        if "errorx" not in cube.variables:
            cube["errorx"] = (
                ("mid_date", "x", "y"),
                np.ones((len(cube["mid_date"]), len(cube["x"]), len(cube["y"]))),
            )
            cube["errory"] = (
                ("mid_date", "x", "y"),
                np.ones((len(cube["mid_date"]), len(cube["x"]), len(cube["y"]))),
            )

        self.ds = cube.load() #crash memory without loading

        # TODO calculate the mean, std, dates_range here for the whole cube
        return obs_filt

    def align_cube(self, cube2, unit='m/y', reproj_vel=True, reproj_coord=True, interp_method='cubic_spline'):
        """
         Reproject a cube2 to match the resolution, projection, and region of self, using a bilinear interpolation
        :param cube2: cube_data_classxr, cube2 to align to self
        :param unit: string, 'm/y' or 'm/d'
        :param reproj_vel: bool, if the velocity have to be reprojected -> it will modify their value
        :return: cube_data_classxr, cube2 aligned to self
        """

        if unit == 'm/y':
            conversion = 365
        elif unit == 'm/d':
            conversion = 1
        else:
            raise NameError('Please enter unit as m/d or m/y')

        if reproj_vel:  # if the velocity components have to be reprojected in the new projection system
            grid = np.meshgrid(cube2.ds['x'], cube2.ds['y'])
            temp = cube2.temp_base_()
            endx = np.array([(np.ma.masked_invalid(cube2.ds['vx'][z]) * temp[z] / conversion) + grid[0] for z in
                             range(
                                 cube2.nz)])  # localisation of the final coordinate of each pixel displaced by the corresponding velocity vector, in x
            endy = np.array(
                [(np.ma.masked_invalid(cube2.ds['vy'][z]) * temp[z] / conversion) + grid[1] for z in
                 range(
                     cube2.nz)])  # localisation of the final coordinate of each pixel displaced by the corresponding velocity vector, in y

            # reprojection of the final coordinate of each pixel displaced by the corresponding velocity vector
            transformer = Transformer.from_crs(cube2.ds.proj4, self.ds.proj4)
            t = np.array([transformer.transform(endx[z], endy[z]) for z in range(cube2.nz)])
            del endx, endy

            # Computation of the difference between final and oringinal coordinates in the new system
            grid = transformer.transform(grid[0], grid[1])
            vx = np.array([(grid[0] - t[z, 0, :, :]) / temp[z] * conversion for z in
                           range(cube2.nz)])  # positive toward the West
            vy = np.array([(t[z, 1, :, :] - grid[1]) / temp[z] * conversion for z in
                           range(cube2.nz)])  # positive toward the North
            cube2.ds['vx'] = xr.DataArray(vx.astype('float32'), dims=['mid_date', 'y', 'x'],
                                          coords={'mid_date': cube2.ds.mid_date, 'y': cube2.ds.y, 'x': cube2.ds.x})
            cube2.ds['vx'].encoding = {'vx': {'dtype': 'float32', 'scale_factor': 0.1, 'units': 'm/y'}}
            cube2.ds['vy'] = xr.DataArray(vy.astype('float32'), dims=['mid_date', 'y', 'x'],
                                          coords={'mid_date': cube2.ds.mid_date, 'y': cube2.ds.y, 'x': cube2.ds.x})
            cube2.ds['vy'].encoding = {'vy': {'dtype': 'float32', 'scale_factor': 0.1, 'units': 'm/y'}}
            del vx, vy
        if reproj_coord:
            # Convert the system of coordinate and ajust the spatial resolution of the cube2 to match the resolution, projection, and region of self, using a bilinear interpolation
            cube2.ds = cube2.ds.rio.write_crs(cube2.ds.proj4)
            self.ds = self.ds.rio.write_crs(self.ds.proj4)
            cube2.ds = cube2.ds.rio.reproject_match(self.ds, resampling=rasterio.enums.Resampling.average)
            # Update of cube_data_classxr attributes
            cube2.ds = cube2.ds.assign_attrs({'proj4': self.ds.proj4})
            # cube2.ds = cube2.ds.rio.write_crs(cube2.proj4, inplace=True)
            cube2.nx = cube2.ds.dims['x']
            cube2.ny = cube2.ds.dims['y']
            cube2.ds = cube2.ds.transpose('mid_date', 'y', 'x')
            cube2.ds = cube2.ds.assign_coords({"x": self.ds.x, "y": cube2.ds.y})
        cube2.ds = cube2.ds.assign_attrs({'author': f'{cube2.ds.author}_aligned'})

        return cube2

    def write_result_TICOI(self, result, source, sensor, filename='Time_series', savepath=None, result_quality=None,
                           verbose=False):
        """
        Write the result from TICOI, stored in result, in an xarray dataset matching the conventions CF-1.10
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.pdf
        units has been changed to unit, since it was producing an error while wirtting the netcdf file
        :param result: list of pd xarray, resulut from the TICOI method
        :param source: string, name of the source
        :param sensor: string, sensors which have been used
        :param filename: string, filename of file to saved
        :param savepath: string, path where to saved the file
        :return:
        """

        non_null_results = [result[i * self.ny + j]['vx'].shape[0] for i in range(self.nx) for j in range(self.ny)
                            if
                            result[i * self.ny + j]['vx'].shape[
                                0] != 0]  # temporal size of the results which are not empty
        first_date_results = [result[i * self.ny + j]['First_date'].iloc[0] for i in range(self.nx) for j in
                              range(self.ny) if
                              result[i * self.ny + j]['vx'].shape[
                                  0] != 0]  # temporal size of the results which are not empty
        if len(non_null_results) == 0:
            return 'There is no results to write and/or save'

        if np.min(non_null_results) == np.max(non_null_results) and all(
                element == first_date_results[0] for element in
                first_date_results):  # if the dates of the results are the same for every pixels
            Non_null_el = next((element for element in result if element.shape[0] != 0),
                               None)  # First result array which is not empty, and have size corresponding to the time period common between every pixels
            del non_null_results, first_date_results
            print('Same time dimension for every pixels')
        else:
            print('Not the same time dimension for every pixels')
            raise ValueError('Not the same time dimension for every pixels')

        cubenew = cube_data_class()
        time = Non_null_el['First_date'] + (Non_null_el['Second_date'] - Non_null_el['First_date']) // 2
        cubenew.ds['date1'] = xr.DataArray(Non_null_el['First_date'], dims='mid_date', coords={'mid_date': time})
        cubenew.ds['date1'].attrs = {'standard_name': 'first_date', 'unit': 'days',
                                     'long_name': 'first date between which the velocity is estimated'}
        cubenew.ds['date2'] = xr.DataArray(Non_null_el['Second_date'], dims='mid_date', coords={'mid_date': time})
        cubenew.ds['date2'].attrs = {'standard_name': 'second_date', 'unit': 'days',
                                     'long_name': 'second date between which the velocity is estimated'}

        long_name = ['velocity in the East/West direction [m/y]', 'velocity in the North/South direction [m/y]',
                     'number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)) in the East/West direction',
                     'number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)) in the North/South direction']
        short_name = ['x_velocity', 'y_velocity', 'x_countx', 'x_county']
        if result_quality is not None and 'X_contribution' in result_quality:
            variables = ['vx', 'vy', 'x_countx', 'x_county']
        else:
            variables = ['vx', 'vy']
        for i, var in enumerate(variables):
            result_arr = np.array(
                [result[i * self.ny + j][var] if result[i * self.ny + j][var].shape[
                                                     0] != 0 else pd.Series(
                    np.full(Non_null_el.shape[0], np.nan)) for i in range(self.nx) for j in range(self.ny)])
            result_arr = result_arr.reshape((self.nx, self.ny, len(time)))
            cubenew.ds[var] = xr.DataArray(result_arr, dims=['x', 'y', 'mid_date'],
                                           coords={'x': self.ds['x'], 'y': self.ds['y'], 'mid_date': time})
            cubenew.ds[var] = cubenew.ds[var].transpose('mid_date', 'y', 'x')
            cubenew.ds[var].attrs = {'standard_name': short_name[i], 'unit': 'm/y', 'long_name': long_name[i]}

        if result_quality is not None and 'Norm_residual' in result_quality:
            long_name = ['Residual from the inversion AX=Y, where Y is the displacement in the direction Est/West [m]',
                         'Residual from the regularisation term for the displacement in the direction Est/West [m]',
                         'Residual from the inversion AX=Y, where Y is the displacement in the direction North/South [m]',
                         'Residual from the regularisation term for the displacement in the direction North/South [m]']
            short_name = ['ResidualAXY_dx', 'ResidualRegu_dx', 'ResidualAXY_dy', 'ResidualRegu_dy']
            for k in range(0, 4):
                result_arr = np.array(
                    [result[i * self.ny + j]['NormR'][k] if result[i * self.ny + j]['NormR'].shape[
                                                                0] != 0 else np.nan for i in range(self.nx) for j in
                     range(self.ny)])
                result_arr = result_arr.reshape((self.nx, self.ny))
                cubenew.ds[short_name[k]] = xr.DataArray(result_arr, dims=['x', 'y'],
                                                         coords={'x': self.ds['x'], 'y': self.ds['y']})
                cubenew.ds[short_name[k]] = cubenew.ds[short_name[k]].transpose('y', 'x')
                cubenew.ds[short_name[k]].attrs = {'standard_name': short_name[k], 'unit': 'm',
                                                   'long_name': long_name[k]}
        if result_quality is not None and 'Error_propagation' in result_quality:
            long_name = [
                'Error propagated for the displacement in the direction Est/West',
                'Error propagated for  the displacement in the direction North/South [m]']
            short_name = ['Error_x', 'Error_y']
            for var in short_name:
                result_arr = np.array(
                    [result[i * self.ny + j][var] if result[i * self.ny + j][var].shape[
                                                         0] != 0 else np.nan for i in range(self.nx) for j in
                     range(self.ny)])
                result_arr = result_arr.reshape((self.nx, self.ny))
                cubenew.ds[short_name[k]] = xr.DataArray(result_arr, dims=['x', 'y'],
                                                         coords={'x': self.ds['x'], 'y': self.ds['y']})
                cubenew.ds[short_name[k]] = cubenew.ds[short_name[k]].transpose('y', 'x')
                cubenew.ds[short_name[k]].attrs = {'standard_name': short_name[k], 'unit': 'm',
                                                   'long_name': long_name[k]}

        del Non_null_el, long_name, result_arr
        cubenew.ds['x'] = self.ds['x']
        cubenew.ds['x'].attrs = {'standard_name': 'projection_x_coordinate', 'unit': 'm',
                                 'long_name': 'x coordinate of projection'}
        cubenew.ds['y'] = self.ds['y']
        cubenew.ds['y'].attrs = {'standard_name': 'projection_y_coordinate', 'unit': 'm',
                                 'long_name': 'y coordinate of projection'}
        cubenew.ds['mid_date'] = time.to_numpy()
        cubenew.ds['mid_date'].attrs = {'standard_name': 'central_date', 'unit': 'days',
                                        'description': 'the date in the middle of the two dates between which a velocity is computed'}
        cubenew.ds['grid_mapping'] = self.ds.proj4
        cubenew.ds.attrs = {'Conventions': 'CF-1.10', 'title': 'Ice velocity time series',
                            'institution': 'Université Grenoble Alpes',
                            'references': 'Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement observation networks. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-10.\n Charrier, L., Yan, Y., Koeniguer, E. C., Trouve, E., Mouginot, J., & Millan, R. (2022, June). Fusion of multi-temporal and multi-sensor ice velocity observations. In International Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences.',
                            'source': source, 'sensor': sensor, 'proj4': self.ds.proj4, 'author': 'L. Charrier',
                            'history': f'Created on the {date.today()}'}
        cubenew.nx = self.nx
        cubenew.ny = self.ny
        cubenew.nz = cubenew.ds['mid_date'].sizes['mid_date']
        cubenew.filename = filename

        if savepath is not None:  # save the dataset to a netcdf file
            cubenew.ds.to_netcdf(f'{savepath}/{filename}.nc')
            if verbose: print(f'Saved to {savepath}/{filename}.nc')

        return cubenew

    def write_result_TICO(self, result, source, sensor, filename='Time_series', savepath=None, result_quality=None,
                          verbose=False):
        """
        Write the result from TICOI, stored in result, in an xarray dataset matching the conventions CF-1.10
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.pdf
        units has been changed to unit, since it was producing an error while wirtting the netcdf file
        :param result: list of pd xarray, resulut from the TICOI method
        :param source: string, name of the source
        :param sensor: string, sensors which have been used
        :param filename: string, filename of file to saved
        :param savepath: string, path where to saved the file
        :return:
        """
        cubenew = cube_data_class()
        cubenew.ds['x'] = self.ds['x']
        cubenew.ds['x'].attrs = {'standard_name': 'projection_x_coordinate', 'unit': 'm',
                                 'long_name': 'x coordinate of projection'}
        cubenew.ds['y'] = self.ds['y']
        cubenew.ds['y'].attrs = {'standard_name': 'projection_y_coordinate', 'unit': 'm',
                                 'long_name': 'y coordinate of projection'}

        long_name = ['first date between which the velocity is estimated',
                     'second date between which the velocity is estimated',
                     'displacement in the East/West direction [m]', 'displacement in the North/South direction [d]',
                     'number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)) in the East/West direction',
                     'number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)) in the North/South direction']
        short_name = ['first_date', 'second_date', 'x_displacement', 'y_displacement', 'x_countx', 'x_county']
        unit = ['days', 'days', 'm', 'm', 'no unit', 'no unit']

        # Build cumulative displacement time series
        df_list = [reconstruct_Common_Ref(df[1], result_quality) for df in result]

        # List of the reference date, i.e. the first date of the cumulative displacement time series
        result_arr = np.array(
            [df_list[i]['Ref_date'][0] for i in range(len(df_list))]).reshape((self.nx, self.ny))
        cubenew.ds['reference_date'] = xr.DataArray(result_arr, dims=['x', 'y'],
                                                    coords={'x': self.ds['x'], 'y': self.ds['y']})
        cubenew.ds['reference_date'].attrs = {'standard_name': 'reference_date', 'unit': 'days',
                                              'description': 'first date of the cumulative displacement time series'}

        # Retrieve the list a second date in the whole data cube
        Second_date_list = list(set(list(itertools.chain.from_iterable([df['Second_date'].values for df in df_list]))))
        Second_date_list.sort()

        # reindex each dataframe according to the list of second date, so that each dataframe have the same temporal size
        df_list2 = []
        for i, df in enumerate(df_list):
            df.index = df['Second_date']
            df_list2.append(df.reindex(Second_date_list))
        del df_list

        # name of variable to store
        if result_quality is not None and 'X_contribution' in result_quality:
            variables = ['dx', 'dy', 'xcountx', 'xcounty']
        else:
            variables = ['dx', 'dy']

        warnings.filterwarnings("ignore",
                                category=UserWarning)  # to avoid the warning  UserWarning: Converting non-nanosecond precision datetime values to nanosecond precision. This behavior can eventually be relaxed in xarray, as it is an artifact from pandas which is now beginning to support non-nanosecond precision values. This warning is caused by passing non-nanosecond np.datetime64 or np.timedelta64 values to the DataArray or Variable constructor; it can be silenced by converting the values to nanosecond precision ahead of time.
        # Store each variable
        for i, var in enumerate(variables):
            result_arr = np.array(
                [df_list2[i][var] for i in range(len(df_list2))])
            result_arr = result_arr.reshape((self.nx, self.ny, len(Second_date_list)))
            cubenew.ds[var] = xr.DataArray(result_arr, dims=['x', 'y', 'second_date'],
                                           coords={'x': self.ds['x'], 'y': self.ds['y'],
                                                   'second_date': Second_date_list})
            cubenew.ds[var] = cubenew.ds[var].transpose('second_date', 'y', 'x')
            cubenew.ds[var].attrs = {'standard_name': short_name[i], 'unit': unit[i], 'long_name': long_name[i]}

        cubenew.ds['grid_mapping'] = self.ds.proj4
        cubenew.ds.attrs = {'Conventions': 'CF-1.10', 'title': 'Cumulative displacement time series',
                            'institution': 'Université Grenoble Alpes',
                            'references': 'Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement observation networks. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-10.\n Charrier, L., Yan, Y., Koeniguer, E. C., Trouve, E., Mouginot, J., & Millan, R. (2022, June). Fusion of multi-temporal and multi-sensor ice velocity observations. In International Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences.',
                            'source': source, 'sensor': sensor, 'proj4': self.ds.proj4, 'author': 'L. Charrier',
                            'history': f'Created on the {date.today()}'}
        cubenew.nx = self.nx
        cubenew.ny = self.ny
        cubenew.nz = cubenew.ds.dims['second_date']
        cubenew.filename = filename

        if savepath is not None:  # save the dataset to a netcdf file
            cubenew.ds.to_netcdf(f'{savepath}/{filename}.nc')
            if verbose: print(f'Saved to {savepath}/{filename}.nc')

        return cubenew

    def average_cube(self):
        '''

        :return: xr dataset, with vx_mean, the mean of vx and vy_mean the mean of vy
        '''
        ds_mean = xr.Dataset({})
        coords = {'y': self.ds.y, 'x': self.ds.x}
        ds_mean['vx_mean'] = xr.DataArray(self.ds['vx'].mean(dim='mid_date'), dims=['y', 'x'], coords=coords)
        ds_mean['vy_mean'] = xr.DataArray(self.ds['vy'].mean(dim='mid_date'), dims=['y', 'x'], coords=coords)
        return ds_mean

    def compute_heatmap_moving(self, points_heatmap, variable='vv', method_interp='linear',
                               verbose=False, freq='MS', method='mean'):
        """
        Compute a heatmap of the average monthly velocity, average all the velocities which are overlapping a given month

        :param points_heatmap: pd dataframe, Points where the heatmap is to be computed
        :param variable: str, What variable is to be computed ('vx', 'vy' or 'vv')
        :param method_interp:str,  Interpolation method used to determine the value at a specified point from the discrete velocities datas
        :param freq: str, frequency used in the pandas.date_range function (default: 'MS' every first day of the month)
        :param method: str, 'mean' or 'median'

        :return: pandas DataFrame, heatmap values where each line corresponds to a date and each row to a point of the line
        """

        date1 = self.date1_()
        date2 = self.date2_()
        # Create a DateTimeIndex range spanning from the minimum date to the maximum date
        date_range = pd.date_range(np.nanmin(date1), np.nanmax(date2), freq=freq)  # 'MS' for start of each month
        data = np.column_stack((date1, date2))  # Combine date1 and date2 into a single 2D array
        # Sort data according to the first date
        data = np.ma.array(sorted(data, key=lambda date: date[0]))  # sort according to the first date

        # Find the index of the dates that have to be averaged, to get the heatmap
        # Each value of the heatmap corresponds to an average of all the velocities which are overlapping a given period
        save_line = [[] for _ in range(len(date_range) - 1)]
        for i_date, date in enumerate(date_range[:-1]):
            i = 0
            while i < data.shape[0] and date_range[i_date + 1] >= data[i, 0]:
                if date_range[i_date] <= data[i, 1]:
                    save_line[i_date].append(i)
                i += 1
        interval_output = pd.Series(
            [(date_range[k + 1] - date_range[k]) / np.timedelta64(1, 'D') for k in range(date_range.shape[0] - 1)])
        dates_c = date_range[1:] - pd.to_timedelta((interval_output / 2).astype('int'), 'D')
        del interval_output, date_range, data

        def data_temporalpoint(k, points_heatmap):
            '''Get the data at a given spatial point contained in points_heatmap'''
            geopoint = points_heatmap['geometry'].iloc[
                k]  # Return a point at the specified distance along a linear geometric object. # True -> interpretate k/n as fraction and not meters
            i, j = geopoint.x, geopoint.y
            if verbose: print('i,j', i, j)
            if variable == 'vv':
                v = np.sqrt(
                    self.ds['vx'].interp(x=i, y=j, method=method_interp).load() ** 2 + self.ds['vy'].interp(x=i, y=j,
                                                                                                            method="linear").load() ** 2)
            elif variable == 'vx' or variable == 'vy':
                v = self.ds[variable].interp(x=i, y=j, method=method_interp).load()
            data = np.array([date1, date2, v.values], dtype=object).T
            data = np.ma.array(sorted(data, key=lambda date: date[0]))  # sort according to the first date
            return data[:, 2]

        for k in range(len(points_heatmap)):
            if verbose: print('k', k)
            data = data_temporalpoint(k, points_heatmap)
            vvmasked = np.ma.masked_invalid(np.ma.array(data, dtype='float'))
            if method == 'mean':
                vvmean = [np.ma.mean(vvmasked[lines]) for lines in save_line]
            elif method == 'median':
                vvmean = [np.ma.median(vvmasked[lines]) for lines in save_line]

            vvdf = pd.DataFrame(vvmean, index=dates_c, columns=[points_heatmap['distance'].iloc[k] / 1000])
            if k > 0:
                line_df_vv = pd.concat([line_df_vv, vvdf], join='inner', axis=1)
            else:
                line_df_vv = vvdf
        return line_df_vv

    # @jit(nopython=True)
    def NCVV(self):
        '''Return the Normalized Coherence Vector Velocity '''
        return np.array([np.sqrt(np.nansum((self.ds['vx'].isel(x=i, y=j) / np.sqrt(
            self.ds['vx'].isel(x=i, y=j) ** 2 + self.ds['vy'].isel(x=i, y=j) ** 2))) ** 2 + np.nansum((self.ds[
                                                                                                           'vy'].isel(
            x=i, y=j) / np.sqrt(self.ds['vx'].isel(x=i, y=j) ** 2 + self.ds['vy'].isel(x=i, y=j) ** 2))) ** 2) / self.nz
                         for i in
                         range(self.nx) for j in range(self.ny)]).reshape(self.nx, self.ny)
