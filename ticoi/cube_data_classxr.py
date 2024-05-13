"""
Class object to store and manipulate velocity observation data

Author : Laurane Charrier Reference: Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021).
Extraction of velocity time series with an optimal temporal sampling from displacement observation networks. IEEE
Transactions on Geoscience and Remote Sensing. Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R.,
& Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations. ISPRS annals of the
photogrammetry, remote sensing and spatial information sciences, 3, 311-318."""

import os
from ticoi.mjd2date import mjd2date  # /ST_RELEASE/UTILITIES/PYTHON/mjd2date.py
import pandas as pd
import dask
from pyproj import Proj, Transformer, CRS
import rasterio.enums
from datetime import date
from ticoi.interpolation_functions import reconstruct_common_ref
import itertools
import warnings
import time
from dask.diagnostics import ProgressBar
from ticoi.inversion_functions import construction_dates_range_np
from ticoi.filtering_functions import *
from typing import Union


# %% ======================================================================== #
#                              CUBE DATA CLASS                                #
# =========================================================================%% #

class cube_data_class:

    def __init__(self):
        self.filedir = ''
        self.filename = ''
        self.nx = 250
        self.ny = 250
        self.nz = 0
        self.author = ''
        self.source = ''
        self.ds = xr.Dataset({})

    def update_dimension(self):
        """
        Update the variable the attribute corresponding to cube dimensions: nx, ny, and nz

        """
        self.nx = self.ds['x'].sizes['x']
        self.ny = self.ds['y'].sizes['y']
        self.nz = self.ds['mid_date'].sizes['mid_date']

    def subset(self, proj: str, subset: list):

        """
        Directly crop the dataset according to 4 coordinates.
        
        :param proj: EPSG system of the coordinates given in subset
        :param subset: A list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
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

    def buffer(self, proj: str, buffer: list):

        """
        Directly crop the dataset around a given pixel, according to a given buffer
        
        :param proj: EPSG system of the coordinates given in subset
        :param buffer:  A list of 3 float, the first two
        are the longitude and the latitude of the central point, the last is the buffer size

        """

        if CRS(self.ds.proj4) != CRS(proj):  # Convert the coordinates from proj to self.ds.proj4
            transformer = Transformer.from_crs(CRS(proj), CRS(self.ds.proj4))
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

    def determine_optimal_chunk_size(self, variable_name: str = "vx", x_dim: str = "x", y_dim: str = "y",
                                     time_dim_name: str = 'mid_date', verbose: bool = False) -> (int, int, int):

        """
        A function to determine the optimal chunk size for a given time series array based on its size.

        :param variable_name: Name of the variable containing the time series array (default is "vx")
        :param x_dim: Name of the x dimension in the array (default is "x")
        :param y_dim: Name of the y dimension in the array (default is "y")
        :param time_dim_name: Name of the z dimension within the original dataset self.ds (default is "mid_date")
        :param verbose: Boolean flag to control verbosity of output (default is True)

        :return tc: Chunk size along the time dimension
        :return yc: Chunk size along the y dimension
        :return xc: Chunk size along the x dimension
        """

        if verbose:
            print("Dask chunk size:")
        # set chunk size to 5 MB if single time series array < 1 MB in size, else increase to max of 1 GB chunk sizes.
        time_series_array_size = (
            self.ds[variable_name]
            .sel(
                {
                    x_dim: self.ds[variable_name][x_dim].values[0],
                    y_dim: self.ds[variable_name][y_dim].values[0],
                }
            )
            .nbytes
        )
        mb = 1048576
        if time_series_array_size < 1e6:
            chunk_size_limit = 50 * mb
        elif time_series_array_size < 1e7:
            chunk_size_limit = 100 * mb
        elif time_series_array_size < 1e8:
            chunk_size_limit = 200 * mb
        else:

            chunk_size_limit = 1000 * mb
        time_axis = self.ds[variable_name].dims.index(time_dim_name)
        x_axis = self.ds[variable_name].dims.index(x_dim)
        y_axis = self.ds[variable_name].dims.index(y_dim)
        axis_sizes = {i: -1 if i == time_axis else "auto" for i in range(3)}
        arr = self.ds[variable_name].data.rechunk(
            axis_sizes, block_size_limit=chunk_size_limit, balance=True
        )
        tc, yc, xc = arr.chunks[time_axis][0], arr.chunks[y_axis][0], arr.chunks[x_axis][0]
        chunksize = self.ds[variable_name][:tc, :yc, :xc].nbytes / 1e6
        if verbose:
            print("Chunk shape:", "(" + ",".join([str(x) for x in [tc, yc, xc]]) + ")")
            print(
                "Chunk size:",
                self.ds[variable_name][:tc, :yc, :xc].nbytes,
                "(" + str(round(chunksize, 1)) + "MB)",
            )
        return tc, yc, xc

    # %% ==================================================================== #
    #                         CUBE LOADING METHODS                            #
    # =====================================================================%% #

    def load_itslive(self, filepath: str, conf: bool = False, subset: list | None = None, buffer: list | None = None,
                     pick_date: list | None = None,
                     pick_sensor: list | None = None, pick_temp_bas: list | None = None, proj: str = 'EPSG:4326',
                     verbose: bool = False):

        """
        Load a cube dataset written by ITS_LIVE.
        
        :param filepath: Filepath of the dataset
        :param conf: If True convert the error in confidence between 0 and 1 (default is False)
        :param subset: A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax] (default is None)
        :param buffer: A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size (default is None)
        :param pick_date: A list of 2 string yyyy-mm-dd, pick the data between these two date (default is None)
        :param pick_sensor: A list of strings, pick only the corresponding sensors (default is None)
        :param pick_temp_bas: A list of 2 integer, pick only the data which have a temporal baseline between these two integers (default is None)
        :param proj: Projection of the buffer or subset which is given (default is 'EPSG:4326')
        :param verbose: Print information throughout the process (default is False)
        """

        if verbose:
            print(filepath)

        self.filedir = os.path.dirname(filepath)  # Path were is stored the netcdf file
        self.filename = os.path.basename(filepath)  # Name of the netcdf file
        self.ds = self.ds.assign_attrs({'proj4': self.ds['mapping'].proj4text})
        self.author = self.ds.author.split(', a NASA')[0]
        self.source = self.ds.url

        if subset is not None:  # Crop according to 4 coordinates
            self.subset(proj, subset)
        elif buffer is not None:  # Crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)
        if pick_date is not None:
            self.ds = self.ds.where(((self.ds['acquisition_date_img1'] >= np.datetime64(pick_date[0])) & (
                    self.ds['acquisition_date_img2'] <= np.datetime64(pick_date[1]))).compute(), drop=True)

        self.update_dimension()  # update self.nx,self.ny,self.nz

        if conf:
            minconfx = np.nanmin(self.ds['vx_error'].values[:])
            maxconfx = np.nanmax(self.ds['vx_error'].values[:])
            minconfy = np.nanmin(self.ds['vy_error'].values[:])
            maxconfy = np.nanmax(self.ds['vy_error'].values[:])

        date1 = np.array([np.datetime64(date_str, 'D') for date_str in self.ds['acquisition_date_img1'].values])
        date2 = np.array([np.datetime64(date_str, 'D') for date_str in self.ds['acquisition_date_img2'].values])
        # np.char.strip is used to remove the null character ('�') from each elemen and np.core.defchararray.add to
        # concatenate array of different types
        sensor = np.core.defchararray.add(np.char.strip(self.ds['mission_img1'].values.astype(str), '�'),
                                          np.char.strip(self.ds['satellite_img1'].values.astype(str), '�')
                                          ).astype('U10')
        sensor[sensor == 'L7'] = 'Landsat-7'
        sensor[sensor == 'L8'] = 'Landsat-8'
        sensor[sensor == 'L9'] = 'Landsat-9'
        sensor[np.isin(sensor, ['S1A', 'S1B'])] = 'Sentinel-1'
        sensor[np.isin(sensor, ['S2A', 'S2B'])] = 'Sentinel-2'

        if conf:  # Normalize the error between 0 and 1, and convert error in confidence
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

        # self.ds = self.ds.unify_chunks()  # to avoid error ValueError: Object has inconsistent chunks along
        # dimension mid_date. This can be fixed by calling unify_chunks(). Create new variable and chunk them
        self.ds['sensor'] = xr.DataArray(sensor, dims='mid_date').chunk(chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['date1'] = xr.DataArray(date1, dims='mid_date').chunk(chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['date2'] = xr.DataArray(date2, dims='mid_date').chunk(chunks=self.ds.chunks['mid_date'])
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

    def load_millan(self, filepath: str, conf: bool = False, subset: list | None = None, buffer: list | None = None,
                    pick_date: list | None = None,
                    pick_sensor: list | None = None, pick_temp_bas: list | None = None, proj: str = 'EPSG:4326',
                    verbose: bool = False):

        """
        Load a cube dataset written by R. Millan et al.

        :param filepath: Filepath of the dataset
        :param conf: If True convert the error in confidence between 0 and 1 (default is False)
        :param subset: A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax] (default is None)
        :param buffer: A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size (default is None)
        :param pick_date: A list of 2 string yyyy-mm-dd, pick the data between these two date (default is None)
        :param pick_sensor: A list of strings, pick only the corresponding sensors (default is None)
        :param pick_temp_bas: A list of 2 integer, pick only the data which have a temporal baseline between these two integers (default is None)
        :param proj: Projection of the buffer or subset which is given (default is 'EPSG:4326')
        :param verbose: Print information throughout the process (default is False)
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

        self.update_dimension()

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

    def load_ducasse(self, filepath: str, conf: bool = False, subset: list | None = None, buffer: list | None = None,
                     pick_date: list | None = None,
                     pick_sensor: list | None = None, pick_temp_bas: list | None = None, proj: str = 'EPSG:4326',
                     verbose: bool = False):

        """
        Load a cube dataset written by E. Ducasse et al.
        
        :param filepath: Filepath of the dataset
        :param conf: If True convert the error in confidence between 0 and 1 (default is False)
        :param subset: A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax] (default is None)
        :param buffer: A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size (default is None)
        :param pick_date: A list of 2 string yyyy-mm-dd, pick the data between these two date (default is None)
        :param pick_sensor: A list of strings, pick only the corresponding sensors (default is None)
        :param pick_temp_bas: A list of 2 integer, pick only the data which have a temporal baseline between these two integers (default is None)
        :param proj: Projection of the buffer or subset which is given (default is 'EPSG:4326')
        :param verbose: Print information throughout the process (default is False)
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

        self.update_dimension()  # update self.nx,self.ny,self.nz

        # Drop variables not in the specified list
        variables_to_keep = ['vx', 'vy', 'mid_date', 'x', 'y', 'date1', 'date2']
        self.ds = self.ds.drop_vars([var for var in self.ds.variables if var not in variables_to_keep])
        self.ds = self.ds.transpose('mid_date', 'y', 'x')

        # Store the variable in xarray dataset
        self.ds['sensor'] = xr.DataArray(['Pleiades'] * self.nz, dims='mid_date').chunk(
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

        # Set errors equal to one (no information on the error here)
        self.ds['errorx'] = xr.DataArray(np.ones(self.ds['mid_date'].size), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['errory'] = xr.DataArray(np.ones(self.ds['mid_date'].size), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])

    def load_charrier(self, filepath: str, conf: bool = False, subset: list | None = None, buffer: list | None = None,
                      pick_date: list | None = None,
                      pick_sensor: list | None = None, pick_temp_bas: list | None = None, proj: str = 'EPSG:4326',
                      verbose: bool = False):

        """
        Load a cube dataset written by L.Charrier et al.

        :param filepath: Filepath of the dataset
        :param conf: If True convert the error in confidence between 0 and 1 (default is False)
        :param subset: A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax] (default is None)
        :param buffer: A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size (default is None)
        :param pick_date: A list of 2 string yyyy-mm-dd, pick the data between these two date (default is None)
        :param pick_sensor: A list of strings, pick only the corresponding sensors (default is None)
        :param pick_temp_bas: A list of 2 integer, pick only the data which have a temporal baseline between these two integers (default is None)
        :param proj: Projection of the buffer or subset which is given (default is 'EPSG:4326')
        :param verbose: Print information throughout the process (default is False)
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

        self.update_dimension()

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

    def load(self, filepath: str, chunks: dict | str | int = {}, conf: bool = False, subset: str | None = None,
             buffer: str | None = None, pick_date: str | None = None,
             pick_sensor: str | None = None, pick_temp_bas: str | None = None, proj: str = 'EPSG:4326',
             verbose: bool = False):

        """        
        Load a cube dataset from a file in format netcdf (.nc) or zarr. The data are directly stored within the present object.
        
        :param filepath: Filepath of the dataset
        :param chunks: Dictionary with the size of chunks for each dimension, if chunks=-1 loads the dataset with dask using a single chunk for all arrays. 
                       chunks={} loads the dataset with dask using engine preferred chunks if exposed by the backend, otherwise with a single chunk for all arrays, 
                       chunks='auto' will use dask auto chunking taking into account the engine preferred chunks.
        :param conf: If True the error is converted in confidence between 0 and 1 (default is False)
        :param subset: A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax] (default is None)
        :param buffer: A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size (default is None)
        :param pick_date: A list of 2 string yyyy-mm-dd, pick the data between these two date (default is None)
        :param pick_sensor: A list of strings, pick only the corresponding sensors (default is None)
        :param pick_temp_bas: A list of 2 integer, pick only the data which have a temporal baseline between these two integers (default is None)
        :param proj: Projection of the buffer or subset which is given (default is 'EPSG:4326')
        :param verbose: Print information throughout the process (default is False)
        """

        time_dim_name = {
            "ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)": 'mid_date',
            "J. Mouginot, R.Millan, A.Derkacheva": 'z',
            "J. Mouginot, R.Millan, A.Derkacheva_aligned": 'mid_date',
            "L. Charrier, L. Guo": 'mid_date',
            "L. Charrier": 'mid_date',
            "E. Ducasse": 'time',
            "S. Leinss, L. Charrier": 'mid_date'
        }

        self.__init__()
        with dask.config.set(
                **{"array.slicing.split_large_chunks": False}
        ):  # To avoid creating the large chunks
            if filepath.split(".")[-1] == "nc":
                try:
                    self.ds = xr.open_dataset(filepath, engine="netcdf4", chunks=chunks)
                except (
                        NotImplementedError):  # Can not use auto rechunking with object dtype. We are unable to estimate the size in bytes of object data
                    chunks = {}
                    self.ds = xr.open_dataset(
                        filepath, engine="netcdf4", chunks=chunks)  # set no chunks

                if "Author" in self.ds.attrs:  # Uniformization of the attribute Author to author
                    self.ds.attrs["author"] = self.ds.attrs.pop("Author")

                if chunks == {}:  # Rechunk with optimal chunk size
                    tc, yc, xc = self.determine_optimal_chunk_size(variable_name="vx", x_dim="x", y_dim="y", 
                                                                   time_dim_name=time_dim_name[self.ds.author], verbose=True)
                    self.ds = self.ds.chunk({time_dim_name[self.ds.author]: tc, "x": xc, "y": yc})

            elif filepath.split(".")[-1] == "zarr":
                if chunks == {}:
                    chunks = "auto"  # Change the default value to auto

                self.ds = xr.open_dataset(filepath, decode_timedelta=False, engine="zarr",
                                          consolidated=True, chunks=chunks)

        if verbose:
            print("file open")

        dico_load = {
            "ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)": self.load_itslive,
            "J. Mouginot, R.Millan, A.Derkacheva": self.load_millan,
            "J. Mouginot, R.Millan, A.Derkacheva_aligned": self.load_charrier,
            "L. Charrier, L. Guo": self.load_charrier,
            "L. Charrier": self.load_charrier,
            "E. Ducasse": self.load_ducasse,
            "S. Leinss, L. Charrier": self.load_charrier,
        }
        dico_load[self.ds.author](filepath, pick_date=pick_date, subset=subset, conf=conf, pick_sensor=pick_sensor,
                                  pick_temp_bas=pick_temp_bas, buffer=buffer, proj=proj
                                  )
        # Reorder the coordinates to keep the consistency
        self.ds = self.ds.copy().sortby("mid_date").transpose("x", "y", "mid_date")
        self.standardize_cube_for_processing()
        # self.ds = self.ds.persist()
        # if there is chunks in time, set no chunks

        # if self.ds['mid_date'].dtype == ('<M8[ns]'): #if the dates are given in ns, convert them to days
        #     self.ds['mid_date'] = self.ds['date2'].astype('datetime64[D]')
        #     self.ds['date1'] = self.ds['date1'].astype('datetime64[D]')
        #     self.ds['date2'] = self.ds['date2'].astype('datetime64[D]')

        if verbose:
            print(self.ds.author)

    def standardize_cube_for_processing(self):
        """
        Prepare the xarray dataset for the processing: transpose the dimension, add a varibale temporal_baseline, errors if they do not exist
        """
        if self.ds.chunksizes['mid_date'] != (self.nz,):
            self.ds = self.ds.chunk({'mid_date': self.nz})
        # create a variable for temporal_baseline,be
        self.ds["temporal_baseline"] = xr.DataArray((self.ds["date2"] - self.ds["date1"]).dt.days.values,
                                                    dims='mid_date')
        if "errorx" not in self.ds.variables:
            self.ds["errorx"] = (
                ("mid_date",
                 np.ones((len(self.ds["mid_date"])))))
            self.ds["errory"] = (
                ("mid_date",
                 np.ones((len(self.ds["mid_date"])))))


    # %% ==================================================================== #
    #                                 ACCESSORS                               #
    # =====================================================================%% #

    def sensor_(self) -> list:
        """

        :return: list of sensor
        """
        return self.ds['sensor'].values.tolist()

    def source_(self) -> list:
        """

        :return: list of source
        """
        return self.ds['source'].values.tolist()

    def temp_base_(self, return_list: bool = True, format_date: str = 'float') -> list | np.ndarray:
        """
        Get the temporal baseline of the dataset
        :param return_list: bool, if True return of a list of date, else return a np array
        :param format_date: 'float' or 'D' format of the date as output
        :return: list or np array of temporal baselines
        """
        if format_date == 'D':
            temp = (self.ds['date2'] - self.ds['date1'])
        elif format_date == 'float':
            # temp = (self.ds['date2'].values-self.ds['date1'].values).astype('timedelta64[D]'))/ np.timedelta64(1, 'D')
            temp = ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))
        else:
            raise NameError('Please enter format as float or D')
        if return_list:
            return temp.values.tolist()
        else:
            return temp.values

    def date1_(self):
        """

        :return: np array of date1
        """
        return np.asarray(self.ds['date1']).astype('datetime64[D]')

    def date2_(self):
        """

         :return: np array of date2
         """
        return np.asarray(self.ds['date2']).astype('datetime64[D]')

    def datec_(self):
        """

         :return: np array of central date
         """
        return (self.date1_() + self.temp_base_(return_list=False, format_date='D') // 2).astype('datetime64[D]')

    def vv_(self):
        """

         :return: np array of velocity magnitude
         """
        return np.sqrt(self.ds['vx'] ** 2 + self.ds['vy'] ** 2)

    # %% ==================================================================== #
    #                         PIXEL LOADING METHODS                           #
    # =====================================================================%% #

    def convert_coordinates(self, i: int | float, j: int | float, proj: str, verbose: bool = False) -> (float, float):
        """
        Convert the coordinate (i,j) which are in projection proj, to projection of the cube dataset
        :param i: pixel coordinate for x
        :param j: pixel coordinate for y
        :param proj: projection, e.g., EPSG:4326
        :param verbose: if True, print text
        :return: converted i,j
        """
        # Convert coordinates if needed
        if proj == 'EPSG:4326':
            myproj = Proj(self.ds.proj4)
            i, j = myproj(i, j)
            if verbose: print(f'Converted to projection {self.ds.proj4}: {i, j}')
        else:
            if CRS(self.ds.proj4) != CRS(proj):
                transformer = Transformer.from_crs(CRS(proj), CRS(self.ds.proj4))
                i, j = transformer.transform(i, j)
                if verbose: print(f'Converted to projection {self.ds.proj4}: {i, j}')
        return i, j


    def load_pixel(self, i:int|float, j:int|float, unit:int=365, regu:int|str=1, coef:int=1, flags:None | xr.Dataset=None, solver:str='LSMR', interp:str='nearest',proj:str='EPSG:4326', visual:bool=False, rolling_mean:np.array=None, verbose=False):
        """

        :param i: pixel coordinate for x
        :param j: pixel coordinate for y
        :param unit: 365 if the unit is m/y and 1 if the unit is m/day
        :param regu: type of regularization
        :param coef: coef of the regularization
        :param flags: if not None, the values of the coefficient used for stable areas, surge glacier and non surge glacier
        :param solver: solver for the inversion
        :param interp: interpolation to get the value of the given pixel
        :param proj: projection of i and j
        :param visual: if the user want to visualize soem figures
        :param rolling_mean: filtered cube using a spatio-temporal filter
        :return: data, a list 2 elements : the first one is np.ndarray with the observed
        :return: mean, a list with average vx and vy if solver=LSMR_ini, but the regularization do not require an apriori on the acceleration
        :return: dates_range: dates between which the displacements will be inverted
        """

        # variables to keep
        var_to_keep = (
            ["date1", "date2", "vx", "vy", "errorx", "errory", "temporal_baseline"]
            if not visual
            else ["date1", "date2", "vx", "vy", "errorx", "errory", "temporal_baseline", "sensor", "source"]
        )

        if proj == 'int':
            data = self.ds.isel(x=i, y=j)[var_to_keep]
        else:
            i, j = self.convert_coordinates(i, j, proj=proj)
            # Interpolate only necessary variables and drop NaN values
            if interp == 'nearest':
                # 74.3 ms ± 1.33 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
                data = self.ds.sel(x=i, y=j, method='nearest')[var_to_keep]
                data = data.dropna(dim='mid_date')
            else:
                data = self.ds.interp(x=i, y=j, method=interp)[var_to_keep].dropna(
                    dim='mid_date')  # 282 ms ± 12.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        if flags is not None:
            if isinstance(regu, dict) and isinstance(coef, dict):
                flag = np.round(flags['flags'].sel(x=i, y=j, method='nearest').values)
                regu = regu[flag]
                coef = coef[flag]
            else:
                raise ValueError("regu must be a dict if assign_flag is True!")

        data_dates = data[['date1', 'date2']].to_array().values.T
        if data_dates.dtype == '<M8[ns]':  # convert to days if needed
            data_dates = data_dates.astype('datetime64[D]')

        if solver == 'LSMR_ini' or regu == '1accelnotnull' or regu == 'directionxy':
            if len(rolling_mean.sizes) == 3:  # if regu == 1accelnotnul, rolling_mean have a time dimesion
                # Load rolling mean for the given pixel, only on the dates available
                dates_range = construction_dates_range_np(
                    data_dates)  # 652 µs ± 3.24 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
                mean = \
                    rolling_mean.sel(mid_date=dates_range[:-1] + np.diff(dates_range) // 2, x=i, y=j, method='nearest')[
                        ['vx_filt', 'vy_filt']]
                mean = [mean[i].values / unit for i in
                        ['vx_filt', 'vy_filt']]  # convert it to m/day
            else:  # elif solver= LSMR_ini, rolling_mean is an average in time per pixel
                mean = rolling_mean.sel(x=i, y=j, method='nearest')[['vx', 'vy']]
                mean = [mean[i].values / unit for i in ['vx', 'vy']]  # convert it to m/day
                dates_range = None

        else:  # if there is no apriori and no initialization
            mean = None
            dates_range = None

        # data_values is composed of vx, vy, errorx, errory, temporal baseline
        if visual:
            data_str = data[['sensor', 'source']].to_array().values.T
            data_values = data.drop_vars(['date1', 'date2', 'sensor', 'source']).to_array().values.T
            data = [data_dates, data_values, data_str]
        else:
            data_values = data.drop_vars(['date1', 'date2']).to_array().values.T
            data = [data_dates, data_values]

        if flags is not None:
            return data, mean, dates_range, regu, coef
        else:
            return data, mean, dates_range

    # %% ==================================================================== #
    #                             CUBE PROCESSING                             #
    # =====================================================================%% #

    
    def delete_outliers(self, delete_outliers:str|float,flags:None | xr.Dataset=None):
        """
        Delete outliers according to a certain criterium
        :param delete_outliers: If int delete all velocities which a quality indicator higher than delete_outliers, if median_filter delete outliers that an angle 45° away from the average vector
        :param flags:

        Returns: nothing, but modify self

        """
        
        if isinstance(delete_outliers, int):
            self.ds = self.ds.where((self.ds["errorx"] < delete_outliers) & (self.ds["errory"] < delete_outliers)
            )
        else:
            # inlier_mask = median_angle_filt_np(self.ds["vx"].values, self.ds["vy"].values, angle_thres=45)
            axis = self.ds['vx'].dims.index('mid_date')
            inlier_mask = dask_filt_warpper(self.ds["vx"], self.ds["vy"], filt_method=delete_outliers, axis=axis)
            
            if flags is not None:
                if delete_outliers != 'vvc_angle':
                    flag = flags['flags'].values if flags['flags'].shape[0] == self.nx else flags['flags'].values.T
                    flag_condition = (flag == 0)
                    flag_condition = np.expand_dims(flag_condition, axis=axis)
                    inlier_mask = np.logical_or(inlier_mask, flag_condition)
            
            inlier_flag = xr.DataArray(inlier_mask, dims=self.ds['vx'].dims)
            
            for var in ["vx", "vy"]:
                self.ds[var] = self.ds[var].where(inlier_flag)
                
        self.ds = self.ds.persist()


    def filter_cube(self, i: int | float | None = None, j: int | float | None = None, smooth_method: str = "gaussian",
                    s_win: int = 3, t_win: int = 90, sigma: int = 3,
                    order: int = 3, unit: int = 365, delete_outliers: str | float | None = None,
                    flags: None | xr.Dataset = None, regu: int | str = 1, solver: str = 'LSMR_ini',
                    proj: str = "EPSG:4326", velo_or_disp: str = "velo", verbose: bool = False) -> xr.Dataset:

        """
        Filter the original data with a spatio-temporal kernel
        
        :param i: x-coordinate of the considered pixel, if None, compute over the whole dataset (default is None)
        :param j: y-coordinate of the considered pixel, if None, compute over the whole dataset (default is None)
        :param smooth_method: Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol') (default is 'gaussian')
        :param s_win: Size of the spatial window (default is 3)
        :param t_win: Time window size for 'ewma' smoothing (default is 90)
        :param sigma: Standard deviation for 'gaussian' filter (default is 3)
        :param order: Order of the smoothing function (default is 3)
        :param unit: 365 if the unit is m/y, 1 if the unit is m/d (default is 365)
        :param delete_outliers: If int delete all velocities which a quality indicator higher than delete_outliers (defau)
        :param regu: Regularisation of the solver (default is 1)
        :param solver: solver used to invert the system
        :param proj: EPSG of i,j projection (default is 'EPSG:4326')
        :param velo_or_disp: 'disp' or 'velo' to indicate the type of the observations : 'disp' mean that self contain displacements values and 'velo' mean it contains velocity (default is 'velo')
        :param verbose: Print information throughout the process (default is False)
        
        :return: filtered dataset
        """

        def loop_rolling(da_arr: xr.DataArray, mid_dates: xr.DataArray, date_range: np.ndarray,
                         smooth_method: str = "gaussian", s_win: int = 3, t_win: int = 90, sigma: int = 3, order=3,
                         baseline: xr.DataArray | None = None, verbose: bool = False) -> (
                np.ndarray, np.ndarray):

            """
            A function to calculate spatial mean, resample data, and calculate exponential smoothed velocity.

            :param da_arr: Original data
            :param mid_dates: Time labels for input array, in datetime format, should have same length as array, central date of the data
            :param date_range: 
            :param smooth_method: Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol') (default is 'gaussian')
            :param s_win: Window size for spatial average (default is 3)
            :param t_win: Time window size for 'ewma' smoothing (default is 90)
            :param sigma: Standard deviation for 'gaussian' filter (default is 3)
            :param order: Order of the smoothing function (default is 3)
            :param baseline:
            :param time_axis: Optional parameter for time axis (default is 2)
            :param verbose: Print information throughout the process (default is False)

            :return: exponential smoothed velocity
            :return: observed dates
            """

            from dask.array.lib.stride_tricks import sliding_window_view

            # Compute the dates of the estimated displacements time series
            date_out = date_range[:-1] + np.diff(date_range) // 2
            if verbose: start = time.time()

            if baseline is not None:
                baseline = baseline.compute()
                idx = np.where(baseline < 700)
                t_thres = 120
                idx = np.where(baseline < t_thres )
                while len(idx[0]) < 3 * len(date_out):
                    t_thres += 30
                    idx = np.where(baseline < t_thres )
                mid_dates = mid_dates.isel(mid_date=idx[0])
                da_arr = da_arr.isel(mid_date=idx[0])

            # find the time axis for dask processing
            time_axis = self.ds['vx'].dims.index('mid_date')
            # Apply the selected kernel in time
            if verbose:
                with ProgressBar():  # Plot a progress bar
                    filtered_in_time = dask_smooth_wrapper(da_arr.data, mid_dates, t_out=date_out,
                                                           smooth_method=smooth_method,
                                                           sigma=sigma, t_win=t_win, order=order,
                                                           axis=time_axis).compute()
            else:
                filtered_in_time = dask_smooth_wrapper(da_arr.data, mid_dates, t_out=date_out,
                                                       smooth_method=smooth_method,
                                                       sigma=sigma, t_win=t_win, order=order, axis=time_axis).compute()

            if verbose: print(f'Smoothing observations took {round((time.time() - start), 1)} s')

            # Spatial average

            if np.min([da_arr['x'].size,da_arr['y'].size]) > s_win :# The spatial average is performed only if the size of the cube is larger than s_win, the spatial window
                
                spatial_axis = tuple(i for i in range(3) if i != time_axis)
                pad_widths = tuple((s_win // 2, s_win // 2) if i != time_axis else (0, 0) for i in range(3))
                spatial_mean = da.nanmean(sliding_window_view(filtered_in_time, (s_win, s_win), axis=spatial_axis), axis=(-1, -2))
                spatial_mean = da.pad(
                    spatial_mean,
                    pad_widths,
                    mode="edge",
                )
            else:
                spatial_mean = filtered_in_time

            # chunk size of spatial mean becomes after the pading: ((1, 9, 1, 1), (1, 20, 2, 1), (61366,))

            return spatial_mean.compute(), np.unique(date_out)

        if i is not None and j is not None:  # Crop the cube dataset around a given pixel
            i, j = self.convert_coordinates(i, j, proj=proj, verbose=verbose)
            if verbose: print(f"Clipping dataset to individual pixel: (x, y) = ({i},{j})")
            buffer = (s_win + 2) * (self.ds["x"][1] - self.ds["x"][0])
            self.buffer(self.ds.proj4, [i, j, buffer])
            self.ds = self.ds.unify_chunks()

        # the rolling smooth should be carried on velocity, while we need displacement during inversion
        if velo_or_disp == "disp":  # to provide velocity values
            self.ds["vx"] = self.ds["vx"] / self.ds["temporal_baseline"] * unit
            self.ds["vy"] = self.ds["vy"] / self.ds["temporal_baseline"] * unit
        
        if flags is not None:
            flags = flags.load()
            if isinstance(regu, dict):
                regu = list(regu.values())
            else:
                raise ValueError("regu must be a dict if assign_flag is True!")
        else:
            if isinstance(regu, int):  # if regu is an integer
                regu = [regu]
            elif isinstance(regu, str):  # if regu is a string
                regu = list(regu.split())
        
        start = time.time()
        if delete_outliers is not None: 
            self.delete_outliers(delete_outliers=delete_outliers, flags=flags)
        print(f'Delete outlier took {round((time.time() - start), 1)} s')

        if ("1accelnotnull" in regu or "directionxy" in regu):

            date_range = np.sort(np.unique(np.concatenate((self.ds['date1'].values, self.ds['date2'].values), axis=0)))
            if verbose: start = time.time()
            vx_filtered, dates_uniq = loop_rolling(
                self.ds["vx"],
                self.ds["mid_date"],
                date_range,
                smooth_method=smooth_method,
                s_win=s_win,
                t_win=t_win,
                sigma=sigma,
                order=order,
                baseline=self.ds["temporal_baseline"]
            )
            vy_filtered, dates_uniq = loop_rolling(
                self.ds["vy"],
                self.ds["mid_date"],
                date_range,
                smooth_method=smooth_method,
                s_win=s_win,
                t_win=t_win,
                sigma=sigma,
                order=order,
                baseline=self.ds["temporal_baseline"]
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

            if verbose: print(
                "Calculating smoothing mean of the observations completed in {:.2f} seconds".format(
                    time.time() - start
                )
            )
        elif solver == 'LSMR_ini':  # The initialization is based on the averaged velocity over the period, for every pixel
            obs_filt = self.ds[['vx', 'vy']].mean(dim='mid_date')
            obs_filt.attrs['description'] = 'Averaged velocity over the period'
            obs_filt.attrs['units'] = 'm/y'

        # unify the observations to displacement
        # to provide displacement values during inversion
        if velo_or_disp == "velo":
            self.ds["vx"] = self.ds["vx"] * self.ds["temporal_baseline"] / unit
            self.ds["vy"] = self.ds["vy"] * self.ds["temporal_baseline"] / unit

        self.ds = self.ds.persist()  # crash memory without loading
        # persist() is particularly useful when using a distributed cluster because the data will be loaded into distributed memory across your machines and be much faster to use than reading repeatedly from disk.

        return obs_filt

    def align_cube(self, cube: "cube_data_class", unit: int = 365, reproj_vel: bool = True, reproj_coord: bool = True,
                   interp_method: str = 'nearest'):

        """
        Reproject cube to match the resolution, projection, and region of self.
        
        :param cube: Cube to align to self
        :param unit: Unit of the velocities (365 for m/y, 1 for m/d) (default is 365)
        :param reproj_vel: Whether the velocity have to be reprojected or not -> it will modify their value (default is True)
        :param reproj_coord: Whether the coordinates have to be interpolated or not (using interp_method) (default is True)
        :param interp_method: Interpolation method used to reproject cube (default is 'nearest')
        
        :return: Cube projected to self
        """

        if reproj_vel:  # if the velocity components have to be reprojected in the new projection system
            grid = np.meshgrid(cube.ds['x'], cube.ds['y'])
            temp = cube.temp_base_()
            endx = np.array([(np.ma.masked_invalid(cube.ds['vx'][z]) * temp[z] / unit) + grid[0] for z in
                             range(
                                 cube.nz)])  # localisation of the final coordinate of each pixel displaced by the corresponding velocity vector, in x
            endy = np.array(
                [(np.ma.masked_invalid(cube.ds['vy'][z]) * temp[z] / unit) + grid[1] for z in
                 range(
                     cube.nz)])  # localisation of the final coordinate of each pixel displaced by the corresponding velocity vector, in y

            # reprojection of the final coordinate of each pixel displaced by the corresponding velocity vector
            transformer = Transformer.from_crs(cube.ds.proj4, self.ds.proj4)
            t = np.array([transformer.transform(endx[z], endy[z]) for z in range(cube.nz)])
            del endx, endy

            # Computation of the difference between final and oringinal coordinates in the new system
            grid = transformer.transform(grid[0], grid[1])
            vx = np.array([(grid[0] - t[z, 0, :, :]) / temp[z] * unit for z in
                           range(cube.nz)])  # positive toward the West
            vy = np.array([(t[z, 1, :, :] - grid[1]) / temp[z] * unit for z in
                           range(cube.nz)])  # positive toward the North
            cube.ds['vx'] = xr.DataArray(vx.astype('float32'), dims=['mid_date', 'y', 'x'],
                                         coords={'mid_date': cube.ds.mid_date, 'y': cube.ds.y, 'x': cube.ds.x})
            cube.ds['vx'].encoding = {'vx': {'dtype': 'float32', 'scale_factor': 0.1, 'units': 'm/y'}}
            cube.ds['vy'] = xr.DataArray(vy.astype('float32'), dims=['mid_date', 'y', 'x'],
                                         coords={'mid_date': cube.ds.mid_date, 'y': cube.ds.y, 'x': cube.ds.x})
            cube.ds['vy'].encoding = {'vy': {'dtype': 'float32', 'scale_factor': 0.1, 'units': 'm/y'}}
            del vx, vy

            # if reproj_coord:
        #     # Convert the system of coordinate and ajust the spatial resolution of self to match the resolution, projection, and region of cube
        #     cube.ds = cube.ds.rio.write_crs(cube.ds.proj4)
        #     self.ds = self.ds.rio.write_crs(self.ds.proj4)
        #     if interp_method == 'nearest':
        #         cube.ds = self.ds.rio.reproject_match(cube.ds, resampling=rasterio.enums.Resampling.nearest)
        #     # Update of cube_data_classxr attributes
        #     self.ds = self.ds.assign_attrs({'proj4': cube.ds.proj4})
        #     # cube2.ds = cube2.ds.rio.write_crs(cube2.proj4, inplace=True)
        #     self.nx = self.ds.dims['x']
        #     self.ny = self.ds.dims['y']

        if reproj_coord:
            # Convert the system of coordinate and ajust the spatial resolution of the cube2 to match the resolution, projection, and region of self, using a bilinear interpolation
            cube.ds = cube.ds.rio.write_crs(cube.ds.proj4)
            self.ds = self.ds.rio.write_crs(self.ds.proj4)
            cube.ds = cube.ds.transpose('mid_date', 'y', 'x')
            if interp_method == 'nearest':
                cube.ds = cube.ds.rio.reproject_match(self.ds, resampling=rasterio.enums.Resampling.nearest)
            # Update of cube_data_classxr attributes
            cube.ds = cube.ds.assign_attrs({'proj4': self.ds.proj4})
            # cube2.ds = cube2.ds.rio.write_crs(cube2.proj4, inplace=True)
            cube.nx = cube.ds.dims['x']
            cube.ny = cube.ds.dims['y']
            cube.ds = cube.ds.assign_coords({"x": self.ds.x, "y": cube.ds.y})

        cube.ds = cube.ds.assign_attrs({'author': f'{cube.ds.author} aligned'})

        return cube

    def merge_cube(self, cube: "cube_data_class"):
        self.ds = xr.concat([self.ds, cube.ds], dim='mid_date')
        self.ds = self.ds.chunk(chunks={'mid_date': self.ds['mid_date'].size})
        self.nz = self.ds['mid_date'].size

    def average_cube(self):
        """

        :return: xr dataset, with vx_mean, the mean of vx and vy_mean the mean of vy
        """
        ds_mean = xr.Dataset({})
        coords = {'y': self.ds.y, 'x': self.ds.x}
        ds_mean['vx_mean'] = xr.DataArray(self.ds['vx'].mean(dim='mid_date'), dims=['y', 'x'], coords=coords)
        ds_mean['vy_mean'] = xr.DataArray(self.ds['vy'].mean(dim='mid_date'), dims=['y', 'x'], coords=coords)
        return ds_mean

    def compute_heatmap_moving(self, points_heatmap: pd.DataFrame, variable: str = 'vv', method_interp: str = 'linear',
                               verbose: bool = False, freq: str = 'MS', method: str = 'mean') -> pd.DataFrame:
        """
        Compute a heatmap of the average monthly velocity, average all the velocities which are overlapping a given month

        :param points_heatmap: Points where the heatmap is to be computed
        :param variable: What variable is to be computed ('vx', 'vy' or 'vv')
        :param method_interp: Interpolation method used to determine the value at a specified point from the discrete velocities datas
        :param freq: frequency used in the pandas.date_range function (default: 'MS' every first day of the month)
        :param method: 'mean' or 'median'
        :param verbose: Print information throughout the process (default is False)


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

        def data_temporalpoint(k: int, points_heatmap):
            """Get the data at a given spatial point contained in points_heatmap"""
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
    def ncvv(self):
        """Return the Normalized Coherence Vector Velocity """
        return np.array([np.sqrt(np.nansum((self.ds['vx'].isel(x=i, y=j) / np.sqrt(
            self.ds['vx'].isel(x=i, y=j) ** 2 + self.ds['vy'].isel(x=i, y=j) ** 2))) ** 2 + np.nansum((self.ds[
                                                                                                           'vy'].isel(
            x=i, y=j) / np.sqrt(self.ds['vx'].isel(x=i, y=j) ** 2 + self.ds['vy'].isel(x=i, y=j) ** 2))) ** 2) / self.nz
                         for i in
                         range(self.nx) for j in range(self.ny)]).reshape(self.nx, self.ny)

    # %% ======================================================================== #
    #                             WRITING RESULTS In A NETCDF                     #
    # =========================================================================%% #

    def write_result_ticoi(self, result: list, source: str, sensor: str, filename: str = 'Time_series',
                           savepath: str | None = None, result_quality: list | None = None,
                           verbose: bool = False) -> Union["cube_data_class", str]:
        """
        Write the result from TICOI, stored in result, in a xarray dataset matching the conventions CF-1.10
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.pdf
        units has been changed to unit, since it was producing an error while wirtting the netcdf file
        :param result: list of pd xarray, resulut from the TICOI method
        :param source: name of the source
        :param sensor: sensors which have been used
        :param filename: filename of file to saved
        :param result_quality: if not None, list of the criterium used to evaluate the quality of the results
        :param savepath: string, path where to save the file
        :param verbose: Print information throughout the process (default is False)

        :return: new cube where the results are saved
        """
        # TODO: need to check the order of dimension: do we need to transpose?
        non_null_results = [result[i * self.ny + j]['vx'].shape[0] for i in range(self.nx) for j in range(self.ny)
                            if
                            result[i * self.ny + j]['vx'].shape[
                                0] != 0]  # temporal size of the results which are not empty
        first_date_results = [result[i * self.ny + j]['First_date'].iloc[0] for i in range(self.nx) for j in
                              range(self.ny) if
                              result[i * self.ny + j]['vx'].shape[
                                  0] != 0]  # temporal size of the results which are not empty
        if len(non_null_results) == 0:
            print('There is no results to write and/or save')
            return 'There is no results to write and/or save'

        if np.min(non_null_results) == np.max(non_null_results) and all(
                element == first_date_results[0] for element in
                first_date_results):  # if the dates of the results are the same for every pixel
            non_null_el = next((element for element in result if element.shape[0] != 0),
                               None)  # First result array which is not empty, and have size corresponding to the time period common between every pixel
            del non_null_results, first_date_results
            print('Same time dimension for every pixels')
        else:
            print('Not the same time dimension for every pixels')
            raise ValueError('Not the same time dimension for every pixels')

        cubenew = cube_data_class()
        time_variable = non_null_el['First_date'] + (non_null_el['Second_date'] - non_null_el['First_date']) // 2
        cubenew.ds['date1'] = xr.DataArray(non_null_el['First_date'], dims='mid_date', coords={'mid_date': time_variable})
        cubenew.ds['date1'].attrs = {'standard_name': 'first_date', 'unit': 'days',
                                     'long_name': 'first date between which the velocity is estimated'}
        cubenew.ds['date2'] = xr.DataArray(non_null_el['Second_date'], dims='mid_date', coords={'mid_date': time_variable})
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
                    np.full(non_null_el.shape[0], np.nan)) for i in range(self.nx) for j in range(self.ny)])
            result_arr = result_arr.reshape((self.nx, self.ny, len(time_variable)))
            cubenew.ds[var] = xr.DataArray(result_arr, dims=['x', 'y', 'mid_date'],
                                           coords={'x': self.ds['x'], 'y': self.ds['y'], 'mid_date': time_variable})
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

        del non_null_el, long_name, result_arr
        cubenew.ds['x'] = self.ds['x']
        cubenew.ds['x'].attrs = {'standard_name': 'projection_x_coordinate', 'unit': 'm',
                                 'long_name': 'x coordinate of projection'}
        cubenew.ds['y'] = self.ds['y']
        cubenew.ds['y'].attrs = {'standard_name': 'projection_y_coordinate', 'unit': 'm',
                                 'long_name': 'y coordinate of projection'}
        cubenew.ds['mid_date'] = time_variable.to_numpy()
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

    def write_result_tico(self, result: list, source: str, sensor: str, filename: str = 'Time_series',
                          savepath: str | None = None, result_quality: list | None = None,
                          verbose: bool = False) -> Union["cube_data_class", str]:
        """
        Write the result from TICOI, stored in result, in a xarray dataset matching the conventions CF-1.10
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.pdf
        units has been changed to unit, since it was producing an error while wirtting the netcdf file
        :param result: list of pd xarray, resulut from the TICOI method
        :param source: name of the source
        :param sensor: sensors which have been used
        :param filename:  filename of file to saved
        :param savepath: path where to save the file
        :param result_quality: if not None, list of the criterium used to evaluate the quality of the results
        :param verbose: Print information throughout the process (default is False)
        :return: new cube where the results are saved
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
        df_list = [reconstruct_common_ref(df[1], result_quality) for df in result]

        # List of the reference date, i.e. the first date of the cumulative displacement time series
        result_arr = np.array(
            [df_list[i]['Ref_date'][0] for i in range(len(df_list))]).reshape((self.nx, self.ny))
        cubenew.ds['reference_date'] = xr.DataArray(result_arr, dims=['x', 'y'],
                                                    coords={'x': self.ds['x'], 'y': self.ds['y']})
        cubenew.ds['reference_date'].attrs = {'standard_name': 'reference_date', 'unit': 'days',
                                              'description': 'first date of the cumulative displacement time series'}

        # Retrieve the list a second date in the whole data cube
        second_date_list = list(set(list(itertools.chain.from_iterable([df['Second_date'].values for df in df_list]))))
        second_date_list.sort()

        # reindex each dataframe according to the list of second date, so that each dataframe have the same temporal size
        df_list2 = []
        for i, df in enumerate(df_list):
            df.index = df['Second_date']
            df_list2.append(df.reindex(second_date_list))
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
            result_arr = result_arr.reshape((self.nx, self.ny, len(second_date_list)))
            cubenew.ds[var] = xr.DataArray(result_arr, dims=['x', 'y', 'second_date'],
                                           coords={'x': self.ds['x'], 'y': self.ds['y'],
                                                   'second_date': second_date_list})
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
