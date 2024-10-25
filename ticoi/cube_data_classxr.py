"""
Class object to store and manipulate velocity observation data

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import contextlib
import itertools
import os
import time
import warnings
from datetime import date
from typing import List, Optional, Union

import dask
import dask.array as da
import geopandas
import numpy as np
import pandas as pd
import pyproj.datadir
import rasterio as rio
import rasterio.enums
import rasterio.warp
import richdem as rd
import xarray as xr
from dask.array.lib.stride_tricks import sliding_window_view
from dask.diagnostics import ProgressBar
import pyproj
print(pyproj.datadir.get_data_dir())
from pyproj import CRS, Proj, Transformer
from rasterio.features import rasterize

from ticoi.filtering_functions import (
    dask_filt_warpper,
    dask_smooth_wrapper,
    median_filter,
)
from ticoi.interpolation_functions import reconstruct_common_ref, smooth_results
from ticoi.inversion_functions import construction_dates_range_np
from ticoi.mjd2date import mjd2date

# %% ======================================================================== #
#                              CUBE DATA CLASS                                #
# =========================================================================%% #


class cube_data_class:
    def __init__(self, cube=None, ds=None, sar=False):

        """
        Initialisation of the main attributes, or copy cube's attributes and ds dataset if given.

        :param cube: [cube_data_class] --- Cube to copy
        :param ds: [xr dataset | None] --- New dataset. If None, copy cube's dataset
        """

        if not isinstance(cube, cube_data_class):
            self.filedir = ""
            self.filename = ""
            self.nx = 250
            self.ny = 250
            self.nz = 0
            self.author = ""
            self.source = ""
            self.ds = xr.Dataset({})
            self.resolution = 50
            self.is_TICO = False
            self.is_SAR = False
            if sar:
                self.is_SAR = True
                self.incidenceAngle = 40
                self.headingAngle = -20

        else:
            self.filedir = cube.filedir
            self.filename = cube.filename
            self.nx = cube.nx
            self.ny = cube.ny
            self.nz = cube.nz
            self.author = cube.author
            self.source = cube.source
            self.ds = cube.ds if ds is None else ds
            self.resolution = cube.resolution
            self.is_TICO = cube.is_TICO
            self.is_SAR = cube.is_SAR
            if self.is_SAR:
                self.incidenceAngle = cube.incidenceAngle
                self.headingAngle = cube.headingAngle


    def update_dimension(self, time_dim: str = "mid_date"):

        """
        Update the attributes corresponding to cube dimensions: nx, ny, and nz

        :param time_dim_name: [str] [default is 'mid_date'] --- Name of the z dimension within the original dataset self.ds
        """

        self.nx = self.ds["x"].sizes["x"]
        self.ny = self.ds["y"].sizes["y"]
        self.nz = self.ds[time_dim].sizes[time_dim]
        if len(self.ds["x"]) != 0 and len(self.ds["y"]) != 0:
            self.resolution = self.ds["x"].values[1] - self.ds["x"].values[0]
        else:
            raise ValueError("Your cube is empty, please check the subset or buffer coordinates you provided")

    def subset(self, proj: str, subset: list):

        """
        Crop the dataset according to 4 coordinates describing a rectangle.

        :param proj: [str] --- EPSG system of the coordinates given in subset
        :param subset: [list] --- A list of 4 float, these values are used to give a subset of the dataset : [xmin, xmax, ymax, ymin]
        """
        print(os.environ.get("PROJ_LIB"))
        if CRS(self.ds.proj4) != CRS(proj):
            transformer = Transformer.from_crs(
                CRS(proj), CRS(self.ds.proj4)
            )  # convert the coordinates from proj to self.ds.proj4
            lon1, lat1 = transformer.transform(subset[2], subset[1])
            lon2, lat2 = transformer.transform(subset[3], subset[1])
            lon3, lat3 = transformer.transform(subset[2], subset[1])
            lon4, lat4 = transformer.transform(subset[3], subset[0])
            self.ds = self.ds.sel(
                x=slice(np.min([lon1, lon2, lon3, lon4]), np.max([lon1, lon2, lon3, lon4])),
                y=slice(np.max([lat1, lat2, lat3, lat4]), np.min([lat1, lat2, lat3, lat4])),
            )
            del lon1, lon2, lon3, lon4, lat1, lat2, lat3, lat4
        else:
            self.ds = self.ds.sel(
                x=slice(np.min([subset[0], subset[1]]), np.max([subset[0], subset[1]])),
                y=slice(np.max([subset[2], subset[3]]), np.min([subset[2], subset[3]])),
            )

        if len(self.ds["x"].values) == 0 and len(self.ds["y"].values) == 0:
            print(f"[Data loading] The given subset is not part of cube {self.filename}")

    def buffer(self, proj: str, buffer: list):

        """
        Crop the dataset around a given pixel, the amount of surroundings pixels kept is given by the buffer.

        :param proj: [str] --- EPSG system of the coordinates given in subset
        :param buffer:  [list] --- A list of 3 float, the first two are the longitude and the latitude of the central point, the last is the buffer size
        """

        if CRS(self.ds.proj4) != CRS(proj):  # Convert the coordinates from proj to self.ds.proj4
            transformer = Transformer.from_crs(CRS(proj), CRS(self.ds.proj4))
            i1, j1 = transformer.transform(buffer[1] + buffer[2], buffer[0] - buffer[2])
            i2, j2 = transformer.transform(buffer[1] - buffer[2], buffer[0] + buffer[2])
            i3, j3 = transformer.transform(buffer[1] + buffer[2], buffer[0] + buffer[2])
            i4, j4 = transformer.transform(buffer[1] - buffer[2], buffer[0] - buffer[2])
            self.ds = self.ds.sel(
                x=slice(np.min([i1, i2, i3, i4]), np.max([i1, i2, i3, i4])),
                y=slice(np.max([j1, j2, j3, j4]), np.min([j1, j2, j3, j4])),
            )
            del i3, i4, j3, j4
        else:
            i1, j1 = buffer[0] - buffer[2], buffer[1] + buffer[2]
            i2, j2 = buffer[0] + buffer[2], buffer[1] - buffer[2]
            self.ds = self.ds.sel(
                x=slice(np.min([i1, i2]), np.max([i1, i2])), y=slice(np.max([j1, j2]), np.min([j1, j2]))
            )
            del i1, i2, j1, j2, buffer

        if len(self.ds["x"].values) == 0 and len(self.ds["y"].values) == 0:
            print(f"[Data loading] The given pixel and its surrounding buffer are not part of cube {self.filename}")

    def determine_optimal_chunk_size(
        self,
        variable_name: str = "vx",
        x_dim: str = "x",
        y_dim: str = "y",
        time_dim: str = "mid_date",
        verbose: bool = False,
    ) -> (int, int, int):  # type: ignore

        """
        A function to determine the optimal chunk size for a given time series array based on its size.

        :param variable_name: [str] [default is 'vx'] --- Name of the variable containing the time series array
        :param x_dim: [str] [default is 'x'] --- Name of the x dimension in the array
        :param y_dim: [str] [default is 'y'] --- Name of the y dimension in the array
        :param time_dim_name: [str] [default is 'mid_date'] --- Name of the z dimension within the original dataset self.ds
        :param verbose: [bool] [default is False] --- Boolean flag to control verbosity of output

        :return tc: [int] --- Chunk size along the time dimension
        :return yc: [int] --- Chunk size along the y dimension
        :return xc: [int] --- Chunk size along the x dimension
        """

        if verbose:
            print("[Data loading] Dask chunk size:")

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

        time_axis = self.ds[variable_name].dims.index(time_dim)
        x_axis = self.ds[variable_name].dims.index(x_dim)
        y_axis = self.ds[variable_name].dims.index(y_dim)
        axis_sizes = {i: -1 if i == time_axis else "auto" for i in range(3)}
        arr = self.ds[variable_name].data.rechunk(axis_sizes, block_size_limit=chunk_size_limit, balance=True)
        tc, yc, xc = arr.chunks[time_axis][0], arr.chunks[y_axis][0], arr.chunks[x_axis][0]
        chunksize = self.ds[variable_name][:tc, :yc, :xc].nbytes / 1e6
        if verbose:
            print("[Data loading] Chunk shape:", "(" + ",".join([str(x) for x in [tc, yc, xc]]) + ")")
            print(
                "[Data loading] Chunk size:",
                self.ds[variable_name][:tc, :yc, :xc].nbytes,
                "(" + str(round(chunksize, 1)) + "MB)",
            )
        return tc, yc, xc

    # %% ==================================================================== #
    #                         CUBE LOADING METHODS                            #
    # =====================================================================%% #

    def load_itslive(
        self,
        filepath: str,
        conf: bool = False,
        subset: list | None = None,
        buffer: list | None = None,
        pick_date: list | None = None,
        pick_sensor: list | None = None,
        pick_temp_bas: list | None = None,
        proj: str = "EPSG:4326",
        verbose: bool = False,
    ):

        """
        Load a cube dataset written by ITS_LIVE.

        :param filepath: [str] --- Filepath of the dataset
        :param conf: [bool] [default is False] --- If True convert the error in confidence between 0 and 1
        :param subset: [list | None] [default is None] --- A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax]
        :param buffer: [list | None] [default is None] --- A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size
        :param pick_date: [list | None] [default is None] --- A list of 2 string yyyy-mm-dd, pick the data between these two date
        :param pick_sensor: [list | None] [default is None] --- A list of strings, pick only the corresponding sensors
        :param pick_temp_bas: [list | None] [default is None] --- A list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param proj: [str] [default is 'EPSG:4326'] --- Projection of the buffer or subset which is given
        :param verbose: [bool] [default is False] --- Print information throughout the process
        """

        if verbose:
            print(f"[Data loading] Path to cube file : {filepath}")

        self.filedir = os.path.dirname(filepath)  # Path were is stored the netcdf file
        self.filename = os.path.basename(filepath)  # Name of the netcdf file
        self.ds = self.ds.assign_attrs({"proj4": self.ds["mapping"].proj4text})
        self.author = self.ds.author.split(", a NASA")[0]
        self.source = self.ds.url

        if subset is not None:  # Crop according to 4 coordinates
            self.subset(proj, subset)
        elif buffer is not None:  # Crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)
        if pick_date is not None:
            self.ds = self.ds.where(
                (
                    (self.ds["acquisition_date_img1"] >= np.datetime64(pick_date[0]))
                    & (self.ds["acquisition_date_img2"] <= np.datetime64(pick_date[1]))
                ).compute(),
                drop=True,
            )

        self.update_dimension()  # Update self.nx,self.ny,self.nz

        if conf:
            minconfx = np.nanmin(self.ds["vx_error"].values[:])
            maxconfx = np.nanmax(self.ds["vx_error"].values[:])
            minconfy = np.nanmin(self.ds["vy_error"].values[:])
            maxconfy = np.nanmax(self.ds["vy_error"].values[:])

        date1 = np.array([np.datetime64(date_str, "D") for date_str in self.ds["acquisition_date_img1"].values])
        date2 = np.array([np.datetime64(date_str, "D") for date_str in self.ds["acquisition_date_img2"].values])

        # np.char.strip is used to remove the null character ('�') from each element and np.core.defchararray.add to
        # concatenate array of different types
        sensor = np.core.defchararray.add(
            np.char.strip(self.ds["mission_img1"].values.astype(str), "�"),
            np.char.strip(self.ds["satellite_img1"].values.astype(str), "�"),
        ).astype("U10")
        sensor[sensor == "L7"] = "Landsat-7"
        sensor[sensor == "L8"] = "Landsat-8"
        sensor[sensor == "L9"] = "Landsat-9"
        sensor[np.isin(sensor, ["S1A", "S1B"])] = "Sentinel-1"
        sensor[np.isin(sensor, ["S2A", "S2B"])] = "Sentinel-2"

        if conf:  # Normalize the error between 0 and 1, and convert error in confidence
            errorx = 1 - (self.ds["vx_error"].values - minconfx) / (maxconfx - minconfx)
            errory = 1 - (self.ds["vy_error"].values - minconfy) / (maxconfy - minconfy)
        else:
            errorx = self.ds["vx_error"].values
            errory = self.ds["vy_error"].values

        # Drop variables not in the specified list
        variables_to_keep = ["vx", "vy", "mid_date", "x", "y"]
        self.ds = self.ds.drop_vars([var for var in self.ds.variables if var not in variables_to_keep])
        # Drop attributes not in the specified list
        attributes_to_keep = ["date_created", "mapping", "author", "proj4"]
        self.ds.attrs = {attr: self.ds.attrs[attr] for attr in attributes_to_keep if attr in self.ds.attrs}

        # self.ds = self.ds.unify_chunks()  # to avoid error ValueError: Object has inconsistent chunks along
        # dimension mid_date. This can be fixed by calling unify_chunks(). Create new variable and chunk them
        self.ds["sensor"] = xr.DataArray(sensor, dims="mid_date").chunk({"mid_date": self.ds.chunks["mid_date"]})
        self.ds = self.ds.unify_chunks()
        self.ds["date1"] = xr.DataArray(date1, dims="mid_date").chunk({"mid_date": self.ds.chunks["mid_date"]})
        self.ds = self.ds.unify_chunks()
        self.ds["date2"] = xr.DataArray(date2, dims="mid_date").chunk({"mid_date": self.ds.chunks["mid_date"]})
        self.ds = self.ds.unify_chunks()
        self.ds["source"] = xr.DataArray(["ITS_LIVE"] * self.nz, dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds = self.ds.unify_chunks()
        self.ds["errorx"] = xr.DataArray(errorx, dims=["mid_date"], coords={"mid_date": self.ds.mid_date}).chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds = self.ds.unify_chunks()
        self.ds["errory"] = xr.DataArray(errory, dims=["mid_date"], coords={"mid_date": self.ds.mid_date}).chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )

        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds["sensor"].isin(pick_sensor))
        if pick_temp_bas is not None:
            temp = (self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")
            self.ds = self.ds.where(((pick_temp_bas[0] < temp) & (temp < pick_temp_bas[1])).compute(), drop=True)
            del temp
        self.ds = self.ds.unify_chunks()

    def load_millan(
        self,
        filepath: str,
        conf: bool = False,
        subset: list | None = None,
        buffer: list | None = None,
        pick_date: list | None = None,
        pick_sensor: list | None = None,
        pick_temp_bas: list | None = None,
        proj: str = "EPSG:4326",
        verbose: bool = False,
    ):

        """
        Load a cube dataset written by R. Millan et al.

        :param filepath: [str] --- Filepath of the dataset
        :param conf: [bool] [default is False] --- If True convert the error in confidence between 0 and 1
        :param subset: [list | None] [default is None] --- A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax]
        :param buffer: [list | None] [default is None] --- A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size
        :param pick_date: [list | None] [default is None] --- A list of 2 string yyyy-mm-dd, pick the data between these two date
        :param pick_sensor: [list | None] [default is None] --- A list of strings, pick only the corresponding sensors
        :param pick_temp_bas: [list | None] [default is None] --- A list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param proj: [str] [default is 'EPSG:4326'] --- Projection of the buffer or subset which is given
        :param verbose: [bool] [default is False] --- Print information throughout the process
        """

        if verbose:
            print(f"[Data loading] Path to cube file : {filepath}")

        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        self.author = "IGE"  # name of the author
        self.source = self.ds.source
        del filepath

        # self.split_cube(n_split=2, dim=['x', 'y'], savepath=f"{self.filedir}/{self.filename[:-3]}_")

        if subset is not None:  # Crop according to 4 coordinates
            self.subset(proj, subset)
        elif buffer is not None:  # Crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        # Uniformization of the name and format of the time coordinate
        self.ds = self.ds.rename({"z": "mid_date"})

        date1 = [mjd2date(date_str) for date_str in self.ds["date1"].values]  # conversion in date
        date2 = [mjd2date(date_str) for date_str in self.ds["date2"].values]
        self.ds = self.ds.unify_chunks()
        self.ds["date1"] = xr.DataArray(np.array(date1).astype("datetime64[ns]"), dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds = self.ds.unify_chunks()
        self.ds["date2"] = xr.DataArray(np.array(date2).astype("datetime64[ns]"), dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds = self.ds.unify_chunks()
        del date1, date2

        # Temporal subset between two dates
        if pick_date is not None:
            self.ds = self.ds.where(
                (
                    (self.ds["date1"] >= np.datetime64(pick_date[0]))
                    & (self.ds["date2"] <= np.datetime64(pick_date[1]))
                ).compute(),
                drop=True,
            )
        del pick_date

        self.ds = self.ds.assign_coords(
            mid_date=np.array(self.ds["date1"] + (self.ds["date2"] - self.ds["date1"]) // 2)
        )
        self.update_dimension()

        if conf and "confx" not in self.ds.data_vars:  # convert the errors into confidence indicators between 0 and 1
            minconfx = np.nanmin(self.ds["error_vx"].values[:])
            maxconfx = np.nanmax(self.ds["error_vx"].values[:])
            minconfy = np.nanmin(self.ds["error_vy"].values[:])
            maxconfy = np.nanmax(self.ds["error_vy"].values[:])
            errorx = 1 - (self.ds["error_vx"].values - minconfx) / (maxconfx - minconfx)
            errory = 1 - (self.ds["error_vy"].values - minconfy) / (maxconfy - minconfy)
        else:
            errorx = self.ds["error_vx"].values[:]
            errory = self.ds["error_vy"].values[:]

        # Homogenize sensors names
        sensor = np.char.strip(
            self.ds["sensor"].values.astype(str), "�"
        )  # np.char.strip is used to remove the null character ('�') from each element
        sensor[np.isin(sensor, ["S1"])] = "Sentinel-1"
        sensor[np.isin(sensor, ["S2"])] = "Sentinel-2"
        sensor[np.isin(sensor, ["landsat-8", "L8", "L8. "])] = "Landsat-8"

        # Drop variables not in the specified list
        self.ds = self.ds.drop_vars(
            [var for var in self.ds.variables if var not in ["vx", "vy", "mid_date", "x", "y", "date1", "date2"]]
        )
        self.ds = self.ds.transpose("mid_date", "y", "x")

        # Store the variable in xarray dataset
        self.ds["sensor"] = xr.DataArray(sensor, dims="mid_date").chunk({"mid_date": self.ds.chunks["mid_date"]})
        del sensor
        self.ds = self.ds.unify_chunks()
        self.ds["source"] = xr.DataArray(["IGE"] * self.nz, dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds = self.ds.unify_chunks()
        self.ds["errorx"] = xr.DataArray(
            np.tile(errorx[:, np.newaxis, np.newaxis], (1, self.ny, self.nx)),
            dims=["mid_date", "y", "x"],
            coords={"mid_date": self.ds.mid_date, "y": self.ds.y, "x": self.ds.x},
        ).chunk(chunks=self.ds.chunks)
        self.ds = self.ds.unify_chunks()
        self.ds["errory"] = xr.DataArray(
            np.tile(errory[:, np.newaxis, np.newaxis], (1, self.ny, self.nx)),
            dims=["mid_date", "y", "x"],
            coords={"mid_date": self.ds.mid_date, "y": self.ds.y, "x": self.ds.x},
        ).chunk(chunks=self.ds.chunks)
        del errorx, errory

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds["sensor"].isin(pick_sensor))
        if pick_temp_bas is not None:
            self.ds = self.ds.sel(
                mid_date=(pick_temp_bas[0] < ((self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")))
                & (((self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")) < pick_temp_bas[1])
            )
        self.ds = self.ds.unify_chunks()

    def load_ducasse(
        self,
        filepath: str,
        conf: bool = False,
        subset: list | None = None,
        buffer: list | None = None,
        pick_date: list | None = None,
        pick_sensor: list | None = None,
        pick_temp_bas: list | None = None,
        proj: str = "EPSG:4326",
        verbose: bool = False,
    ):

        """
        Load a cube dataset written by E. Ducasse et al. (Pleiades data)

        :param filepath: [str] --- Filepath of the dataset
        :param conf: [bool] [default is False] --- If True convert the error in confidence between 0 and 1
        :param subset: [list | None] [default is None] --- A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax]
        :param buffer: [list | None] [default is None] --- A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size
        :param pick_date: [list | None] [default is None] --- A list of 2 string yyyy-mm-dd, pick the data between these two date
        :param pick_sensor: [list | None] [default is None] --- A list of strings, pick only the corresponding sensors
        :param pick_temp_bas: [list | None] [default is None] --- A list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param proj: [str] [default is 'EPSG:4326'] --- Projection of the buffer or subset which is given
        :param verbose: [bool] [default is False] --- Print information throughout the process
        """

        if verbose:
            print(f"[Data loading] Path to cube file : {filepath}")

        self.ds = self.ds.chunk({"x": 125, "y": 125, "time": 2000})  # set chunk
        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        self.author = "IGE"  # name of the author
        del filepath

        # Spatial subset
        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)
        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        # Uniformization of the name and format of the time coordinate
        self.ds = self.ds.rename({"time": "mid_date"})

        date1 = [date_str.split(" ")[0] for date_str in self.ds["mid_date"].values]
        date2 = [date_str.split(" ")[1] for date_str in self.ds["mid_date"].values]
        self.ds["date1"] = xr.DataArray(np.array(date1).astype("datetime64[ns]"), dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds["date2"] = xr.DataArray(np.array(date2).astype("datetime64[ns]"), dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        del date1, date2

        # Temporal subset between two dates
        if pick_date is not None:
            self.ds = self.ds.where(
                (
                    (self.ds["date1"] >= np.datetime64(pick_date[0]))
                    & (self.ds["date2"] <= np.datetime64(pick_date[1]))
                ).compute(),
                drop=True,
            )
        del pick_date

        self.ds = self.ds.assign_coords(
            mid_date=np.array(self.ds["date1"] + (self.ds["date2"] - self.ds["date1"]) // 2)
        )
        self.update_dimension()  # update self.nx, self.ny and self.nz

        # Drop variables not in the specified list
        variables_to_keep = ["vx", "vy", "mid_date", "x", "y", "date1", "date2"]
        self.ds = self.ds.drop_vars([var for var in self.ds.variables if var not in variables_to_keep])
        self.ds = self.ds.transpose("mid_date", "y", "x")

        # Store the variable in xarray dataset
        self.ds["sensor"] = xr.DataArray(["Pleiades"] * self.nz, dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds["source"] = xr.DataArray(["IGE"] * self.nz, dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds["vy"] = -self.ds["vy"]

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds["sensor"].isin(pick_sensor))
        if pick_temp_bas is not None:
            self.ds = self.ds.sel(
                mid_date=(pick_temp_bas[0] < ((self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")))
                & (((self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")) < pick_temp_bas[1])
            )

        # Set errors equal to one (no information on the error here)
        self.ds["errorx"] = xr.DataArray(np.ones(self.ds["mid_date"].size), dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )
        self.ds["errory"] = xr.DataArray(np.ones(self.ds["mid_date"].size), dims="mid_date").chunk(
            {"mid_date": self.ds.chunks["mid_date"]}
        )

    def load_charrier(
        self,
        filepath: str,
        conf: bool = False,
        subset: list | None = None,
        buffer: list | None = None,
        pick_date: list | None = None,
        pick_sensor: list | None = None,
        pick_temp_bas: list | None = None,
        proj: str = "EPSG:4326",
        verbose: bool = False,
    ):

        """
        Load a cube dataset written by L.Charrier et al.

        :param filepath: [str] --- Filepath of the dataset
        :param conf: [bool] [default is False] --- If True convert the error in confidence between 0 and 1
        :param subset: [list | None] [default is None] --- A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax]
        :param buffer: [list | None] [default is None] --- A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size
        :param pick_date: [list | None] [default is None] --- A list of 2 string yyyy-mm-dd, pick the data between these two date
        :param pick_sensor: [list | None] [default is None] --- A list of strings, pick only the corresponding sensors
        :param pick_temp_bas: [list | None] [default is None] --- A list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param proj: [str] [default is 'EPSG:4326'] --- Projection of the buffer or subset which is given
        :param verbose: [bool] [default is False] --- Print information throughout the process
        """

        if verbose:
            print(f'[Data loading] Path to cube file {"(TICO cube)" if self.is_TICO else ""} : {filepath}')

        # information about the cube
        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # Name of the netcdf file
        if self.ds.author == "J. Mouginot, R.Millan, A.Derkacheva_aligned":
            self.author = "IGE"  # Name of the author
        else:
            self.author = self.ds.author
        self.source = self.ds.source
        del filepath

        # Select specific data within the cube
        if subset is not None:  # Crop according to 4 coordinates
            self.subset(proj, subset)
        elif buffer is not None:  # Crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        time_dim = "mid_date" if not self.is_TICO else "second_date"  # 'date2' if we load TICO data
        self.update_dimension(time_dim)

        # Temporal subset between two dates
        if pick_date is not None:
            if not self.is_TICO:
                self.ds = self.ds.where(
                    (
                        (self.ds["date1"] >= np.datetime64(pick_date[0]))
                        & (self.ds["date2"] <= np.datetime64(pick_date[1]))
                    ).compute(),
                    drop=True,
                )
            else:
                self.ds = self.ds.where(
                    (
                        (self.ds["second_date"] >= np.datetime64(pick_date[0]))
                        & (self.ds["second_date"] <= np.datetime64(pick_date[1]))
                    ).compute(),
                    drop=True,
                )
        del pick_date

        self.update_dimension(time_dim)

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            if not self.is_TICO:
                self.ds = self.ds.sel(mid_date=self.ds["sensor"].isin(pick_sensor))
            else:
                self.ds = self.ds.sel(second_date=self.ds["sensor"].isin(pick_sensor))

        # Following properties are not available for TICO cubes
        if not self.is_TICO:
            # Pick specific temporal baselines
            if pick_temp_bas is not None:
                self.ds = self.ds.sel(
                    mid_date=(pick_temp_bas[0] < ((self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")))
                    & (((self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")) < pick_temp_bas[1])
                )

            # Convert the errors into confidence indicators between 0 and 1
            if conf and "confx" not in self.ds.data_vars:
                minconfx = np.nanmin(self.ds["errorx"].values[:])
                maxconfx = np.nanmax(self.ds["errorx"].values[:])
                minconfy = np.nanmin(self.ds["errory"].values[:])
                maxconfy = np.nanmax(self.ds["errory"].values[:])
                errorx = 1 - (self.ds["errorx"].values - minconfx) / (maxconfx - minconfx)
                errory = 1 - (self.ds["errory"].values - minconfy) / (maxconfy - minconfy)
                self.ds["errorx"] = xr.DataArray(
                    errorx,
                    dims=["mid_date", "y", "x"],
                    coords={"mid_date": self.ds.mid_date, "y": self.ds.y, "x": self.ds.x},
                ).chunk(chunks=self.ds.chunks)
                self.ds["errory"] = xr.DataArray(
                    errory,
                    dims=["mid_date", "y", "x"],
                    coords={"mid_date": self.ds.mid_date, "y": self.ds.y, "x": self.ds.x},
                ).chunk(chunks=self.ds.chunks)

            # For cube written with write_result_TICOI
            if "source" not in self.ds.variables:
                self.ds["source"] = xr.DataArray([self.ds.author] * self.nz, dims="mid_date").chunk(
                    {"mid_date": self.ds.chunks["mid_date"]}
                )
            if "sensor" not in self.ds.variables:
                self.ds["sensor"] = xr.DataArray([self.ds.sensor] * self.nz, dims="mid_date").chunk(
                    {"mid_date": self.ds.chunks["mid_date"]}
                )

    def load(
        self,
        filepath: list | str,
        chunks: dict | str | int = {},
        conf: bool = False,
        subset: str | None = None,
        buffer: str | None = None,
        pick_date: str | None = None,
        pick_sensor: str | None = None,
        pick_temp_bas: str | None = None,
        proj: str = "EPSG:4326",
        mask: str | xr.DataArray = None,
        reproj_coord: bool = False,
        reproj_vel: bool = False,
        sar_info: str | None = None,
        verbose: bool = False,
    ):

        """
        Load a cube dataset from a file in format netcdf (.nc) or zarr. The data are directly stored within the present object.

        :param filepath: [list | str] --- Filepath of the dataset, if list of filepaths, load all the cubes and merge them
        :param chunks: [dict] --- Dictionary with the size of chunks for each dimension, if chunks=-1 loads the dataset with dask using a single chunk for all arrays.
                                  chunks={} loads the dataset with dask using engine preferred chunks if exposed by the backend, otherwise with a single chunk for all arrays,
                                  chunks='auto' will use dask auto chunking taking into account the engine preferred chunks.
        :param conf: [bool] [default is False] --- If True convert the error in confidence between 0 and 1
        :param subset: [list | None] [default is None] --- A list of 4 float, these values are used to give a subset of the dataset in the form [xmin, xmax, ymin, ymax]
        :param buffer: [list | None] [default is None] --- A list of 3 float, the first two are the longitude and the latitude of the central point, the last one is the buffer size
        :param pick_date: [list | None] [default is None] --- A list of 2 string yyyy-mm-dd, pick the data between these two date
        :param pick_sensor: [list | None] [default is None] --- A list of strings, pick only the corresponding sensors
        :param pick_temp_bas: [list | None] [default is None] --- A list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param proj: [str] [default is 'EPSG:4326'] --- Projection of the buffer or subset which is given
        :param mask: [str | xr dataarray | None] [default is None] --- Mask some of the data of the cube, either a dataarray with 0 and 1, or a path to a dataarray or an .shp file
        :param reproj_coord: [bool] [default is False] --- If True reproject the second cube of the list filepath to the grid coordinates of the first cube
        :param reproj_vel: [bool] [default is False] --- If True reproject the velocity components, to match the coordinate system of the first cube

        :param verbose: [bool] [default is False] --- Print information throughout the process
        """
        self.__init__()

        time_dim_name = {
            "ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)": "mid_date",
            "J. Mouginot, R.Millan, A.Derkacheva": "z",
            "J. Mouginot, R.Millan, A.Derkacheva_aligned": "mid_date",
            "L. Charrier, L. Guo": "mid_date",
            "L. Charrier": "mid_date",
            "E. Ducasse": "time",
            "S. Leinss, L. Charrier": "mid_date",
            "IGE": "mid_date",
        }

        assert (
            type(filepath) == list or type(filepath) == str
        ), f"The filepath must be a string (path to the cube file) or a list of strings, not {type(filepath)}."

        if type(filepath) == list:  # Merge several cubes
            self.load(
                filepath[0],
                chunks=chunks,
                conf=conf,
                subset=subset,
                buffer=buffer,
                pick_date=pick_date,
                pick_sensor=pick_sensor,
                pick_temp_bas=pick_temp_bas,
                proj=proj,
                mask=mask,
                verbose=verbose,
            )

            for n in range(1, len(filepath)):
                cube2 = cube_data_class()
                # res = self.ds['x'].values[1] - self.ds['x'].values[0] # Resolution of the main data
                sub = [
                    self.ds["x"].min().values,
                    self.ds["x"].max().values,
                    self.ds["y"].min().values,
                    self.ds["y"].max().values,
                ]
                cube2.load(
                    filepath[n],
                    chunks=chunks,
                    conf=conf,
                    subset=sub,
                    pick_date=pick_date,
                    pick_sensor=pick_sensor,
                    pick_temp_bas=pick_temp_bas,
                    proj=self.ds.proj4,
                    mask=mask,
                    verbose=verbose,
                )
                # Align the new cube to the main one (interpolate the coordinate and/or reproject it)
                if reproj_vel or reproj_coord:
                    cube2 = self.align_cube(
                        cube2, reproj_vel=reproj_vel, reproj_coord=reproj_coord, interp_method="nearest"
                    )
                self.merge_cube(cube2)  # Merge the new cube to the main one
            del cube2
            if chunks == {}:  # Rechunk with optimal chunk size
                var_name = "vx" if not self.is_TICO else "dx"
                time_dim = time_dim_name[self.ds.author] if not self.is_TICO else "second_date"
                tc, yc, xc = self.determine_optimal_chunk_size(
                    variable_name=var_name, x_dim="x", y_dim="y", time_dim=time_dim, verbose=True
                )
                self.ds = self.ds.chunk({time_dim: tc, "x": xc, "y": yc})

        else:  # Load one cube
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):  # To avoid creating the large chunks
                if filepath.split(".")[-1] == "nc":
                    try:
                        self.ds = xr.open_dataset(filepath, engine="netcdf4", chunks=chunks)
                    except NotImplementedError:  # Can not use auto rechunking with object dtype. We are unable to estimate the size in bytes of object data
                        chunks = {}
                        self.ds = xr.open_dataset(filepath, engine="netcdf4", chunks=chunks)  # Set no chunks

                    if "Author" in self.ds.attrs:  # Uniformization of the attribute Author to author
                        self.ds.attrs["author"] = self.ds.attrs.pop("Author")

                    self.is_TICO = False if time_dim_name[self.ds.author] in self.ds.dims else True
                    time_dim = time_dim_name[self.ds.author] if not self.is_TICO else "second_date"
                    var_name = "vx" if not self.is_TICO else "dx"

                    if chunks == {}:  # Rechunk with optimal chunk size
                        tc, yc, xc = self.determine_optimal_chunk_size(
                            variable_name=var_name, x_dim="x", y_dim="y", time_dim=time_dim, verbose=True
                        )
                        self.ds = self.ds.chunk({time_dim: tc, "x": xc, "y": yc})

                elif filepath.split(".")[-1] == "zarr":
                    if chunks == {}:
                        chunks = "auto"  # Change the default value to auto
                    self.ds = xr.open_dataset(
                        filepath, decode_timedelta=False, engine="zarr", consolidated=True, chunks=chunks
                    )
                    self.is_TICO = False
                    time_dim = "mid_date"
                    var_name = "vx"

                if verbose:
                    print("[Data loading] File open")

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

                time_dim = "mid_date" if not self.is_TICO else "second_date"
                # Rechunk again if the size of the cube is changed:
                if any(x is not None for x in [pick_date, subset, buffer, pick_sensor, pick_temp_bas]):
                    tc, yc, xc = self.determine_optimal_chunk_size(
                        variable_name=var_name, x_dim="x", y_dim="y", time_dim=time_dim, verbose=True
                    )
                    self.ds = self.ds.chunk({time_dim: tc, "x": xc, "y": yc})

                # Reorder the coordinates to keep the consistency
                self.ds = self.ds.copy().sortby(time_dim).transpose("x", "y", time_dim)
                self.standardize_cube_for_processing(time_dim)

                if mask is not None:
                    self.mask_cube(mask)

                if self.is_SAR:
                    if sar_info is None:
                        raise ValueError("The sar_info file must be provided for SAR data.")
                    else:
                        import json
                        if sar_info.split(".")[-1] != "json":
                            raise ValueError("The sar_info file must be a json file.")
                        with open(sar_info) as f:
                            sar_info = json.load(f)
                            self.incidenceAngle = sar_info["incidenceAngle"]
                            self.headingAngle = sar_info["headingAngle"]
                # if self.ds['mid_date'].dtype == ('<M8[ns]'): #if the dates are given in ns, convert them to days
                #     self.ds['mid_date'] = self.ds['date2'].astype('datetime64[D]')
                #     self.ds['date1'] = self.ds['date1'].astype('datetime64[D]')
                #     self.ds['date2'] = self.ds['date2'].astype('datetime64[D]')

                if verbose:
                    print(f"[Data loading] Author : {self.ds.author}")

    def standardize_cube_for_processing(self, time_dim="mid_date"):

        """
        Prepare the xarray dataset for the processing: transpose the dimension, add a variable temporal_baseline, errors if they do not exist

        :param time_dim_name: [str] [default is 'mid_date'] --- Name of the z dimension within the original dataset self.ds
        """

        self.ds = self.ds.unify_chunks()
        if self.ds.chunksizes[time_dim] != (self.nz,):
            self.ds = self.ds.chunk({time_dim: self.nz})

        if not self.is_TICO:
            # Create a variable for temporal_baseline
            self.ds["temporal_baseline"] = xr.DataArray(
                (self.ds["date2"] - self.ds["date1"]).dt.days.values, dims="mid_date"
            )

            # Add errors if not already there
            if "errorx" not in self.ds.variables:
                self.ds["errorx"] = ("mid_date", np.ones(len(self.ds["mid_date"])))
                self.ds["errory"] = ("mid_date", np.ones(len(self.ds["mid_date"])))

    # %% ==================================================================== #
    #                                 ACCESSORS                               #
    # =====================================================================%% #

    def sensor_(self) -> list:

        """
        Accessor to the sensors whoch captured the data.

        :return: [list] --- List of sensor
        """

        return self.ds["sensor"].values.tolist()

    def source_(self) -> list:

        """
        Accessor to the source of the data.

        :return: [list] --- List of source
        """

        return self.ds["source"].values.tolist()

    def temp_base_(self, return_list: bool = True, format_date: str = "float") -> list | np.ndarray:

        """
        Get the temporal baseline of the dataset.

        :param return_list: [bool] [default is True] --- If True return of a list of date, else return a np array
        :param format_date: [str] [default is 'float'] --- 'float' or 'D' format of the date as output

        :return: [list | np array] --- List of the temporal baselines
        """

        if format_date == "D":
            temp = self.ds["date2"] - self.ds["date1"]
        elif format_date == "float":
            # temp = (self.ds['date2'].values-self.ds['date1'].values).astype('timedelta64[D]'))/ np.timedelta64(1, 'D')
            temp = (self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")
        else:
            raise NameError("Please enter format as float or D")
        if return_list:
            return temp.values.tolist()
        else:
            return temp.values

    def date1_(self):

        """
        Accessor to the first dates of acquisition.

        :return: [np array] --- np array of date1
        """

        return np.asarray(self.ds["date1"]).astype("datetime64[D]")

    def date2_(self):

        """
        Accessor to the second dates of acquisition.

        :return: [np array] --- np array of date2
        """

        return np.asarray(self.ds["date2"]).astype("datetime64[D]")

    def datec_(self):

        """
        Accessor to the central dates of the data.

        :return: [np array] --- np array of central date
        """

        return (self.date1_() + self.temp_base_(return_list=False, format_date="D") // 2).astype("datetime64[D]")

    def vv_(self):

        """
        Accessor to the magnitude of the velocities.

        :return: [np array] --- np array of velocity magnitude
        """

        return np.sqrt(self.ds["vx"] ** 2 + self.ds["vy"] ** 2)

    # %% ==================================================================== #
    #                         PIXEL LOADING METHODS                           #
    # =====================================================================%% #

    def convert_coordinates(self, i: int | float, j: int | float, proj: str, verbose: bool = False) -> (float, float):  # type: ignore

        """
        Convert the coordinate (i, j) which are in projection proj, to projection of the cube dataset.

        :params i, j: [int | float] --- Coordinates to be converted
        :param proj: [str] --- Projection of (i, j) coordinates
        :param verbose: [bool] [default is False] --- If True, print some text

        :return i, j: [int | float] --- Converted (i, j)
        """

        # Convert coordinates if needed
        if proj == "EPSG:4326":
            myproj = Proj(self.ds.proj4)
            i, j = myproj(i, j)
            if verbose:
                print(f"[Data loading] Converted to projection {self.ds.proj4}: {i, j}")
        else:
            if CRS(self.ds.proj4) != CRS(proj):
                transformer = Transformer.from_crs(CRS(proj), CRS(self.ds.proj4))
                i, j = transformer.transform(i, j)
                if verbose:
                    print(f"[Data loading] Converted to projection {self.ds.proj4}: {i, j}")
        return i, j

    def load_pixel(
        self,
        i: int | float,
        j: int | float,
        unit: int = 365,
        regu: int | str = 1,
        coef: int = 100,
        flag: xr.Dataset | None = None,
        solver: str = "LSMR",
        interp: str = "nearest",
        proj: str = "EPSG:4326",
        rolling_mean: xr.Dataset | None = None,
        visual: bool = False,
        output_format="np",
    ) -> (Optional[list], Optional[list], Optional[np.array], Optional[np.array], Optional[np.array]):  # type: ignore

        """
        Load data at pixel (i, j) and compute prior to inversion (rolling mean, mean, dates range...).

        :params i, j: [int | float] --- Coordinates to be converted
        :param unit: [int] [default is 365] --- 1 for m/d, 365 for m/y
        :param regu: [int | str] [default is 1] --- Type of regularization
        :param coef: [int] [default is 100] --- Coef of Tikhonov regularisation
        :param flag: [xr dataset | None] [default is None] --- If not None, the values of the coefficient used for stable areas, surge glacier and non surge glacier
        :param solver: [str] [default is 'LSMR'] --- Solver of the inversion: 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
        :param interp: [str] [default is 'nearest'] --- Interpolation method used to load the pixel when it is not in the dataset ('nearest' or 'linear')
        :param proj: [str] [default is 'EPSG:4326'] --- Projection of (i, j) coordinates
        :param rolling_mean: [xr dataset | None] [default is None] --- Filtered dataset (e.g. rolling mean)
        :param visual: [bool] [default is False] --- Return additional information (sensor and source) for future plots
        :param output_format [str] [default is np] --- Format of the output data (np for numpy or df for pandas dataframe)

        :return data: [list | None] --- A list 2 elements : the first one is np.ndarray with the observed
        :return mean: [list | None] --- A list with average vx and vy if solver=LSMR_ini, but the regularization do not require an apriori on the acceleration
        :return dates_range: [list | None] --- Dates between which the displacements will be inverted
        :return regu: [np array | Nothing] --- If flag is not None, regularisation method to be used for each pixel
        :return coef: [np array | Nothing] --- If flag is not None, regularisation coefficient to be used for each pixel
        """

        # Variables to keep
        var_to_keep = (
            ["date1", "date2", "vx", "vy", "errorx", "errory", "temporal_baseline"]
            if not visual
            else ["date1", "date2", "vx", "vy", "errorx", "errory", "temporal_baseline", "sensor", "source"]
        )

        if proj == "int":
            data = self.ds.isel(x=i, y=j)[var_to_keep]
        else:
            i, j = self.convert_coordinates(i, j, proj=proj)
            # Interpolate only necessary variables and drop NaN values
            if interp == "nearest":
                # 74.3 ms ± 1.33 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
                data = self.ds.sel(x=i, y=j, method="nearest")[var_to_keep]
                data = data.dropna(dim="mid_date")
            else:
                data = self.ds.interp(x=i, y=j, method=interp)[var_to_keep].dropna(
                    dim="mid_date"
                )  # 282 ms ± 12.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        if flag is not None:
            if isinstance(regu, dict) and isinstance(coef, dict):
                flag = np.round(flag["flag"].sel(x=i, y=j, method="nearest").values)
                regu = regu[flag]
                coef = coef[flag]
            else:
                raise ValueError("regu must be a dict if assign_flag is True!")

        data_dates = data[["date1", "date2"]].to_array().values.T
        if data_dates.dtype == "<M8[ns]":  # Convert to days if needed
            data_dates = data_dates.astype("datetime64[D]")

        if (solver == "LSMR_ini" or regu == "1accelnotnull" or regu == "directionxy") and rolling_mean is not None:
            if len(rolling_mean.sizes) == 3:  # if regu == 1accelnotnul, rolling_mean have a time dimesion
                # Load rolling mean for the given pixel, only on the dates available
                dates_range = construction_dates_range_np(
                    data_dates
                )  # 652 µs ± 3.24 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
                mean = rolling_mean.sel(
                    mid_date=dates_range[:-1] + np.diff(dates_range) // 2, x=i, y=j, method="nearest"
                )[["vx_filt", "vy_filt"]]
                mean = [mean[i].values / unit for i in ["vx_filt", "vy_filt"]]  # Convert it to m/day

            else:  # elif solver= LSMR_ini, rolling_mean is an average in time per pixel
                mean = rolling_mean.sel(x=i, y=j, method="nearest")[["vx", "vy"]]
                mean = [mean[i].values / unit for i in ["vx", "vy"]]  # Convert it to m/day
                dates_range = None

        else:  # If there is no apriori and no initialization
            mean = None
            dates_range = None

        # data_values is composed of vx, vy, errorx, errory, temporal baseline
        if visual:
            if output_format == "np":
                data_str = data[["sensor", "source"]].to_array().values.T
                data_values = data.drop_vars(["date1", "date2", "sensor", "source"]).to_array().values.T
                data = [data_dates, data_values, data_str]
            elif output_format == "df":
                data = data.to_pandas()
            else:
                raise ValueError(
                    "Please enter np if you want to have as output a numpy array, and df if you want a pandas dataframe"
                )
        else:
            data_values = data.drop_vars(["date1", "date2"]).to_array().values.T
            data = [data_dates, data_values]

        if flag is not None:
            return data, mean, dates_range, regu, coef
        else:
            return data, mean, dates_range

    # %% ==================================================================== #
    #                             CUBE PROCESSING                             #
    # =====================================================================%% #

    def delete_outliers(
        self,
        delete_outliers: str | float,
        flag: xr.Dataset | None = None,
        slope: xr.Dataset | None = None,
        aspect: xr.Dataset | None = None,
        direction: xr.Dataset | None = None,
        **kwargs,
    ):

        """
        Delete outliers according to a certain criterium.

        :param delete_outliers: [str | float] --- If float delete all velocities which a quality indicator higher than delete_outliers, if median_filter delete outliers that an angle 45° away from the average vector
        :param flag: [xr dataset | None] [default is None] --- If not None, the values of the coefficient used for stable areas, surge glacier and non surge glacier
        """

        if isinstance(delete_outliers, int) or isinstance(delete_outliers, str):
            if isinstance(delete_outliers, int):  # filter according to the maximal error
                inlier_mask = dask_filt_warpper(
                    self.ds["vx"], self.ds["vy"], filt_method="error", error_thres=delete_outliers
                )

            elif isinstance(delete_outliers, str):  # filter according to vcc_angle, zscore, median_angle
                axis = self.ds["vx"].dims.index("mid_date")
                inlier_mask = dask_filt_warpper(
                    self.ds["vx"],
                    self.ds["vy"],
                    filt_method=delete_outliers,
                    slope=slope,
                    aspect=aspect,
                    direction=direction,
                    axis=axis,
                    **kwargs,
                )

                if flag is not None:
                    if delete_outliers != "vvc_angle":
                        flag = flag["flag"].values if flag["flag"].shape[0] == self.nx else flag["flag"].values.T
                        flag_condition = flag == 0
                        flag_condition = np.expand_dims(flag_condition, axis=axis)
                        inlier_mask = np.logical_or(inlier_mask, flag_condition)

            inlier_flag = xr.DataArray(inlier_mask, dims=self.ds["vx"].dims)
            for var in ["vx", "vy"]:
                self.ds[var] = self.ds[var].where(inlier_flag)

            self.ds = self.ds.persist()

        elif isinstance(delete_outliers, dict):
            for method in delete_outliers.keys():
                if method == "error":
                    if delete_outliers["error"] is None:
                        self.delete_outliers("error", flag)
                    else:
                        self.delete_outliers(delete_outliers["error"], flag)
                elif method == "magnitude":
                    if delete_outliers["magnitude"] is None:
                        self.delete_outliers("magnitude", flag)
                    else:
                        self.delete_outliers("magnitude", flag, magnitude_thres=delete_outliers["magnitude"])
                elif method == "median_magnitude":
                    if delete_outliers["median_magnitude"] is None:
                        self.delete_outliers("median_magnitude", flag)
                    else:
                        self.delete_outliers(
                            "median_magnitude", flag, median_magnitude_thres=delete_outliers["median_magnitude"]
                        )
                elif method == "z_score":
                    if delete_outliers["z_score"] is None:
                        self.delete_outliers("z_score", flag)
                    else:
                        self.delete_outliers("z_score", flag, z_thres=delete_outliers["z_score"])

                elif method == "median_angle":
                    if delete_outliers["median_angle"] is None:
                        self.delete_outliers("median_angle", flag)
                    else:
                        self.delete_outliers("median_angle", flag, z_thres=delete_outliers["median_angle"])

                elif method == "vvc_angle":
                    if delete_outliers["vvc_angle"] is None:
                        self.delete_outliers("vvc_angle", flag)
                    else:
                        self.delete_outliers("vvc_angle", flag, **delete_outliers["vvc_angle"])
                elif method == "topo_angle":
                    self.delete_outliers("topo_angle", flag, slope=slope, aspect=aspect)
                elif method == "flow_angle":
                    self.delete_outliers("flow_angle", flag, direction=direction)
                elif method == "mz_score":
                    if delete_outliers["mz_score"] is None:
                        self.delete_outliers("mz_score", flag)
                    else:
                        self.delete_outliers("mz_score", flag, z_thres=delete_outliers["mz_score"])
                else:
                    raise ValueError(
                        f"Filtering method should be either 'median_angle', 'vvc_angle', 'topo_angle', 'z_score','mz_score', 'magnitude', 'median_magnitude' or 'error'."
                    )
        else:
            raise ValueError("delete_outliers must be a int, a string or a dict, not {type(delete_outliers)}")

    def mask_cube(self, mask: xr.DataArray | str):

        """
        Mask some of the data of the cube (putting it to np.nan).

        :param mask: [str | xr dataarray] --- Either a DataArray with 1 the data to keep and 0 the ones to remove, or a path to a file containing a DataArray or a shapefile to be rasterized
        """

        if type(mask) is str:
            if mask[-3:] == "shp":  # Convert the shp file to an xarray dataset (rasterize the shapefile)
                polygon = geopandas.read_file(mask).to_crs(CRS(self.ds.proj4))
                raster = rasterize(
                    [polygon.geometry[0]],
                    out_shape=self.ds.rio.shape,
                    transform=self.ds.rio.transform(),
                    fill=0,
                    dtype="int16",
                )
                mask = xr.DataArray(data=raster.T, dims=["x", "y"], coords=self.ds[["x", "y"]].coords)
            else:
                mask = xr.open_dataarray(mask)
            mask.load()

        # Mask the velocities and the errors
        if not self.is_TICO:
            self.ds[["vx", "vy", "errorx", "errory"]] = (
                self.ds[["vx", "vy", "errorx", "errory"]]
                .where(mask.sel(x=self.ds.x, y=self.ds.y, method="nearest") == 1)
                .astype("float32")
            )
        else:
            self.ds[["dx", "dy", "xcount_x", "xcount_y"]] = (
                self.ds[["dx", "dy", "xcount_x", "xcount_y"]]
                .where(mask.sel(x=self.ds.x, y=self.ds.y, method="nearest") == 1)
                .astype("float32")
            )

    def reproject_geotiff_to_cube(self, file_path):

        """
        Reproject the geotiff file to the same geometry of the cube
        :param: file_path: [str] --- path of the geotifffile to be wrapped
        :return: warped data [np.ndarray] --- warped data with same shape and resolution as the cube
        """
        if file_path.split(".")[-1] == "tif":
            with rio.open(file_path) as src:
                src_data = src.read(1)

            dst_data = np.empty(shape=self.ds.rio.shape, dtype=np.float32)
            dst_data, _ = rio.warp.reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_crs=CRS.from_proj4(self.ds.proj4),
                dst_transform=self.ds.rio.transform(),
                dst_shape=self.ds.rio.shape,
                resampling=rio.warp.Resampling.bilinear,
            )
            dst_data[dst_data == src.nodata] = np.nan
        return dst_data

    def compute_flow_direction(self, vx_file: str | None = None, vy_file: str | None = None) -> xr.DataArray:

        """
        Compute the avaerage flow direction from the input vx and vy files or just from the observations
        :param: vx_file | vy_file: [str] --- path of the flow velocity file, should be geotiff format
        :return: direction: [xr.DataArray] --- computed average flow direction at each pixel
        """
        if vx_file is not None and vy_file is not None:
            vx = self.reproject_geotiff_to_cube(vx_file)
            vy = self.reproject_geotiff_to_cube(vy_file)
        else:
            vx = self.ds["vx"]
            vy = self.ds["vy"]

        temporal_baseline = self.ds["temporal_baseline"].values
        temporal_baseline = temporal_baseline[np.newaxis, np.newaxis, :]
        vx_weighted = np.nansum(vx.values * temporal_baseline, axis=2) / np.nansum(temporal_baseline, axis=2)
        vy_weighted = np.nansum(vy.values * temporal_baseline, axis=2) / np.nansum(temporal_baseline, axis=2)

        v_mean_weighted = np.sqrt(vx_weighted**2 + vy_weighted**2)

        direction = np.arctan2(vx_weighted, vy_weighted)
        direction = (np.rad2deg(direction) + 360) % 360

        direction = np.where(v_mean_weighted < 1.5, np.nan, direction)

        direction = xr.Dataset(
            data_vars=dict(
                direction=(["y", "x"], np.array(direction.T)),
            ),
            coords=dict(x=(["x"], self.ds.x.data), y=(["y"], self.ds.y.data)),
        )

        return direction

    def compute_slo_asp(self, dem_file: str, blur_size: int = 5) -> (xr.DataArray, xr.DataArray):
        """

        :param dem_file: [str] --- path of the DEM
        :param blur_size: [int] --- spatial smoothing to apply to the DEM
        :return: slope [xr.DataArray] --- slope of the smoothed DEM
        :return: aspect [xr.DataArray] --- aspect of the smoothed D DEM

        """

        # decorator to define a generator-based context manager. This allows the user to use the with statement with a generator function, here to remove every printed messages
        @contextlib.contextmanager
        def suppress_stdout_stderr():
            with open(os.devnull, "w") as devnull:
                old_stdout = os.dup(1)
                old_stderr = os.dup(2)
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                try:
                    yield
                finally:
                    os.dup2(old_stdout, 1)
                    os.dup2(old_stderr, 2)

        if CRS.from_proj4(self.ds.proj4) == CRS.from_epsg(4326):
            raise ValueError("The CRS of the cube must be projected in meters for calculating slope and aspect")
        # Open the DEM file
        dem = self.reproject_geotiff_to_cube(dem_file)

        # Set no_data value if src.nodata is None
        with rio.open(dem_file) as src:
            no_data = src.nodata if src.nodata is not None else -9999

        dem = median_filter(dem, size=blur_size)  # Blur the DEM, should be done first before computing slope and aspect
        # Create richdem array with suppressed output, richDEM is very quick for even very large DEMs.
        with suppress_stdout_stderr():
            dem_rd = rd.rdarray(dem, no_data=no_data)

        dem_rd.geotransform = self.ds.rio.transform().to_gdal()

        # Calculate slope and aspect with suppressed output message
        with suppress_stdout_stderr():
            slope = rd.TerrainAttribute(dem_rd, attrib="slope_degrees")
            aspect = rd.TerrainAttribute(dem_rd, attrib="aspect")

        # Replace no_data values in slope and aspect with NaN
        slope[slope == slope.no_data] = np.nan
        aspect[aspect == aspect.no_data] = np.nan

        # Convert slope and aspect to xarray Datasets
        slope = xr.Dataset(
            data_vars=dict(
                slope=(["y", "x"], np.array(slope)),
            ),
            coords=dict(x=(["x"], self.ds.x.data), y=(["y"], self.ds.y.data)),
        )
        aspect = xr.Dataset(
            data_vars=dict(
                aspect=(["y", "x"], np.array(aspect)),
            ),
            coords=dict(x=(["x"], self.ds.x.data), y=(["y"], self.ds.y.data)),
        )

        return slope, aspect

    def create_flag(self, flag: str = None, field_name: str | None = None, default_value: str | int | None = None):

        """
        Create a flag dataset based on the provided shapefile and shapefile field.
        Which is usually used to divide the pixels into different types, especially for surging glaciers.
        If you just want to divide by polygon, set the shp_field to None

        :param flag (str, optional): The path to the shapefile. Defaults to None.
        :param shp_field (str, optional): The name of the shapefile field. Defaults to 'surge_type' (used in RGI7).
        :param default_value (str | int | None, optional): The default value for the shapefile field. Defaults to 0.
        :Returns flag: xr.Dataset, The flag dataset with dimensions 'y' and 'x'.
        """

        if isinstance(flag, str):
            if flag.split(".")[-1] == "nc":  # If flag is a netCDF file
                flag = xr.open_dataset(flag)

            elif flag.split(".")[-1] in ["shp", "gpkg"]:  # If flag is a shape file
                flag = geopandas.read_file(flag).to_crs(self.ds.proj4).clip(self.ds.rio.bounds())

                # surge-type glacier: 2, other glacier: 1, stable area: 0
                if field_name is None:
                    if "surge_type" in flag.columns:  # RGI inventory, surge-type glacier: 2, other glacier: 0
                        default_value = 0
                        field_name = "surge_type"
                    elif (
                        "Surge_class" in flag.columns
                    ):  # HMA surging glacier inventory, surge-type glacier: 2, other glacier: ''
                        default_value = None
                        field_name = "Surge_class"

                if field_name is not None:
                    flag_id = flag[field_name].apply(lambda x: 2 if x != default_value else 1).astype("int16")
                    geom_value = ((geom, value) for geom, value in zip(flag.geometry, flag_id))
                else:
                    # inside the polygon: 1, outside: 0
                    geom_value = ((geom, 1) for geom in flag.geometry)

                try:
                    flag = rasterio.features.rasterize(
                        geom_value,
                        out_shape=(self.ny, self.nx),
                        transform=self.ds.rio.transform(),
                        all_touched=True,
                        fill=0,  # background value
                        dtype="int16",
                    )
                except:
                    flag = np.zeros(shape=(self.ny, self.nx), dtype="int16")

                flag = xr.Dataset(
                    data_vars=dict(
                        flag=(["y", "x"], flag),
                    ),
                    coords=dict(
                        x=(["x"], self.ds.x.data),
                        y=(["y"], self.ds.y.data),
                    ),
                )

            elif not isinstance(flag, xr.Dataset):
                raise ValueError("flag file must be .nc or .shp")

        if "flags" in list(flag.variables):
            flag = flag.rename({"flags": "flag"})

        return flag

    def filter_cube(
        self,
        i: int | float | None = None,
        j: int | float | None = None,
        smooth_method: str = "gaussian",
        s_win: int = 3,
        t_win: int = 90,
        sigma: int = 3,
        order: int = 3,
        unit: int = 365,
        delete_outliers: str | float | None = None,
        flag: xr.Dataset | str | None = None,
        dem_file: str | None = None,
        regu: int | str = 1,
        solver: str = "LSMR_ini",
        proj: str = "EPSG:4326",
        velo_or_disp: str = "velo",
        select_baseline: int | None = None,
        verbose: bool = False,
    ) -> xr.Dataset:

        """
           Filter the original data before the inversion:
        -delete outliers according to the provided criterium
        -compute a spatio-temporal kernel of the data, which can be used as apriori for the inversion (for "1accelnotnull" or "directionxy" )
        -compute mean velocity along x and y ( for solver = 'LSMR_ini' if regu is not "1accelnotnull" or "directionxy" )

        :params i, j: [int | float] --- Coordinates to be converted
        :param smooth_method: [str] [default is 'gaussian'] --- Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
        :param s_win: [int] [default is 3] --- Size of the spatial window
        :param t_win: [int] [default is 90] --- Time window size for 'ewma' smoothing
        :param sigma: [int] [default is 3] --- Standard deviation for 'gaussian' filter
        :param order: [int] [default is 3] --- Order of the smoothing function
        :param unit: [int] [default is 365] --- 365 if the unit is m/y, 1 if the unit is m/d
        :param delete_outliers: [str | float | None] [default is None] --- If float delete all velocities which a quality indicator higher than delete_outliers
        :param flag: [xr dataset | None] [default is None] --- If not None, the values of the coefficient used for stable areas, surge glacier and non surge glacier
        :param regu: [int | str] [default is 1] --- Regularisation of the solver
        :param solver: [str] [default is 'LSMR_ini'] --- Solver used to invert the system
        :param proj: [str] [default is 'EPSG:4326'] --- EPSG of i,j projection
        :param velo_or_disp: [str] [default is 'velo'] --- 'disp' or 'velo' to indicate the type of the observations : 'disp' mean that self contain displacements values and 'velo' mean it contains velocity
        :param select_baseline: [int | None] [default is None] --- threshold of the temporal baseline to select, if the number of observation is lower than 3 times the number of estimated displacement with this threshold, it is increased by 30 days
        :param verbose: [bool] [default is False] --- Print information throughout the process

        :return obs_filt: [xr dataset | None] --- Filtered dataset
        """

        def loop_rolling(da_arr: xr.Dataset, select_baseline: int | None = None) -> (np.ndarray, np.ndarray):  # type: ignore

            """
            A function to calculate spatial mean, resample data, and calculate exponential smoothed velocity.

            :param da_arr: [xr dataset] --- Original data
            :param select_baseline: [int] [default is 200] --- Threshold over the temporal baselines

            :return spatial_mean: [np array] --- Exponential smoothed velocity
            :return date_out: [np array] --- Observed dates
            """

            # Compute the dates of the estimated displacements time series
            date_out = date_range[:-1] + np.diff(date_range) // 2
            mid_dates = self.ds["mid_date"]

            if verbose:
                start = time.time()
            if select_baseline is not None:
                baseline = self.ds["temporal_baseline"].compute()
                idx = np.where(
                    baseline < select_baseline
                )  # Take only the temporal baseline lower than the threshold selecr_baseline
                while len(idx[0]) < 3 * len(date_out) & (
                    select_baseline < 200
                ):  # Increase the threshold by 30, if the number of observation is lower than 3 times the number of estimated displacement
                    # while (len(idx[0]) < 3 * len(
                    #             date_out)):
                    select_baseline += 30
                    idx = np.where(baseline < select_baseline)
                mid_dates = mid_dates.isel(mid_date=idx[0])
                da_arr = da_arr.isel(mid_date=idx[0])

            # Find the time axis for dask processing
            time_axis = self.ds["vx"].dims.index("mid_date")
            # Apply the selected kernel in time
            if verbose:
                with ProgressBar():  # Plot a progress bar
                    filtered_in_time = dask_smooth_wrapper(
                        da_arr.data,
                        mid_dates,
                        t_out=date_out,
                        smooth_method=smooth_method,
                        sigma=sigma,
                        t_win=t_win,
                        order=order,
                        axis=time_axis,
                    ).compute()
            else:
                filtered_in_time = dask_smooth_wrapper(
                    da_arr.data,
                    mid_dates,
                    t_out=date_out,
                    smooth_method=smooth_method,
                    sigma=sigma,
                    t_win=t_win,
                    order=order,
                    axis=time_axis,
                ).compute()

            if verbose:
                print(f"[Data filtering] Smoothing observations took {round((time.time() - start), 1)} s")

            # Spatial average
            if (
                np.min([da_arr["x"].size, da_arr["y"].size]) > s_win
            ):  # The spatial average is performed only if the size of the cube is larger than s_win, the spatial window
                spatial_axis = tuple(i for i in range(3) if i != time_axis)
                pad_widths = tuple((s_win // 2, s_win // 2) if i != time_axis else (0, 0) for i in range(3))
                spatial_mean = da.nanmedian(
                    sliding_window_view(filtered_in_time, (s_win, s_win), axis=spatial_axis), axis=(-1, -2)
                )
                spatial_mean = da.pad(spatial_mean, pad_widths, mode="edge")
            else:
                spatial_mean = filtered_in_time

            return spatial_mean.compute(), np.unique(date_out)

        if np.isnan(self.ds["date1"].values).all():
            print("[Data filtering] Empty sub-cube (masked data ?)")
            return None

        if i is not None and j is not None:  # Crop the cube dataset around a given pixel
            i, j = self.convert_coordinates(i, j, proj=proj, verbose=verbose)
            if verbose:
                print(f"[Data filtering] Clipping dataset to individual pixel: (x, y) = ({i},{j})")
            buffer = (s_win + 2) * (self.ds["x"][1] - self.ds["x"][0])
            self.buffer(self.ds.proj4, [i, j, buffer])
            self.ds = self.ds.unify_chunks()

        # The rolling smooth should be carried on velocity, while we need displacement during inversion
        if velo_or_disp == "disp":  # to provide velocity values
            self.ds["vx"] = self.ds["vx"] / self.ds["temporal_baseline"] * unit
            self.ds["vy"] = self.ds["vy"] / self.ds["temporal_baseline"] * unit

        if flag is not None:
            flag = self.create_flag(flag)
            flag.load()

            if isinstance(regu, dict):
                regu = list(regu.values())
            else:
                raise ValueError("regu must be a dict if flag is Not None")
        else:
            if isinstance(regu, int):  # if regu is an integer
                regu = [regu]
            elif isinstance(regu, str):  # if regu is a string
                regu = list(regu.split())

        start = time.time()

        if delete_outliers is not None:
            slope, aspect, direction = None, None, None
            if (isinstance(delete_outliers, str) and delete_outliers == "topo_angle") or (
                isinstance(delete_outliers, dict) and "topo_angle" in delete_outliers.keys()
            ):
                if isinstance(dem_file, str):
                    slope, aspect = self.compute_slo_asp(dem_file=dem_file)
                else:
                    raise ValueError("dem_file must be given if delete_outliers is 'topo_angle'")

            elif (isinstance(delete_outliers, str) and delete_outliers == "flow_angle") or (
                isinstance(delete_outliers, dict) and "flow_angle" in delete_outliers.keys()
            ):
                direction = self.compute_flow_direction(vx_file=None, vy_file=None)
            self.delete_outliers(
                delete_outliers=delete_outliers, flag=None, slope=slope, aspect=aspect, direction=direction
            )
            if verbose:
                print(f"[Data filtering] Delete outlier took {round((time.time() - start), 1)} s")

        if "1accelnotnull" in regu or "directionxy" in regu:
            date_range = np.sort(
                np.unique(
                    np.concatenate(
                        (
                            self.ds["date1"].values[~np.isnan(self.ds["date1"].values)],
                            self.ds["date2"].values[~np.isnan(self.ds["date2"].values)],
                        ),
                        axis=0,
                    )
                )
            )
            if verbose:
                start = time.time()

            vx_filtered, dates_uniq = loop_rolling(self.ds["vx"], select_baseline=select_baseline)
            vy_filtered, dates_uniq = loop_rolling(self.ds["vy"], select_baseline=select_baseline)

            # The time dimension of the smoothed velocity observations is different from the original,
            # which is because of the possible duplicate mid_date of different image pairs...
            obs_filt = xr.Dataset(
                data_vars=dict(
                    vx_filt=(["x", "y", "mid_date"], vx_filtered), vy_filt=(["x", "y", "mid_date"], vy_filtered)
                ),
                coords=dict(x=(["x"], self.ds.x.data), y=(["y"], self.ds.y.data), mid_date=dates_uniq),
                attrs=dict(description="Smoothed velocity observations", units="m/y", proj4=self.ds.proj4),
            )
            del vx_filtered, vy_filtered

            if verbose:
                print(
                    "[Data filtering] Calculating smoothing mean of the observations completed in {:.2f} seconds".format(
                        time.time() - start
                    )
                )

        elif (
            solver == "LSMR_ini"
        ):  # The initialization is based on the averaged velocity over the period, for every pixel
            obs_filt = self.ds[["vx", "vy"]].mean(dim="mid_date")
            obs_filt.attrs["description"] = "Averaged velocity over the period"
            obs_filt.attrs["units"] = "m/y"
        else:
            obs_filt = None

        # Unify the observations to displacement to provide displacement values during inversion
        self.ds["vx"] = self.ds["vx"] * self.ds["temporal_baseline"] / unit
        self.ds["vy"] = self.ds["vy"] * self.ds["temporal_baseline"] / unit

        if obs_filt != None:
            obs_filt.load()
        self.ds = self.ds.load()  # Crash memory without loading
        # persist() is particularly useful when using a distributed cluster because the data will be loaded into distributed memory across your machines and be much faster to use than reading repeatedly from disk.

        return obs_filt, flag

    def split_cube(self, n_split: int = 2, dim: str | list = "x", savepath: str | None = None):

        """
        Split the cube into smaller cubes (taking less memory to load) according to the given dimensions.

        :param n_split: [int] [default is 2] --- Number of split to compute along each dimensions in dim
        :param dim: [str | list] [default is "x"] --- Dimension.s along which must be split the cube
        :param savepath: [str | None] [default is None] --- If not None, save the new cubes at this location

        :return cubes: [dict] --- Dictionary of the splitcubes (keys describe the position of the cube)
        """

        cubes = dict()
        for s in range(n_split):
            if isinstance(dim, str):
                cube = cube_data_class(
                    self,
                    self.ds.isel(
                        {dim: slice(s * len(self.ds[dim].values) // 2, (s + 1) * len(self.ds[dim].values) // 2, 1)}
                    ),
                )
                if savepath is not None:
                    cube.ds.to_netcdf(f"{savepath}{dim}_{s}.nc")
                    print(f"Split cube saved at {savepath}{dim}_{s}.nc")
                cubes[f"{savepath.split('/')[-1]}{dim}_{s}"] = cube
            elif isinstance(dim, list):
                cube = cube_data_class(
                    self,
                    self.ds.isel(
                        {
                            dim[0]: slice(
                                s * len(self.ds[dim[0]].values) // 2, (s + 1) * len(self.ds[dim[0]].values) // 2, 1
                            )
                        }
                    ),
                )
                if len(dim) > 1:
                    cubes |= cube.split_cube(n_split=n_split, dim=dim[1:], savepath=f"{savepath}{dim[0]}_{s}_")
                else:
                    if savepath is not None:
                        cube.ds.to_netcdf(f"{savepath}{dim[0]}_{s}.nc")
                        print(f"Split cube saved at {savepath}{dim[0]}_{s}.nc")
                    cubes[f"{savepath.split('/')[-1]}{dim}_{s}"] = cube

        return cubes

    def align_cube(
        self,
        cube: "cube_data_class",
        unit: int = 365,
        reproj_vel: bool = True,
        reproj_coord: bool = True,
        interp_method: str = "nearest",
    ):

        """
        Reproject cube to match the resolution, projection, and region of self.

        :param cube: Cube to align to self
        :param unit: Unit of the velocities (365 for m/y, 1 for m/d) (default is 365)
        :param reproj_vel: Whether the velocity have to be reprojected or not -> it will modify their value (default is True)
        :param reproj_coord: Whether the coordinates have to be interpolated or not (using interp_method) (default is True)
        :param interp_method: Interpolation method used to reproject cube (default is 'nearest')

        :return: Cube projected to self
        """
        # TODO: fix bug due to the order of cube.ds could be different from 'mid_date', 'y', 'x'
        if reproj_vel:  # if the velocity components have to be reprojected in the new projection system
            grid = np.meshgrid(cube.ds["x"], cube.ds["y"])
            temp = cube.temp_base_()
            endx = np.array(
                [(np.ma.masked_invalid(cube.ds["vx"][z]) * temp[z] / unit) + grid[0] for z in range(cube.nz)]
            )  # localisation of the final coordinate of each pixel displaced by the corresponding velocity vector, in x
            endy = np.array(
                [(np.ma.masked_invalid(cube.ds["vy"][z]) * temp[z] / unit) + grid[1] for z in range(cube.nz)]
            )  # localisation of the final coordinate of each pixel displaced by the corresponding velocity vector, in y

            # reprojection of the final coordinate of each pixel displaced by the corresponding velocity vector
            transformer = Transformer.from_crs(cube.ds.proj4, self.ds.proj4)
            t = np.array([transformer.transform(endx[z], endy[z]) for z in range(cube.nz)])
            del endx, endy

            # Computation of the difference between final and oringinal coordinates in the new system
            grid = transformer.transform(grid[0], grid[1])
            vx = np.array(
                [(grid[0] - t[z, 0, :, :]) / temp[z] * unit for z in range(cube.nz)]
            )  # positive toward the West
            vy = np.array(
                [(t[z, 1, :, :] - grid[1]) / temp[z] * unit for z in range(cube.nz)]
            )  # positive toward the North
            cube.ds["vx"] = xr.DataArray(
                vx.astype("float32"),
                dims=["mid_date", "y", "x"],
                coords={"mid_date": cube.ds.mid_date, "y": cube.ds.y, "x": cube.ds.x},
            )
            cube.ds["vx"].encoding = {"vx": {"dtype": "float32", "scale_factor": 0.1, "units": "m/y"}}
            cube.ds["vy"] = xr.DataArray(
                vy.astype("float32"),
                dims=["mid_date", "y", "x"],
                coords={"mid_date": cube.ds.mid_date, "y": cube.ds.y, "x": cube.ds.x},
            )
            cube.ds["vy"].encoding = {"vy": {"dtype": "float32", "scale_factor": 0.1, "units": "m/y"}}
            del vx, vy

            # if reproj_coord:
        #     # Convert the system of coordinate and adjust the spatial resolution of self to match the resolution, projection, and region of cube
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
            # Convert the system of coordinate and adjust the spatial resolution of the cube2 to match the resolution, projection, and region of self, using a bilinear interpolation
            cube.ds = cube.ds.rio.write_crs(cube.ds.proj4)
            self.ds = self.ds.rio.write_crs(self.ds.proj4)
            cube.ds = cube.ds.transpose("mid_date", "y", "x")
            # Reproject coordinates
            if interp_method == "nearest":
                cube.ds = cube.ds.rio.reproject_match(self.ds, resampling=rasterio.enums.Resampling.nearest)
            # Reject abnormal data (when the cube sizes are not the same and data are missing, the interpolation leads to infinite or nearly-infinite values)
            cube.ds[["vx", "vy"]] = cube.ds[["vx", "vy"]].where(
                (np.abs(cube.ds["vx"].values) < 10000) | (np.abs(cube.ds["vy"].values) < 10000), np.nan
            )

            # Update of cube_data_classxr attributes
            cube.ds = cube.ds.assign_attrs({"proj4": self.ds.proj4})
            cube.nx = cube.ds.dims["x"]
            cube.ny = cube.ds.dims["y"]
            cube.ds = cube.ds.assign_coords({"x": cube.ds.x, "y": cube.ds.y})

        cube.ds = cube.ds.assign_attrs({"author": f"{cube.ds.author} aligned"})

        return cube

    def merge_cube(self, cube: "cube_data_class"):

        """
        Merge another cube to the present one. It must have been aligned first (using align_cube)

        :param cube: [cube_data_class] --- The cube to be merged to self
        """

        # Merge the cubes (must be previously aligned before using align_cube)
        self.ds = xr.concat([self.ds, cube.ds.sel(x=self.ds["x"], y=self.ds["y"])], dim="mid_date")

        # Update the attributes
        self.ds = self.ds.chunk(chunks={"mid_date": self.ds["mid_date"].size})
        self.nz = self.ds["mid_date"].size
        if (
            type(self.filedir) != list
            and type(self.filename) != list
            and type(self.author) != list
            and type(self.source) != list
        ):
            self.filedir = [self.filedir]
            self.filename = [self.filename]
            self.author = [self.author]
            self.source = [self.source]
        self.filedir.append(cube.filedir)
        self.filename.append(cube.filename)
        self.author.append(cube.author)
        self.source.append(cube.source)

    def average_cube(
        self,
        return_format: str = "geotiff",
        return_variable: list = ["vv"],
        save: bool = True,
        path_save: str | None = None,
    ):

        """
        Compute the mean velocity at each pixel of he cube.

        :param return_format: [str] [default is 'geotiff'] --- Type of the file to be returned ('nc' or 'geotiff')
        :param return_variable: [list] [default is ['vv']] --- Which variable's mean must be returned
        :param save: [bool] [default is True] --- If True, save the file to path_save
        :param path_save: [str | None] [default is None] --- Path where to save the mean velocity file

        :return: xr dataset, with vx_mean, the mean of vx and vy_mean the mean of vy
        """

        vx_mean = self.ds["vx"].mean(dim="mid_date")
        vy_mean = self.ds["vy"].mean(dim="mid_date")
        dico_variable = {"vx": vx_mean, "vx": vy_mean}
        if "vv" in return_variable:
            vv_mean = np.sqrt(vx_mean**2 + vy_mean**2)
            dico_variable["vv"] = vv_mean

        if return_format == "nc":
            ds_mean = xr.Dataset({})
            coords = {"y": self.ds.y, "x": self.ds.x}
            for variable in return_variable:
                ds_mean[f"{variable}_mean"] = xr.DataArray(dico_variable[variable], dims=["y", "x"], coords=coords)
            if save:
                ds_mean.to_netcdf(path_save)
            return ds_mean

        elif return_format == "geotiff":
            ds_mean = []
            for variable in return_variable:
                mean_v = dico_variable[variable].to_numpy().astype(np.float32)
                mean_v = np.flip(mean_v.T, axis=0)

                if save:
                    # Create the GeoTIFF file
                    with rasterio.open(
                        f"{path_save}/mean_velocity_{variable}.tif",
                        "w",
                        driver="GTiff",
                        height=mean_v.shape[0],
                        width=mean_v.shape[1],
                        count=1,
                        dtype=str(mean_v.dtype),
                        crs=CRS.from_proj4(self.ds.proj4),
                        transform=self.ds.rio.transform(),
                    ) as dst:
                        dst.write(mean_v, 1)

                ds_mean.append(mean_v)

            return ds_mean
        else:
            raise ValueError("Please enter geotiff or nc")

    def compute_heatmap_moving(
        self,
        points_heatmap: pd.DataFrame,
        variable: str = "vv",
        method_interp: str = "linear",
        verbose: bool = False,
        freq: str = "MS",
        method: str = "mean",
    ) -> pd.DataFrame:

        """
        Compute a heatmap of the average monthly velocity, average all the velocities which are overlapping a given month

        :param points_heatmap: Points where the heatmap is to be computed
        :param variable: What variable is to be computed ('vx', 'vy' or 'vv')
        :param method_interp: Interpolation method used to determine the value at a specified point from the discrete velocities data
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
        data = np.ma.array(sorted(data, key=lambda date: date[0]))  # Sort according to the first date

        # Find the index of the dates that have to be averaged, to get the heatmap
        # Each value of the heatmap corresponds to an average of all the velocities which are overlapping a given period
        save_line = [[] for _ in range(len(date_range) - 1)]
        for i_date, _ in enumerate(date_range[:-1]):
            i = 0
            while i < data.shape[0] and date_range[i_date + 1] >= data[i, 0]:
                if date_range[i_date] <= data[i, 1]:
                    save_line[i_date].append(i)
                i += 1
        interval_output = pd.Series(
            [(date_range[k + 1] - date_range[k]) / np.timedelta64(1, "D") for k in range(date_range.shape[0] - 1)]
        )
        dates_c = date_range[1:] - pd.to_timedelta((interval_output / 2).astype("int"), "D")
        del interval_output, date_range, data

        def data_temporalpoint(k: int, points_heatmap):

            """Get the data at a given spatial point contained in points_heatmap"""

            geopoint = points_heatmap["geometry"].iloc[
                k
            ]  # Return a point at the specified distance along a linear geometric object. # True -> interpretate k/n as fraction and not meters

            i, j = geopoint.x, geopoint.y
            if verbose:
                print("i,j", i, j)

            if variable == "vv":
                v = np.sqrt(
                    self.ds["vx"].interp(x=i, y=j, method=method_interp).load() ** 2
                    + self.ds["vy"].interp(x=i, y=j, method="linear").load() ** 2
                )
            elif variable == "vx" or variable == "vy":
                v = self.ds[variable].interp(x=i, y=j, method=method_interp).load()

            data = np.array([date1, date2, v.values], dtype=object).T
            data = np.ma.array(sorted(data, key=lambda date: date[0]))  # Slort according to the first date

            return data[:, 2]

        for k in range(len(points_heatmap)):
            if verbose:
                print("k", k)

            data = data_temporalpoint(k, points_heatmap)
            vvmasked = np.ma.masked_invalid(np.ma.array(data, dtype="float"))

            if method == "mean":
                vvmean = [np.ma.mean(vvmasked[lines]) for lines in save_line]
            elif method == "median":
                vvmean = [np.ma.median(vvmasked[lines]) for lines in save_line]

            vvdf = pd.DataFrame(vvmean, index=dates_c, columns=[points_heatmap["distance"].iloc[k] / 1000])

            if k > 0:
                line_df_vv = pd.concat([line_df_vv, vvdf], join="inner", axis=1)
            else:
                line_df_vv = vvdf

        return line_df_vv

    # @jit(nopython=True)
    def ncvv(self):
        """Return the Normalized Coherence Vector Velocity"""
        return np.array(
            [
                np.sqrt(
                    np.nansum(
                        self.ds["vx"].isel(x=i, y=j)
                        / np.sqrt(self.ds["vx"].isel(x=i, y=j) ** 2 + self.ds["vy"].isel(x=i, y=j) ** 2)
                    )
                    ** 2
                    + np.nansum(
                        self.ds["vy"].isel(x=i, y=j)
                        / np.sqrt(self.ds["vx"].isel(x=i, y=j) ** 2 + self.ds["vy"].isel(x=i, y=j) ** 2)
                    )
                    ** 2
                )
                / self.nz
                for i in range(self.nx)
                for j in range(self.ny)
            ]
        ).reshape(self.nx, self.ny)

    # %% ======================================================================== #
    #                            WRITING RESULTS AS NETCDF                        #
    # =========================================================================%% #

    def write_result_ticoi(
        self,
        result: list,
        source: str,
        sensor: str,
        filename: str = "Time_series",
        savepath: str | None = None,
        result_quality: list | None = None,
        smooth_res: bool = False,
        smooth_window_size: int = 3,
        smooth_filt: np.ndarray | None = None,
        return_result: bool = False,
        verbose: bool = False,
    ) -> Union["cube_data_class", str]:

        """
        Write the result from TICOI, stored in result, in a xarray dataset matching the conventions CF-1.10
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.pdf
        units has been changed to unit, since it was producing an error while writing the netcdf file

        :param result: [list] --- List of pd xarray, resulut from the TICOI method
        :param source: [str] --- Name of the source
        :param sensor: [str] --- Sensors which have been used
        :param filename: [str] [default is Time_series] --- Filename of file to saved
        :param result_quality: [list | str | None] [default is None] --- Which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)):param savepath: string, path where to save the file
        :param verbose: [bool] [default is None] --- Print information throughout the process (default is False)

        :return cubenew: [cube_data_class] --- New cube where the results are saved
        """

        # TODO: need to check the order of dimension: do we need to transpose?
        non_null_results = [
            result[i * self.ny + j]["vx"].shape[0]
            for i in range(self.nx)
            for j in range(self.ny)
            if result[i * self.ny + j]["vx"].shape[0] != 0
        ]  # Temporal size of the results which are not empty
        first_date_results = [
            result[i * self.ny + j]["date1"].iloc[0]
            for i in range(self.nx)
            for j in range(self.ny)
            if result[i * self.ny + j]["vx"].shape[0] != 0
        ]  # Temporal size of the results which are not empty
        if len(non_null_results) == 0:
            print("[Writing results] There is no results to write and/or save")
            return "There is no results to write and/or save"

        if np.min(non_null_results) == np.max(non_null_results) and all(
            element == first_date_results[0] for element in first_date_results
        ):  # If the dates of the results are the same for every pixel
            non_null_el = next(
                (element for element in result if element.shape[0] != 0), None
            )  # First result array which is not empty, and have size corresponding to the time period common between every pixel
            del non_null_results, first_date_results
        else:
            print("Not the same time dimension for every pixels")
            raise ValueError("Not the same time dimension for every pixels, cannot save cube")

        cubenew = cube_data_class()
        time_variable = non_null_el["date1"] + (non_null_el["date2"] - non_null_el["date1"]) // 2
        cubenew.ds["date1"] = xr.DataArray(non_null_el["date1"], dims="mid_date", coords={"mid_date": time_variable})
        cubenew.ds["date1"].attrs = {
            "standard_name": "date1",
            "unit": "days",
            "long_name": "first date between which the velocity is estimated",
        }
        cubenew.ds["date2"] = xr.DataArray(non_null_el["date2"], dims="mid_date", coords={"mid_date": time_variable})
        cubenew.ds["date2"].attrs = {
            "standard_name": "date2",
            "unit": "days",
            "long_name": "second date between which the velocity is estimated",
        }

        long_name = [
            "velocity in the East/West direction [m/y]",
            "velocity in the North/South direction [m/y]",
            "number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)) in the East/West direction",
            "number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)) in the North/South direction",
            "Error propagated for the displacement in the direction Est/West [m/y]",
            "Error propagated for  the displacement in the direction North/South [m/y]",
        ]
        short_name = ["x_velocity", "y_velocity", "xcount_x", "xcount_y", "error_x", "error_y"]

        variables = ["vx", "vy"]
        if result_quality is not None:
            if "X_contribution" in result_quality:
                variables += ["xcount_x", "xcount_y"]
            if "Error_propagation" in result_quality:
                variables += ["error_x", "error_y"]

        for i, var in enumerate(variables):
            result_arr = np.array(
                [
                    result[i * self.ny + j][var]
                    if result[i * self.ny + j][var].shape[0] != 0
                    else pd.Series(np.full(non_null_el.shape[0], np.nan))
                    for i in range(self.nx)
                    for j in range(self.ny)
                ]
            )
            result_arr = result_arr.reshape((self.nx, self.ny, len(time_variable)))

            # Spatially smooth (apply a convolutional filter on) the data
            if smooth_res:
                result_arr = smooth_results(result_arr, window_size=smooth_window_size, filt=smooth_filt)
                for x in range(self.nx):
                    for y in range(self.ny):
                        result[x * self.ny + y][var] = result_arr[x, y, :]

            cubenew.ds[var] = xr.DataArray(
                result_arr,
                dims=["x", "y", "mid_date"],
                coords={"x": self.ds["x"], "y": self.ds["y"], "mid_date": time_variable},
            )
            cubenew.ds[var] = cubenew.ds[var].transpose("mid_date", "y", "x")
            cubenew.ds[var].attrs = {"standard_name": short_name[i], "unit": "m/y", "long_name": long_name[i]}

        if result_quality is not None and "Norm_residual" in result_quality:
            long_name = [
                "Residual from the inversion AX=Y, where Y is the displacement in the direction Est/West [m]",
                "Residual from the regularisation term for the displacement in the direction Est/West [m]",
                "Residual from the inversion AX=Y, where Y is the displacement in the direction North/South [m]",
                "Residual from the regularisation term for the displacement in the direction North/South [m]",
            ]
            short_name = ["ResidualAXY_dx", "ResidualRegu_dx", "ResidualAXY_dy", "ResidualRegu_dy"]
            for k in range(0, 4):
                result_arr = np.array(
                    [
                        result[i * self.ny + j]["NormR"][k]
                        if result[i * self.ny + j]["NormR"].shape[0] != 0
                        else np.nan
                        for i in range(self.nx)
                        for j in range(self.ny)
                    ]
                )
                result_arr = result_arr.reshape((self.nx, self.ny))

                # Spatially smooth (apply a convolutional filter on) the data
                if smooth_res:
                    result_arr = smooth_results(result_arr, window_size=smooth_window_size, filt=smooth_filt)
                    for x in range(self.nx):
                        for y in range(self.ny):
                            result[x * self.ny + y][var] = result_arr[x, y, :]

                cubenew.ds[short_name[k]] = xr.DataArray(
                    result_arr, dims=["x", "y"], coords={"x": self.ds["x"], "y": self.ds["y"]}
                )
                cubenew.ds[short_name[k]] = cubenew.ds[short_name[k]].transpose("y", "x")
                cubenew.ds[short_name[k]].attrs = {
                    "standard_name": short_name[k],
                    "unit": "m",
                    "long_name": long_name[k],
                }

        if result_quality is not None and "Error_propagation" in result_quality:
            long_name = ["Sigma0 Est/West", "Sigma0 North/South [m]", "T value Est/West", "T value North/South"]
            short_name = ["sigma0_x", "sigma0_y", "t_valuex", "t_valuey"]
            for k, var in enumerate(short_name):

                result_arr = np.array(
                    [
                        result[i * self.ny + j]["sigma0"][~np.isnan(result[i * self.ny + j]["sigma0"])].iloc[k]
                        if result[i * self.ny + j]["sigma0"].shape[0] != 0
                        else np.nan
                        for i in range(self.nx)
                        for j in range(self.ny)
                    ]
                )
                result_arr = result_arr.reshape((self.nx, self.ny))

                # Spatially smooth (apply a convolutional filter on) the data
                if smooth_res:
                    result_arr = smooth_results(result_arr, window_size=smooth_window_size, filt=smooth_filt)
                    for x in range(self.nx):
                        for y in range(self.ny):
                            result[x * self.ny + y][var] = result_arr[x, y, :]

                cubenew.ds[short_name[k]] = xr.DataArray(
                    result_arr, dims=["x", "y"], coords={"x": self.ds["x"], "y": self.ds["y"]}
                )
                cubenew.ds[short_name[k]] = cubenew.ds[short_name[k]].transpose("y", "x")
                cubenew.ds[short_name[k]].attrs = {
                    "standard_name": short_name[k],
                    "unit": "m",
                    "long_name": long_name[k],
                }
        del non_null_el, long_name, result_arr

        cubenew.ds["x"] = self.ds["x"]
        cubenew.ds["x"].attrs = {
            "standard_name": "projection_x_coordinate",
            "unit": "m",
            "long_name": "x coordinate of projection",
        }
        cubenew.ds["y"] = self.ds["y"]
        cubenew.ds["y"].attrs = {
            "standard_name": "projection_y_coordinate",
            "unit": "m",
            "long_name": "y coordinate of projection",
        }
        cubenew.ds["mid_date"] = time_variable.to_numpy()
        cubenew.ds["mid_date"].attrs = {
            "standard_name": "central_date",
            "unit": "days",
            "description": "the date in the middle of the two dates between which a velocity is computed",
        }
        cubenew.ds["grid_mapping"] = self.ds.proj4
        cubenew.ds.attrs = {
            "Conventions": "CF-1.10",
            "title": "Ice velocity time series",
            "institution": "Université Grenoble Alpes",
            "references": "Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement observation networks. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-10.\n Charrier, L., Yan, Y., Koeniguer, E. C., Trouve, E., Mouginot, J., & Millan, R. (2022, June). Fusion of multi-temporal and multi-sensor ice velocity observations. In International Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences.",
            "source": source,
            "sensor": sensor,
            "proj4": self.ds.proj4,
            "author": "L. Charrier",
            "history": f"Created on the {date.today()}",
        }
        cubenew.nx = self.nx
        cubenew.ny = self.ny
        cubenew.nz = cubenew.ds["mid_date"].sizes["mid_date"]
        cubenew.filename = filename

        if savepath is not None:  # Save the dataset to a netcdf file
            encoding = {
                "vx": {"zlib": True, "complevel": 5, "dtype": "float32"},
                "vy": {"zlib": True, "complevel": 5, "dtype": "float32"},
            }
            if result_quality is not None:
                if "X_contribution" in result_quality:
                    encoding["xcount_x"] = {"zlib": True, "complevel": 5, "dtype": "int16"}
                    encoding["xcount_y"] = {"zlib": True, "complevel": 5, "dtype": "int16"}
                if "Error_propagation" in result_quality:
                    encoding["error_x"] = {"zlib": True, "complevel": 5, "dtype": "float32"}
                    encoding["error_y"] = {"zlib": True, "complevel": 5, "dtype": "float32"}
            start = time.time()
            cubenew.ds.to_netcdf(f"{savepath}/{filename}.nc", engine="h5netcdf", encoding=encoding)
            if verbose:
                print(f"[Writing results] Saved to {savepath}/{filename}.nc took: {round(time.time() - start, 3)} s")

        if return_result:
            return cubenew, result
        return cubenew

    def write_result_tico(
        self,
        result: list,
        source: str,
        sensor: str,
        filename: str = "Time_series",
        savepath: str | None = None,
        result_quality: list | None = None,
        verbose: bool = False,
    ) -> Union["cube_data_class", str]:

        """
        Write the result from TICOI, stored in result, in a xarray dataset matching the conventions CF-1.10
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.pdf
        units has been changed to unit, since it was producing an error while wirtting the netcdf file

        :param result: [list] --- List of pd xarray, resulut from the TICOI method
        :param source: [str] --- Name of the source
        :param sensor: [str] --- Sensors which have been used
        :param filename: [str] [default is Time_series] --- Filename of file to saved
        :param result_quality: [list | str | None] [default is None] --- Which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)):param savepath: string, path where to save the file
        :param verbose: [bool] [default is None] --- Print information throughout the process (default is False)

        :return cubenew: [cube_data_class] --- New cube where the results are saved
        """

        cubenew = cube_data_class()
        cubenew.ds["x"] = self.ds["x"]
        cubenew.ds["x"].attrs = {
            "standard_name": "projection_x_coordinate",
            "unit": "m",
            "long_name": "x coordinate of projection",
        }
        cubenew.ds["y"] = self.ds["y"]
        cubenew.ds["y"].attrs = {
            "standard_name": "projection_y_coordinate",
            "unit": "m",
            "long_name": "y coordinate of projection",
        }

        long_name = [
            "displacement in the East/West direction [m]",
            "displacement in the North/South direction [m]",
            "number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)) in the East/West direction",
            "number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)) in the North/South direction",
        ]
        short_name = ["x_displacement", "y_displacement", "xcount_x", "xcount_y"]
        unit = ["m", "m", "no unit", "no unit"]

        start = time.time()
        from joblib import Parallel, delayed

        start = time.time()
        second_date_list = [
            np.datetime64(date, "s") for date in sorted({date for df in result for date in df["date2"]})
        ]

        df_list = Parallel(n_jobs=-1)(delayed(reconstruct_common_ref)(df, second_date_list) for df in result)

        print(f"[Writing result] Building cumulative displacement time series took: {round(time.time() - start, 3)} s")
        start = time.time()

        # List of the reference date, i.e. the first date of the cumulative displacement time series
        result_arr = np.array([df_list[i]["Ref_date"][0] for i in range(len(df_list))]).reshape((self.nx, self.ny))
        cubenew.ds["reference_date"] = xr.DataArray(
            result_arr, dims=["x", "y"], coords={"x": self.ds["x"], "y": self.ds["y"]}
        )
        cubenew.ds["reference_date"].attrs = {
            "standard_name": "reference_date",
            "unit": "days",
            "description": "first date of the cumulative displacement time series",
        }

        # name of variable to store
        if result_quality is not None and "X_contribution" in result_quality:
            variables = ["dx", "dy", "xcount_x", "xcount_y"]
        else:
            variables = ["dx", "dy"]

        warnings.filterwarnings(
            "ignore", category=UserWarning
        )  # to avoid the warning  UserWarning: Converting non-nanosecond precision datetime values to nanosecond precision. This behavior can eventually be relaxed in xarray, as it is an artifact from pandas which is now beginning to support non-nanosecond precision values. This warning is caused by passing non-nanosecond np.datetime64 or np.timedelta64 values to the DataArray or Variable constructor; it can be silenced by converting the values to nanosecond precision ahead of time.
        # Store each variable
        big_df = pd.concat(df_list, keys=range(len(df_list)))
        del df_list

        for i, var in enumerate(variables):
            result_arr = big_df[var].values.reshape((self.nx, self.ny, len(second_date_list)))

            cubenew.ds[var] = xr.DataArray(
                result_arr, dims=["x", "y", "second_date"], coords=cubenew.ds.coords
            ).transpose("second_date", "y", "x")

            cubenew.ds[var].attrs = {"standard_name": short_name[i], "unit": unit[i], "long_name": long_name[i]}

        cubenew.ds["grid_mapping"] = self.ds.proj4
        cubenew.ds.attrs = {
            "Conventions": "CF-1.10",
            "title": "Cumulative displacement time series",
            "institution": "Université Grenoble Alpes",
            "references": "Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement observation networks. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-10.\n Charrier, L., Yan, Y., Koeniguer, E. C., Trouve, E., Mouginot, J., & Millan, R. (2022, June). Fusion of multi-temporal and multi-sensor ice velocity observations. In International Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences.",
            "source": source,
            "sensor": sensor,
            "proj4": self.ds.proj4,
            "author": "L. Charrier",
            "history": f"Created on the {date.today()}",
        }
        cubenew.nx = self.nx
        cubenew.ny = self.ny
        cubenew.nz = cubenew.ds.dims["second_date"]
        cubenew.filename = filename

        if savepath is not None:  # save the dataset to a netcdf file
            encoding = {
                "dx": {"zlib": True, "complevel": 5, "dtype": "float32"},
                "dy": {"zlib": True, "complevel": 5, "dtype": "float32"},
            }
            if result_quality is not None and "X_contribution" in result_quality:
                encoding["xcount_x"] = {"zlib": True, "complevel": 5, "dtype": "int16"}
                encoding["xcount_y"] = {"zlib": True, "complevel": 5, "dtype": "int16"}
            start = time.time()
            cubenew.ds.to_netcdf(f"{savepath}/{filename}.nc", engine="h5netcdf", encoding=encoding)
            if verbose:
                print(f"[Writing results] Saved to {savepath}/{filename}.nc took: {round(time.time() - start, 3)} s")

        return cubenew

    def write_results_ticoi_or_tico(
        self,
        result: list,
        source: str,
        sensor: str,
        filename: str = "Time_series",
        savepath: str | None = None,
        result_quality: list | None = None,
        verbose: bool = False,
    ) -> Union["cube_data_class", str]:

        """
        Write the result from TICOI or TICO, stored in result, in a xarray dataset matching the conventions CF-1.10
        It recognizes whether the results are irregular or regular and uses the appropriate saving method
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.pdf
        units has been changed to unit, since it was producing an error while wirtting the netcdf file

        :param result: [list] --- List of pd xarray, resulut from the TICOI method
        :param source: [str] --- Name of the source
        :param sensor: [str] --- Sensors which have been used
        :param filename: [str] [default is Time_series] --- Filename of file to saved
        :param result_quality: [list | str | None] [default is None] --- Which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)):param savepath: string, path where to save the file
        :param verbose: [bool] [default is None] --- Print information throughout the process (default is False)

        :return cubenew: [cube_data_class] --- New cube where the results are saved
        """

        if result[0].columns[0] == "date1":
            self.write_result_ticoi(
                result=result,
                source=source,
                sensor=sensor,
                filename=filename,
                savepath=savepath,
                result_quality=result_quality,
                verbose=verbose,
            )
        else:
            self.write_result_tico(
                result=result,
                source=source,
                sensor=sensor,
                filename=filename,
                savepath=savepath,
                result_quality=result_quality,
                verbose=verbose,
            )

class cube_3d_class:
    def __init__(self):
        self.at = cube_data_class(sar=True)
        self.dt = cube_data_class(sar=True)
        self.nz = self.at.nz + self.dt.nz
        self.nx = self.at.nx
        self.ny = self.at.ny
    
        self.filedir = [self.at.filedir, self.dt.filedir]
        self.filename = [self.at.filename, self.dt.filename]
        self.author = self.at.author
        self.source = self.at.source

        self.resolution = self.at.resolution
        self.is_TICO = self.at.is_TICO
    
    def load(self,
        filepath_at: list | str,
        filepath_dt: list | str,
        load_kwargs,
        sar_info_at: str | None = None,
        sar_info_dt: str | None = None,
    ):  
        
        self.at.load(filepath_at, **load_kwargs, sar_info=sar_info_at)
        self.dt.load(filepath_dt, **load_kwargs, sar_info=sar_info_dt)
        
        # rechunking to ensure the same chunksize at x and y dimension for both cubes
        chunks_at = self.at.ds.chunksizes
        chunks_dt = self.dt.ds.chunksizes
        if chunks_at['x'] != chunks_dt['x'] or chunks_at['y'] != chunks_dt['y']:
            new_chunks = {dim: max(chunks_at[dim], chunks_dt[dim]) for dim in ['x', 'y']}
            self.at.ds = self.at.ds.chunk(new_chunks)
            self.dt.ds = self.dt.ds.chunk(new_chunks)
        # self.cube_dt = self.cube_dt.align_cube(self.cube_at)
        
    def filter_cube(self, preData_kwargs):
        
        if isinstance(preData_kwargs['regu'], str):
            if preData_kwargs['regu'] == '1accelnotnull' or '1accelnotnull' in preData_kwargs['regu'].values():
                raise ValueError("The regularization '1accelnotnull' is not available right now for 3D inversion")
        obs_filt_at, regu = self.at.filter_cube(**preData_kwargs)
        obs_filt_dt, regu = self.dt.filter_cube(**preData_kwargs)
        obs_filt = [obs_filt_at, obs_filt_dt]
        
        return obs_filt, regu
    
    def load_pixel(
        self,
        i: int | float,
        j: int | float,
        unit: int = 365,
        regu: int | str = 1,
        coef: int = 100,
        flag: xr.Dataset | None = None,
        solver: str = "LSMR",
        interp: str = "nearest",
        proj: str = "EPSG:4326",
        rolling_mean: List[xr.Dataset] | None = None,
        visual: bool = False,
        output_format="np",
    ):

        data_at = self.at.load_pixel(i, j, unit=unit, regu=regu, coef=coef, flag=flag, solver=solver, interp=interp, proj=proj, rolling_mean=rolling_mean[0], visual=visual, output_format=output_format)
        data_dt = self.dt.load_pixel(i, j, unit=unit, regu=regu, coef=coef, flag=flag, solver=solver, interp=interp, proj=proj, rolling_mean=rolling_mean[1], visual=visual, output_format=output_format)
        
        
        data_at[0].append(np.full((data_at[0][0].shape[0], 1), 'S1_at'))
        data_dt[0].append(np.full((data_dt[0][0].shape[0], 1), 'S1_dt'))
        
        data_values = []
        for i in range(len(data_at[0])):
            data_values.append(np.concatenate((data_at[0][i], data_dt[0][i]), axis=0))
        
        mean = [data_at[1], data_dt[1]]  # mean[0] for ascending, mean[1] for descending
        dates_range = [data_at[2], data_dt[2]] # dates_range[0] for ascending, dates_range[1] for descending
        
        if len(data_at) > 3:
            flag, regu = data_at[3], data_dt[4]
            return data_values, mean, dates_range, flag, regu
        else:
            return data_values, mean, dates_range
