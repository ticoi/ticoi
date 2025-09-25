"""
Class object to store and manipulate velocity observation data

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import itertools
import os
import time
import warnings
from functools import reduce
from typing import Optional

import dask.array as da
import geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.enums
import rasterio.warp
import xarray as xr
from dask.array.lib.stride_tricks import sliding_window_view
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from pyproj import CRS, Proj, Transformer
from rasterio.features import rasterize
from tqdm import tqdm

from ticoi.filtering_functions import dask_filt_warpper, dask_smooth_wrapper
from ticoi.inversion_functions import construction_dates_range_np
from ticoi.mjd2date import mjd2date

# %% ======================================================================== #
#                              CUBE DATA CLASS                                #
# =========================================================================%% #


class CubeDataClass:
    _loader_registry = {
        "ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)": "_loader_itslive",
        "J. Mouginot, R.Millan, A.Derkacheva": "_loader_millan",
        "L. Charrier, L. Guo": "_loader_charrier",
        "L. Charrier": "_loader_charrier",
        "E. Ducasse": "_loader_ducasse",
        "IGE": "_loader_charrier",
    }

    def __init__(self, cube=None, ds=None):
        """
        Initialisation of the main attributes, or copy cube's attributes and ds dataset if given.

        :param cube: [cube_data_class] --- Cube to copy
        :param ds: [xr dataset | None] --- New dataset. If None, copy cube's dataset
        """

        if not isinstance(cube, CubeDataClass):
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

        else:  # load the cube information
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

        # choose standardizer based on author

    def find_time_dimensions(self):
        """
        Find the name of the time dimension
        """
        coords_not_spatial = [n for n in self.ds.coords if n not in {"x", "y"}]
        return coords_not_spatial[0]

    def update_dimension(self, time_dim: str = "mid_date"):
        """
        Update the attributes corresponding to cube dimensions: nx, ny, and nz

        :param time_dim: [str] [default is 'mid_date'] --- Name of the z dimension within the original dataset self.ds
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
            del i1, i2, j1, j2, i3, i4, j3, j4
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
        This function is from gtsa DOI 10.5281/zenodo.8188085.

        :param variable_name: [str] [default is 'vx'] --- Name of the variable containing the time series array
        :param x_dim: [str] [default is 'x'] --- Name of the x dimension in the array
        :param y_dim: [str] [default is 'y'] --- Name of the y dimension in the array
        :param time_dim: [str] [default is 'mid_date'] --- Name of the z dimension within the original dataset self.ds
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

    def _standardize_sensor_names(self, sensor_input: np.ndarray | str) -> np.ndarray:
        """
        Standardize sensor names
        :param sensor_input:  [str or np.ndarray] --- Sensor input, which can of the form L8, S2 for example
        :return: Sensor output array
        """
        SENSOR_MAP = {
            "L7": "Landsat-7",
            "L8": "Landsat-8",
            "L9": "Landsat-9",
            "landsat-8": "Landsat-8",
            "L8. ": "Landsat-8",
            "LS8": "Landsat-8",
            "LS7": "Landsat-7",
            "LS9": "Landsat-9",
            "S1": "Sentinel-1",
            "S1A": "Sentinel-1",
            "S1B": "Sentinel-1",
            "S2": "Sentinel-2",
            "S2A": "Sentinel-2",
            "S2B": "Sentinel-2",
            # other
            "Pleiades": "Pleiades",
            # add more mappings as needed
        }

        if isinstance(sensor_input, str):
            cleaned_name = sensor_input.strip()
            return SENSOR_MAP.get(cleaned_name, cleaned_name)

        elif isinstance(sensor_input, (np.ndarray, list, tuple)):
            cleaned_array = np.char.strip(np.asarray(sensor_input).astype(str))

            vectorized_map = np.vectorize(lambda s: SENSOR_MAP.get(s, s))
            return vectorized_map(cleaned_array)

        return sensor_input

    def _normalize_error_to_confidence(self, error_da: xr.DataArray) -> xr.DataArray:
        """
        normalize error to confidence between [0, 1]
        :param error_da:  [np.ndarray] --- array of errors
        :return:
        """
        min_val = error_da.min()
        max_val = error_da.max()
        if max_val == min_val:
            return xr.ones_like(error_da)
        return 1 - (error_da - min_val) / (max_val - min_val)

    def _apply_data_subset_in_space(self, proj: str, subset: list, buffer: list):
        """
        Spatial subset, using a buffer and or subset
        :param proj: [str] --- EPSG system of the coordinates given in subset
        :param subset: [list] --- A list of 4 float, these values are used to give a subset of the dataset : [xmin, xmax, ymax, ymin]
        :param buffer:  [list] --- A list of 3 float, the first two are the longitude and the latitude of the central point, the last is the buffer size
        """
        # spatial subset
        if subset is not None:
            self.subset(proj, subset)
        elif buffer is not None:
            self.buffer(proj, buffer)

        # update dimensions after spatial filtering
        self.update_dimension()

    def _apply_data_selection(self, pick_date: list | None, pick_sensor: list | None, pick_temp_bas: list | None):
        """
        selection of dates, sensors, temporal baselines
        :param pick_date: [list] --- list of date to select
        :param pick_sensor: [list] --- list of sensors to select
        :param pick_temp_bas: [list] --- list of temporal baselines to select
        """

        # 1. update time dimension name and format
        if pick_date is not None:
            mask = (self.ds["date1"] >= np.datetime64(pick_date[0])) & (self.ds["date2"] <= np.datetime64(pick_date[1]))
            self.ds = self.ds.where(mask.compute(), drop=True)

        # sensor selection
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds["sensor"].isin(pick_sensor))

        # 4. temporal baseline selection (based on standardized date1, date2)
        if pick_temp_bas is not None:
            temp_baseline = (self.ds["date2"] - self.ds["date1"]) / np.timedelta64(1, "D")
            mask = (temp_baseline >= pick_temp_bas[0]) & (temp_baseline <= pick_temp_bas[1])
            self.ds = self.ds.where(mask, drop=True)

        # final dimension update
        self.update_dimension()

    def _add_standardized_variable(self, standard_data: dict):
        """
        Add standardized variable to the cube dataarry self.ds
        :param standard_data: [dict] --- name and values of standardized variables
        """
        for var_name, data in standard_data.items():
            if isinstance(data, (str, float)):  # if sensor is a string or error is a float
                data = np.repeat(
                    data, self.ds.sizes["mid_date"]
                )  # create a np array of lenght self.ds.sizes[time_dim], with the string

            if data.ndim == 1:  # for sensor, source, date1, and date2
                dims = ("mid_date",)
            elif data.ndim == 3:  # for vx, vy
                dims = ("mid_date", "y", "x")
            else:  # if error is already 2D
                dims = data.dims

            self.ds[var_name] = xr.DataArray(data, dims=dims)

        self.ds = self.ds.unify_chunks()

    @classmethod
    def register_loader(cls, author: str, func: callable):
        """
        Allow users to register their own loader function.
        :param author: [str] --- name of the author
        :param func: [callable] --- function to be registered
        :return:
        """
        cls._loader_registry[author] = func

    def _loader_generic(self, conf: bool) -> dict:
        """
                standardize dataset with unrecognized author based on variable names.

        it assumes the dataset is already in a standard format, which means
        it contains the required variable names with correct meanings.
        variables:
                    [vx, vy, date1, date2]: necessary,
                    [sensor, source, errorx, errory]: optional
        if you want to specify the name of the author and the source of the dataset, it should be inside an attribute called author and source respectively.
        :param conf: [bool] --- if the errors need to be converted as confidence
        :return:
        """
        # required variables
        REQUIRED_VARS = ["vx", "vy", "date1", "date2"]

        # check if all required variables are present
        missing_vars = [var for var in REQUIRED_VARS if var not in self.ds.variables]

        if missing_vars:
            raise ValueError(
                f"Data loading failed due to missing required variables: {missing_vars}. "
                f"The dataset must at least contain the following variables: {REQUIRED_VARS}."
            )

        # provide default values for optional variables if they are missing
        self.author = self.ds.attrs.get(
            "author", "Unknown"
        )  # if the attribute auhtor does not exist, put the attribute to Unknown
        self.source = self.ds.attrs.get("source", "Unknown")

        # standardize sensor names if sensor variable exists
        if "sensor" in self.ds:
            sensor = self.ds["sensor"].values
        elif "sensor" in self.ds.attrs:
            sensor = self.ds.attrs["sensor"]
        else:
            sensor = "Unknown"

        errorx = self.ds.get("errorx", 1.0)
        errory = self.ds.get("errory", 1.0)
        # if data has errorx/errory and need normalization
        if conf and "errorx" in self.ds:
            errorx = self._normalize_error_to_confidence(errorx)
            errory = self._normalize_error_to_confidence(errory)

        if (self.ds.vx == 0).any().values:  # mask values equal to 0
            mask = (self.ds.vx != 0) & (self.ds.vy != 0)
            self.ds[["vx", "vy"]] = self.ds[["vx", "vy"]].where(mask)

        standard_data = {
            "date1": self.ds["date1"].astype("datetime64[ns]"),
            "date2": self.ds["date2"].astype("datetime64[ns]"),
            "sensor": sensor,
            "source": self.source,
            "errorx": errorx,
            "errory": errory,
        }

        return standard_data

    def _loader_itslive(self, conf: bool) -> dict:
        """
        load ITS_LIVE dataset
        :param conf: [bool] --- if the errors need to be converted as confidence
        :return:
        """
        self.author = self.ds.author.split(", a NASA")[0]
        self.source = self.ds.url
        self.ds.attrs["proj4"] = self.ds["mapping"].proj4text

        # standardize sensor names
        sensor_raw = np.core.defchararray.add(
            np.char.strip(self.ds["mission_img1"].values.astype(str), " "),
            np.char.strip(self.ds["satellite_img1"].values.astype(str), " "),
        ).astype("U10")

        # normalize error if needed
        errorx = self.ds["vx_error"]
        errory = self.ds["vy_error"]
        if conf:
            errorx = self._normalize_error_to_confidence(errorx)
            errory = self._normalize_error_to_confidence(errory)

        return {
            "date1": self.ds["acquisition_date_img1"].astype("datetime64[ns]"),
            "date2": self.ds["acquisition_date_img2"].astype("datetime64[ns]"),
            "sensor": self._standardize_sensor_names(sensor_raw),
            "source": "ITS_LIVE",
            "errorx": errorx,
            "errory": errory,
        }

    def _loader_millan(self, conf: bool) -> dict:
        """
        load Millan dataset
        :param conf: [bool] --- if the errors need to be converted as confidence
        :return:
        """
        self.author = "IGE"
        self.source = self.ds.source
        self.ds = self.ds.rename({"z": "mid_date"})

        # standardize sensor names
        sensor_raw = np.char.strip(self.ds["sensor"].values.astype(str), " ")

        # normalize error if needed
        errorx_1d = self.ds["error_vx"]
        errory_1d = self.ds["error_vy"]
        if conf:
            errorx_1d = self._normalize_error_to_confidence(errorx_1d)
            errory_1d = self._normalize_error_to_confidence(errory_1d)
        ny, nx = self.ds.dims["y"], self.ds.dims["x"]
        errorx = np.tile(errorx_1d.values[:, np.newaxis, np.newaxis], (1, ny, nx))
        errory = np.tile(errory_1d.values[:, np.newaxis, np.newaxis], (1, ny, nx))
        # standardize date format
        date1 = xr.DataArray([mjd2date(d) for d in self.ds["date1"].values], dims="mid_date").astype("datetime64[ns]")
        date2 = xr.DataArray([mjd2date(d) for d in self.ds["date2"].values], dims="mid_date").astype("datetime64[ns]")
        self.ds = self.ds.assign_coords(mid_date=date1 + (date2 - date1) / 2)

        return {
            "date1": date1,
            "date2": date2,
            "sensor": self._standardize_sensor_names(sensor_raw),
            "source": "IGE",
            "errorx": errorx,
            "errory": errory,
        }

    def _loader_ducasse(self, conf: bool) -> dict:
        """
        load Ducasse dataset
        :param conf: [bool] --- if the errors need to be converted as confidence
        :return:
        """
        self.author = "IGE"
        self.source = "IGE"
        self.ds = self.ds.rename({"time": "mid_date"})
        self.ds = self.ds.transpose("mid_date", "y", "x")  # transpose coordinates

        # standardize date format
        dates = self.ds["mid_date"].values
        date1 = xr.DataArray([d.split(" ")[0] for d in dates], dims="mid_date").astype("datetime64[ns]")
        date2 = xr.DataArray([d.split(" ")[1] for d in dates], dims="mid_date").astype("datetime64[ns]")
        self.ds = self.ds.assign_coords(mid_date=date1 + (date2 - date1) / 2)

        return {
            "vx": self.ds["vx"],
            "vy": -self.ds["vy"],  # special case for ducasse
            "date1": date1,
            "date2": date2,
            "sensor": "Pleiades",
            "source": "IGE",
            "errorx": 1.0,
            "errory": 1.0,
        }

    def _loader_charrier(self, conf: bool) -> dict:
        """
        load Charrier dataset
        :param conf:
        :return:
        """
        self.author = "IGE" if "Mouginot" in self.ds.author else self.ds.author
        self.source = self.ds.source

        # normalize error if needed
        errorx, errory = self.ds.get("errorx"), self.ds.get("errory")
        if conf and errorx is not None:
            errorx = self._normalize_error_to_confidence(errorx)
            errory = self._normalize_error_to_confidence(errory)
        sensor = self.ds.attrs["sensor"] if "sensor" in self.ds.attrs else self.ds["sensor"]
        source = self.ds.attrs["source"] if "source" in self.ds.attrs else self.ds["source"]
        return {
            "sensor": self._standardize_sensor_names(sensor),
            "source": source,
            "errorx": errorx if errorx is not None else 1.0,
            "errory": errory if errory is not None else 1.0,
        }

    def load(
        self,
        filepath: list | str,
        chunks: dict | str | int = "auto",
        conf: bool = False,
        subset: list | None = None,
        buffer: list | None = None,
        pick_date: list | None = None,
        pick_sensor: list | None = None,
        pick_temp_bas: list | None = None,
        proj: str = "EPSG:4326",
        mask: str | xr.DataArray = None,
        reproj_coord: bool = False,
        reproj_vel: bool = False,
        verbose: bool = False,
    ):
        self.__init__()

        if isinstance(filepath, list):
            if verbose:
                print(f"[Data loading] Loading cube 1 from file: {filepath}")
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
                if verbose:
                    print(f"[Data loading] Loading and merging cube {n + 1} from file: {filepath[n]}")
                cube2 = CubeDataClass()
                cube2.load(
                    filepath[n],
                    chunks=chunks,
                    conf=conf,
                    subset=None,
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
                else:
                    cube2.ds = cube2.ds.sel(x=self.ds.x, y=self.ds.y, method="nearest")
                    if cube2.nx == self.nx and cube2.ny == self.ny:
                        cube2.ds = cube2.ds.assign_coords(x=self.ds.x, y=self.ds.y)
                self.merge_cube(cube2)  # Merge the new cube to the main one
                del cube2

            if chunks == {}:  # Rechunk with optimal chunk size
                var_name = "vx" if not self.is_TICO else "dx"
                time_dim = "mid_date" if not self.is_TICO else "second_date"
                tc, yc, xc = self.determine_optimal_chunk_size(
                    variable_name=var_name, x_dim="x", y_dim="y", time_dim=time_dim, verbose=verbose
                )
                self.ds = self.ds.chunk({time_dim: tc, "x": xc, "y": yc})
                self.update_dimension()
            return

        if verbose:
            print(f"[Data loading] Opening file: {filepath}")
        ext = filepath.split(".")[-1]
        if ext == "nc":
            try:
                self.ds = xr.open_dataset(filepath, chunks=chunks)
            except NotImplementedError:
                if verbose:
                    print(
                        "[Data loading] Warning: Auto-chunking failed (possibly due to object dtype). "
                        "Falling back to no chunks during initial load."
                    )
                self.ds = xr.open_dataset(filepath, chunks={})
        elif ext == "zarr":
            self.ds = xr.open_dataset(filepath, decode_timedelta=False, engine="zarr", consolidated=True, chunks=chunks)
        else:
            raise ValueError(f"File extension {ext} not recognized, only .nc and .zarr are supported.")

        if "Author" in self.ds.attrs:
            self.ds.attrs["author"] = self.ds.attrs.pop("Author")

        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)

        # get standardized data
        author = self.ds.attrs.get("author", "Unknown")
        loader = self._loader_registry.get(author, self._loader_generic)
        if isinstance(loader, str):
            loader = getattr(self, loader)
        if verbose:
            print(
                f"[Data loading] Warning: Unrecognized author '{author}'. Attempting to load based on defined variable names."
            )
        standard_data = loader(conf)

        # keep only certain variable and attributes
        variables_to_keep = ["vx", "vy", "mid_date", "x", "y", "date1", "date2"]
        self.ds = self.ds.drop_vars([var for var in self.ds.variables if var not in variables_to_keep])
        attributes_to_keep = ["author", "source", "date_created", "proj4", "mapping"]
        self.ds.attrs = {k: v for k, v in self.ds.attrs.items() if k in attributes_to_keep}

        # construct cubedataclass from standardized data
        self._add_standardized_variable(standard_data)

        # apply subsetting and filtering
        if verbose:
            print("[Data loading] Applying data selection...")
        if subset is not None or buffer is not None:
            self._apply_data_subset_in_space(proj, subset, buffer)
        if pick_date is not None or pick_sensor is not None or pick_temp_bas is not None:
            self._apply_data_selection(pick_date, pick_sensor, pick_temp_bas)
        elif subset is None and buffer is None:
            self.update_dimension()

        if mask:
            if verbose:
                print("[Data loading] Masking cube...")
            self.mask_cube(mask)

        if verbose:
            print("[Data loading] Computing optimal chunk size...")
        tc, yc, xc = self.determine_optimal_chunk_size(verbose=verbose)
        self.ds = self.ds.chunk({"mid_date": tc, "x": xc, "y": yc})

        self.ds = self.ds.sortby("mid_date")
        self.standardize_cube_for_processing()

        if self.ds.rio.crs is None and "proj4" in self.ds.attrs:
            self.ds.rio.write_crs(self.ds.attrs["proj4"])

        if verbose:
            print("[Data loading] Cube loaded successfully.")

    def standardize_cube_for_processing(self, time_dim="mid_date"):
        """
        Prepare the xarray dataset for the processing: add a variable temporal_baseline, errors if they do not exist

        :param time_dim_name: [str] [default is 'mid_date'] --- Name of the z dimension within the original dataset self.ds
        """

        self.ds = self.ds.unify_chunks()
        if self.ds.chunksizes[time_dim] != (self.nz,):  # no chunk in time
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

    def prepare_interpolation_date(
        self,
    ) -> (np.datetime64, np.datetime64):  # type: ignore
        """
        Define the first and last date required for the interpolation, as the first date and last in the observations.
        The purpose is to have homogenized results

        :param cube: dataset

        :return: first and last date required for the interpolation
        """

        # Prepare interpolation dates
        cube_date1 = self.date1_().tolist()
        cube_date1 = cube_date1 + self.date2_().tolist()
        cube_date1.remove(np.min(cube_date1))
        first_date_interpol = np.min(cube_date1)
        last_date_interpol = np.max(self.date2_())

        return first_date_interpol, last_date_interpol

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

    def date1_(self) -> np.array:
        """
        Accessor to the first dates of acquisition.

        :return: [np array] --- np array of date1
        """

        return np.asarray(self.ds["date1"]).astype("datetime64[D]")

    def date2_(self) -> np.array:
        """
        Accessor to the second dates of acquisition.

        :return: [np array] --- np array of date2
        """

        return np.asarray(self.ds["date2"]).astype("datetime64[D]")

    def datec_(self) -> np.array:
        """
        Accessor to the central dates of the data.

        :return: [np array] --- np array of central date
        """

        return (self.date1_() + self.temp_base_(return_list=False, format_date="D") // 2).astype("datetime64[D]")

    def vv_(self) -> np.array:
        """
        Accessor to the magnitude of the velocities.

        :return: [np array] --- np array of velocity magnitude
        """

        return np.sqrt(self.ds["vx"] ** 2 + self.ds["vy"] ** 2)

    def EPSG_code_(self) -> int:
        """
        Accessor to the EPSG code of the dataset.
        """

        return self.ds.rio.crs.to_epsg()

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
        regu: int | str = "1accelnotnull",
        coef: int = 100,
        flag: xr.Dataset | None = None,
        solver: str = "LSMR_ini",
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
        :param regu: [int | str] [default is '1accelnotnull'] --- Type of regularization
        :param coef: [int] [default is 100] --- Coef of Tikhonov regularisation
        :param flag: [xr dataset | None] [default is None] --- If not None, the values of the coefficient used for stable areas, surge glacier and non surge glacier
        :param solver: [str] [default is 'LSMR_ini'] --- Solver of the inversion: 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
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
            i, j = self.convert_coordinates(i, j, proj=proj)  # convert the coordinates to the projection of the cube
            # Interpolate only necessary variables and drop NaN values
            if interp == "nearest":
                data = self.ds.sel(x=i, y=j, method="nearest")[var_to_keep]
                data = data.dropna(dim="mid_date")
            else:
                data = self.ds.interp(x=i, y=j, method=interp)[var_to_keep].dropna(dim="mid_date")

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
                dates_range = construction_dates_range_np(data_dates)
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
                        "Filtering method should be either 'median_angle', 'vvc_angle', 'topo_angle', 'z_score','mz_score', 'magnitude', 'median_magnitude' or 'error'."
                    )
        else:
            raise ValueError("delete_outliers must be a int, a string or a dict, not {type(delete_outliers)}")

    def mask_cube(self, mask: xr.DataArray | str):
        """
        Mask some of the data of the cube (putting it to np.nan).

        :param mask: [str | xr dataarray] --- Either a DataArray with 1 the data to keep and 0 the ones to remove, or a path to a file containing a DataArray or a shapefile to be rasterized
        """

        if type(mask) is str:
            if (
                mask[-3:] == "shp" or mask[-4:] == "gpkg"
            ):  # Convert the shp file or geopackage to an xarray dataset (rasterize the shapefile)
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
        Compute the average flow direction from the input vx and vy files or just from the observations
        :param: vx_file | vy_file: [str] --- path of the flow velocity file, should be geotiff format
        :return: direction: [xr.DataArray] --- computed average flow direction at each pixel
        """
        if vx_file is not None and vy_file is not None:
            vx = self.reproject_geotiff_to_cube(vx_file)
            vy = self.reproject_geotiff_to_cube(vy_file)
        else:
            vx = self.ds["vx"].values
            vy = self.ds["vy"].values

        temporal_baseline = self.ds["temporal_baseline"].values
        temporal_baseline = temporal_baseline[np.newaxis, np.newaxis, :]
        vx_weighted = np.nansum(vx * temporal_baseline, axis=2) / np.nansum(temporal_baseline, axis=2)
        vy_weighted = np.nansum(vy * temporal_baseline, axis=2) / np.nansum(temporal_baseline, axis=2)

        v_mean_weighted = np.sqrt(vx_weighted**2 + vy_weighted**2)

        direction = np.arctan2(vx_weighted, vy_weighted)
        direction = (np.rad2deg(direction) + 360) % 360

        direction = np.where(v_mean_weighted < 1, np.nan, direction)

        direction = xr.Dataset(
            data_vars=dict(
                direction=(["y", "x"], np.array(direction.T)),
            ),
            coords=dict(x=(["x"], self.ds.x.data), y=(["y"], self.ds.y.data)),
        )

        return direction

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

    def filter_cube_before_inversion(
        self,
        i: int | float | None = None,
        j: int | float | None = None,
        smooth_method: str = "savgol",
        s_win: int = 3,
        t_win: int = 90,
        sigma: int = 3,
        order: int = 3,
        unit: int = 365,
        delete_outliers: str | float | None = None,
        flag: xr.Dataset | str | None = None,
        dem_file: str | None = None,
        regu: int | str = "1accelnotnull",
        solver: str = "LSMR_ini",
        proj: str = "EPSG:4326",
        velo_or_disp: str = "velo",
        select_baseline: int | None = 180,
        verbose: bool = False,
    ) -> xr.Dataset:
        """
           Filter the original data before the inversion:
        -delete outliers according to the provided criterion
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
        :param regu: [int | str] [default is "1accelnotnull"] --- Regularisation of the solver
        :param solver: [str] [default is 'LSMR_ini'] --- Solver used to invert the system
        :param proj: [str] [default is 'EPSG:4326'] --- EPSG of i,j projection
        :param velo_or_disp: [str] [default is 'velo'] --- 'disp' or 'velo' to indicate the type of the observations : 'disp' mean that self contain displacements values and 'velo' mean it contains velocity
        :param select_baseline: [int | None] [default is None] --- threshold of the temporal baseline to select, if the number of observation is lower than 3 times the number of estimated displacement with this threshold, it is increased by 30 days
        :param verbose: [bool] [default is False] --- Print information throughout the process

        :return obs_filt: [xr dataset | None] --- Filtered dataset
        """

        def loop_rolling(da_arr: xr.Dataset, select_baseline: int | None = 180) -> (np.ndarray, np.ndarray):  # type: ignore
            """
            A function to calculate spatial mean, resample data, and calculate smoothed velocity.

            :param da_arr: [xr dataset] --- Original data
            :param select_baseline: [int] [default is None] --- Threshold over the temporal baselines

            :return spatial_mean: [np array] --- smoothed velocity
            :return date_out: [np array] --- Observed dates
            """

            # Compute the dates of the estimated displacements time series
            date_out = date_range[:-1] + np.diff(date_range) // 2
            mid_dates = self.ds["mid_date"]

            if verbose:
                start = time.time()
            if select_baseline is not None:  # select data with a temporal baseline lower than a threshold
                baseline = self.ds["temporal_baseline"].compute()
                idx = np.where(baseline < select_baseline)
                while (
                    len(idx[0]) < 3 * len(date_out) & (select_baseline < 200)
                ):  # Increase the threshold by 30, if the number of observation is lower than 3 times the number of estimated displacement
                    select_baseline += 30
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
                spatial_mean = da.nanmean(
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

        # The spatio-temporal smoothing should be carried on velocity, while we need displacement during inversion
        if velo_or_disp == "disp":  # to provide velocity values
            self.ds["vx"] = self.ds["vx"] / self.ds["temporal_baseline"] * unit
            self.ds["vy"] = self.ds["vy"] / self.ds["temporal_baseline"] * unit

        if flag is not None:  # create a flag, to identify stable,areas, and eventually surges
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

        if delete_outliers is not None:  # remove outliers beforehand
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

        if "1accelnotnull" in regu or "directionxy" in regu:  # compute velocity smoothed using a spatio-temporal filter
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
            )  # dates between which the displacement should be estimated
            if verbose:
                start = time.time()

            # spatio-temporal filter
            vx_filtered, dates_uniq = loop_rolling(
                self.ds["vx"], select_baseline=select_baseline
            )  # dates_uniq correspond to the central date of dates_range
            vy_filtered, dates_uniq = loop_rolling(self.ds["vy"], select_baseline=select_baseline)

            # We obtain one smoothed value for each unique date in date_range
            obs_filt = xr.Dataset(
                data_vars=dict(
                    vx_filt=(["mid_date", "y", "x"], vx_filtered), vy_filt=(["mid_date", "y", "x"], vy_filtered)
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

        if obs_filt is not None:
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

        cubes = []
        for s in range(n_split):
            if isinstance(dim, str):
                cube = CubeDataClass(
                    self,
                    self.ds.isel(
                        {
                            dim: slice(
                                s * len(self.ds[dim].values) // n_split,
                                (s + 1) * len(self.ds[dim].values) // n_split,
                                1,
                            )
                        }
                    ),
                )
                cube.update_dimension()
                if savepath is not None:
                    cube.ds.to_netcdf(f"{savepath}{dim}_{s}.nc")
                    print(f"Split cube saved at {savepath}{dim}_{s}.nc")
                cubes.append(cube)
            elif isinstance(dim, list):
                cube = CubeDataClass(
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
                    cubes.append(cube)

        return cubes

    def reproj_coord(
        self,
        new_proj: Optional[str] = None,
        new_res: Optional[float] = None,
        interp_method: str = "nearest",
        cube_to_match: Optional["CubeDataClass"] = None,
    ):
        """
        Repreject the cube_data_self to a given projection system, and (optionally) resample this cube to a given resolution.
        The new projection can be defined by the variable new_proj or by a cube stored in cube_to_match.
        The new resolution can be defined by the variable new_res or by a cube stored in cube_to_match.

        :param new_proj: [str]  --- EPSG code of the new projection
        :param new_res: [float]  --- new resolution in the unit of the new projection system
        :param interp_method: [str]  ---
        :param cube_to_match:  [cube_data_class]  --- cube used as a reference to reproject self
        """
        # assign coordinate system
        if cube_to_match is not None:
            cube_to_match.ds = cube_to_match.ds.rio.write_crs(cube_to_match.ds.proj4)
        self.ds = self.ds.rio.write_crs(self.ds.proj4)
        if tuple(self.ds.dims) != ("mid_date", "y", "x"):
            self.ds = self.ds.transpose("mid_date", "y", "x")

        # Reproject coordinates
        if cube_to_match is not None:
            if interp_method == "nearest":
                self.ds = self.ds.rio.reproject_match(cube_to_match.ds, resampling=rasterio.enums.Resampling.nearest)
            elif interp_method == "bilinear":
                self.ds = self.ds.rio.reproject_match(cube_to_match.ds, resampling=rasterio.enums.Resampling.bilinear)
            if new_res is not None or new_proj is not None:
                print("The new projection has been defined according to cube_to_match.")
        elif new_res is None:
            self.ds = self.ds.rio.reproject(new_proj)
        else:
            self.ds = self.ds.rio.reproject(new_proj, resolution=new_res)

        # Reject abnormal data (when the cube sizes are not the same and data are missing, the interpolation leads to infinite or nearly-infinite values)
        self.ds[["vx", "vy"]] = self.ds[["vx", "vy"]].where(
            (np.abs(self.ds["vx"].values) < 10000) | (np.abs(self.ds["vy"].values) < 10000), np.nan
        )

        # Update of cube_data_classxr attributes
        warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")  # prevent to have a warning
        if new_proj is None:
            new_proj = cube_to_match.ds.proj4
            self.ds = self.ds.assign_attrs({"proj4": new_proj})
        else:
            self.ds = self.ds.assign_attrs({"proj4": CRS.from_epsg(new_proj[5:]).to_proj4()})
        self.ds = self.ds.assign_coords({"x": self.ds.x, "y": self.ds.y})
        self.update_dimension()

    def reproj_vel(
        self,
        new_proj: Optional[str] = None,
        cube_to_match: Optional["CubeDataClass"] = None,
        unit: int = 365,
        nb_cpu: int = 8,
    ):
        """
        Reproject the velocity vector in a new projection grid (i.e. the x and y variables are not changed, only vx and vy are modified).
        The new projection can be defined by the variable new_proj or by a cube stored in cube_to_match.

        :param new_proj: [str]  --- EPSG code of the new projection
        :param cube_to_match: [cube_data_class]  --- cube used as a reference to reproject self
        :param unit: [int] [default is 365] --- 365 if the unit of the velocity are m/y, 1 if they are m/d
        :param nb_cpu: [int] [default is 8] --- number of CPUs used for the parallelization
        """

        if new_proj is None:
            if cube_to_match is not None:
                new_proj = cube_to_match.ds.proj4
                transformer = Transformer.from_crs(self.ds.proj4, new_proj)
            else:
                raise ValueError("Please provide new_proj or cube_to_match")
        else:
            transformer = Transformer.from_crs(self.ds.proj4, CRS.from_epsg(new_proj[5:]).to_proj4())

        # Prepare grid and transformer
        grid = np.meshgrid(self.ds["x"], self.ds["y"])
        grid_transformed = transformer.transform(grid[0], grid[1])
        # temp = self.temp_base_()
        temp = np.array([30] * self.nz)

        def transform_slice(z):
            """Transform the velocity slice for a single time step."""
            # compute the coordinate for the ending point of the vector
            vx_slice = self.ds["vx"].isel(mid_date=z).values
            vy_slice = self.ds["vy"].isel(mid_date=z).values
            if vx_slice.shape != grid[0].shape:
                vx_slice = vx_slice.T
                vy_slice = vy_slice.T
            endx = (vx_slice * temp[z] / unit) + grid[0]
            endy = (vy_slice * temp[z] / unit) + grid[1]

            # Transform final coordinates
            t = transformer.transform(endx, endy)
            # Compute differences in the new coordinate system
            vx = (grid_transformed[0] - t[0]) / temp[z] * unit
            vy = (t[1] - grid_transformed[1]) / temp[z] * unit

            return vx, vy

        results = np.array(Parallel(n_jobs=nb_cpu, verbose=0)(delayed(transform_slice)(z) for z in range(self.nz)))

        # results = np.array(
        #     Parallel(n_jobs=nb_cpu, verbose=0)(
        #         delayed(transform_slice, temp, grid, transformer)(z) for z in range(self.nz)
        #     )
        # )
        # Unpack the results
        vx, vy = results[:, 0, :, :], results[:, 1, :, :]

        # Updating DataArrays
        self.ds["vx"] = xr.DataArray(
            vx.astype("float32"),
            dims=["mid_date", "y", "x"],
            coords={"mid_date": self.ds.mid_date, "y": self.ds.y, "x": self.ds.x},
        )
        self.ds["vx"].encoding = {"vx": {"dtype": "float32", "scale_factor": 0.1, "units": "m/y"}}

        self.ds["vy"] = xr.DataArray(
            vy.astype("float32"),
            dims=["mid_date", "y", "x"],
            coords={"mid_date": self.ds.mid_date, "y": self.ds.y, "x": self.ds.x},
        )
        self.ds["vy"].encoding = {"vy": {"dtype": "float32", "scale_factor": 0.1, "units": "m/y"}}

        del grid, transformer, temp, vx, vy

    def align_cube(
        self,
        cube: "CubeDataClass",
        unit: int = 365,
        reproj_vel: bool = True,
        reproj_coord: bool = True,
        interp_method: str = "nearest",
        nb_cpu: int = 8,
    ):
        """
        Reproject cube to match the resolution, projection, and region of self.

        :param cube: Cube to align to self
        :param unit: Unit of the velocities (365 for m/y, 1 for m/d) (default is 365)
        :param reproj_vel: Whether the velocity have to be reprojected or not -> it will modify their value (default is True)
        :param reproj_coord: Whether the coordinates have to be interpolated or not (using interp_method) (default is True)
        :param interp_method: Interpolation method used to reproject cube (default is 'nearest')
        :param nb_cpu: [int] [default is 8] --- number of CPUs used for the parallelization

        :return: Cube projected to self
        """
        # if the velocity components have to be reprojected in the new projection system
        if reproj_vel:
            cube.reproj_vel(cube_to_match=self, unit=unit, nb_cpu=nb_cpu)

        # if the coordinates have to be reprojected in the new projection system
        if reproj_coord:
            cube.reproj_coord(cube_to_match=self)

        cube.ds = cube.ds.assign_attrs({"author": f"{cube.ds.author} aligned"})
        cube.update_dimension()

        return cube

    def merge_cube(self, cube: "CubeDataClass"):
        """
        Merge another cube to the present one. It must have been aligned first (using align_cube)

        :param cube: [cube_data_class] --- The cube to be merged to self
        """
        if self.ds.y.to_pandas().duplicated().any() or self.ds.x.to_pandas().duplicated().any():
            print("Warning: Main cube has duplicate coordinates, removing...")
            if self.ds.y.to_pandas().duplicated().any():
                _, unique_y = np.unique(self.ds.y.values, return_index=True)
                self.ds = self.ds.isel(y=np.sort(unique_y))
            if self.ds.x.to_pandas().duplicated().any():
                _, unique_x = np.unique(self.ds.x.values, return_index=True)
                self.ds = self.ds.isel(x=np.sort(unique_x))

        # Check if the cube to be merged has duplicate coordinates
        if cube.ds.y.to_pandas().duplicated().any() or cube.ds.x.to_pandas().duplicated().any():
            print("Warning: Cube to merge has duplicate coordinates, removing...")
            if cube.ds.y.to_pandas().duplicated().any():
                _, unique_y = np.unique(cube.ds.y.values, return_index=True)
                cube.ds = cube.ds.isel(y=np.sort(unique_y))
            if cube.ds.x.to_pandas().duplicated().any():
                _, unique_x = np.unique(cube.ds.x.values, return_index=True)
                cube.ds = cube.ds.isel(x=np.sort(unique_x))

        # Merge the cubes (must be previously aligned before using align_cube)
        self.ds = xr.concat([self.ds, cube.ds], dim="mid_date")

        # Update the attributes
        self.ds = self.ds.chunk(chunks={"mid_date": self.ds["mid_date"].size})
        self.nz = self.ds["mid_date"].size
        if (
            isinstance(self.filedir, list)
            and isinstance(self.filename, list)
            and isinstance(self.author, list)
            and isinstance(self.source, list)
        ) is False:
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
        time_dim = "mid_date" if "mid_date" in self.ds.dims else "time"
        vx_mean = self.ds["vx"].mean(dim=time_dim)
        vy_mean = self.ds["vy"].mean(dim=time_dim)
        dico_variable = {"vx": vx_mean, "vy": vy_mean}
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

            geopoint = points_heatmap[
                "geometry"
            ].iloc[
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
    def nvvc(self, nb_cpu=8, verbose=True):
        """
        Compute the Normalized Coherence Vector Velocity for every pixel of the cube.

        """

        def ncvv_pixel(cube, i, j):
            return (
                np.sqrt(
                    np.nansum(
                        cube.ds["vx"].isel(x=i, y=j)
                        / np.sqrt(cube.ds["vx"].isel(x=i, y=j) ** 2 + cube.ds["vy"].isel(x=i, y=j) ** 2)
                    )
                    ** 2
                    + np.nansum(
                        cube.ds["vy"].isel(x=i, y=j)
                        / np.sqrt(cube.ds["vx"].isel(x=i, y=j) ** 2 + cube.ds["vy"].isel(x=i, y=j) ** 2)
                    )
                    ** 2
                )
                / cube.nz
            )

        xy_values = itertools.product(range(self.nx), range(self.ny))
        xy_values_tqdm = tqdm(xy_values, total=self.nx * self.ny, mininterval=0.5)

        return np.array(
            Parallel(n_jobs=nb_cpu, verbose=0)(
                delayed(ncvv_pixel)(self, i, j) for i, j in (xy_values_tqdm if verbose else xy_values)
            )
        ).reshape(self.nx, self.ny)

    def compute_med_stable_areas(
        self, shapefile_path, return_as="dataframe", stat_name="med", var_list=["vx", "vy"], invert=True
    ):
        """
        Compute MAD per time step using Dask and apply_ufunc over a shapefile-defined area.

        Parameters:

            shapefile_path (str): Path to shapefile.
            return_as (str): 'dataframe' or 'cube'.
            stat_name (str): Base variable name for new data.
            invert (bool): Whether to invert the shapefile mask.

        Returns:
            pd.DataFrame or xr.Dataset
        """
        # Ensure data has Dask chunks
        # self.ds = self.ds.chunk({'y': -1, 'x': -1, 'mid_date': 10})
        print(var_list)
        # Clip with shapefile
        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(self.ds.rio.crs)
        masked = self.ds.rio.clip(gdf.geometry, gdf.crs, drop=False, all_touched=True, invert=invert)

        print("Clipped")

        # Return as DataFrame
        if return_as == "dataframe":
            df_vx = (
                masked["vx"]
                .median(dim=["x", "y"])
                .compute()
                .to_dataframe(name=f"{stat_name}_vx")
                .reset_index()[["mid_date", f"{stat_name}_vx"]]
            )
            df_vy = (
                masked["vy"]
                .median(dim=["x", "y"])
                .compute()
                .to_dataframe(name=f"{stat_name}_vy")
                .reset_index()[["mid_date", f"{stat_name}_vy"]]
            )
            if len(var_list) == 3:
                df_v = (
                    masked[var_list[2]]
                    .median(dim=["x", "y"])
                    .compute()
                    .to_dataframe(name=f"{stat_name}_v")
                    .reset_index()[["mid_date", f"{stat_name}_v"]]
                )

            # Merge on time coordinate (e.g., 'mid_date')
            if len(var_list) == 3:
                merged_df = reduce(
                    lambda left, right: pd.merge(left, right, on="mid_date", how="outer"), [df_vx, df_vy, df_v]
                )
            else:
                merged_df = pd.merge(df_vx, df_vy, on="mid_date")

            return merged_df

        # # Return as cube
        # elif return_as == 'cube':
        #     return self.assign({f'{stat_name}_vx': mad_results['vx'], f'{stat_name}_vy': mad_results['vy']})

        else:
            raise ValueError("return_as must be 'dataframe' or 'cube'")

    def compute_mad(self, shapefile_path, return_as="dataframe", stat_name="mad", var_list=["vx", "vy"], invert=True):
        """
        Compute MAD per time step using Dask and apply_ufunc over a shapefile-defined area.

        Parameters:

            shapefile_path (str): Path to shapefile.
            return_as (str): 'dataframe' or 'cube'.
            stat_name (str): Base variable name for new data.
            invert (bool): Whether to invert the shapefile mask.

        Returns:
            pd.DataFrame or xr.Dataset
        """
        # Ensure data has Dask chunks
        self.ds = self.ds.chunk({"y": -1, "x": -1, "mid_date": 10})
        print(var_list)
        # Clip with shapefile
        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(self.ds.rio.crs)
        masked = self.ds.rio.clip(gdf.geometry, gdf.crs, drop=False, all_touched=True, invert=invert)

        print("Clipped")

        # Define MAD function
        def mad_2d(arr):
            median = np.nanmedian(arr)
            return 1.483 * np.nanmedian(np.abs(arr - median))

        mad_results = {}  # Store MAD DataArrays

        for var in var_list:
            data = masked[var]

            mad = xr.apply_ufunc(
                mad_2d,
                data,
                input_core_dims=[["y", "x"]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[data.dtype],
            )

            mad.name = f"{stat_name}_{var}"
            mad_results[var] = mad

        # Return as DataFrame
        if return_as == "dataframe":
            df_vx = (
                mad_results["vx"]
                .compute()
                .to_dataframe(name=f"{stat_name}_vx")
                .reset_index()[["mid_date", f"{stat_name}_vx"]]
            )
            df_vy = (
                mad_results["vy"]
                .compute()
                .to_dataframe(name=f"{stat_name}_vy")
                .reset_index()[["mid_date", f"{stat_name}_vy"]]
            )
            if len(var_list) == 3:
                df_v = (
                    mad_results[var_list[2]]
                    .compute()
                    .to_dataframe(name=f"{stat_name}_v")
                    .reset_index()[["mid_date", f"{stat_name}_v"]]
                )

            # Merge on time coordinate (e.g., 'mid_date')
            if len(var_list) == 3:
                merged_df = reduce(
                    lambda left, right: pd.merge(left, right, on="mid_date", how="outer"), [df_vx, df_vy, df_v]
                )
            else:
                merged_df = pd.merge(df_vx, df_vy, on="mid_date")

            return merged_df

        # Return as cube
        elif return_as == "cube":
            return self.assign({f"{stat_name}_vx": mad_results["vx"], f"{stat_name}_vy": mad_results["vy"]})

        else:
            raise ValueError("return_as must be 'dataframe' or 'cube'")
