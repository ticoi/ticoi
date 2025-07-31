import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from ticoi.cube_data_classxr import CubeDataClass
from ticoi.interpolation_functions import smooth_results

# %% ======================================================================== #
#                              Hardcoded configs                              #
# =========================================================================%% #

BASE_CONFIGS = {
    "velocity": {
        "suffixes": ["x", "y", "z", "h"],
        "directions": ["East/West", "North/South", "Up/Down", "nSPF"],
        "unit": "m year-1",
        "var_prefix": "v",
        "final_var_tpl": "v{dim}",
        "long_name_tpl": "velocity in the {direction} direction",
    },
    "displacement": {
        "suffixes": ["x", "y", "z", "h"],
        "directions": ["East/West", "North/South", "Up/Down", "nSPF"],
        "unit": "m",
        "var_prefix": "result_d",
        "final_var_tpl": "d{dim}",
        "long_name_tpl": "cumulative displacement in the {direction} direction",
    },
    "contribution": {
        "flag": "X_contribution",
        "suffixes": ["x", "y", "z", "h"],
        "unit": "count",
        "var_prefix": "xcount_",
        "final_var_tpl": "xcount_{dim}",
        "long_name_tpl": "number of Y observations contributing to X estimation ({dim_upper})",
    },
    "error": {
        "flag": "Error_propagation",
        "suffixes": ["x", "y", "z", "h"],
        "unit": "m year-1",
        "var_prefix": "error_",
        "final_var_tpl": "error_{dim}",
        "long_name_tpl": "Error propagated for the displacement in {dim_upper} direction",
    },
}

QUALITY_METRIC_CONFIGS = {
    "Norm_residual": {
        "vars": ["ResidualAXY_dx", "ResidualRegu_dx", "ResidualAXY_dy", "ResidualRegu_dy"],
        "source_col": "NormR",
        "long_names": [
            "Residual from the inversion AX=Y, where Y is the displacement in the direction Est/West",
            "Residual from the regularisation term for the displacement in the direction Est/West",
            "Residual from the inversion AX=Y, where Y is the displacement in the direction North/South",
            "Residual from the regularisation term for the displacement in the direction North/South",
        ],
        "unit": "m",
    }
}


# %% ======================================================================== #
#                            WRITING RESULTS AS NETCDF                        #
# =========================================================================%% #


class CubeResultsWriter:
    def __init__(self, cube: CubeDataClass):
        self.ds = cube.ds
        self.nx = cube.nx
        self.ny = cube.ny
        self.proj4 = cube.ds.proj4
        self.variable_configs = {}

    def write_result_ticoi(
        self,
        result: list,
        source: str,
        sensor: str,
        filename: str = "Time_series",
        savepath: Optional[str] = None,
        result_quality: Optional[List[str]] = None,
        smooth_res: bool = False,
        smooth_window_size: int = 3,
        return_result: bool = False,
        verbose: bool = False,
    ) -> Union["CubeDataClass", str, Tuple["CubeDataClass", list]]:
        """
        Write the result from TICOI, stored in result, in a xarray dataset matching the conventions CF-1.11
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.pdf

        :param result: [list] --- List of pd xarray, results from the TICOI method
        :param source: [str] --- Name of the source
        :param sensor: [str] --- Sensors which have been used
        :param filename: [str] [default is Time_series] --- Filename of file to saved
        :param savepath: [Optional[str]] [default is None] --- Path to save file
        :param result_quality: [list | str | None] [default is None] --- Which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight)):param savepath: string, path where to save the file
        :param smooth_res: [bool] [default is False] --- Smooth the residuals before saving
        :param smooth_window_size:[int] [default is 3] --- Size of the smoothing kernel
        :param return_result: [bool] [default is False] --- If True, return result
        :param verbose: [bool] [default is False] --- Print information throughout the process

        :return cubenew: [cube_data_class] --- New cube where the results are saved
        """
        if not self._validate_input(result):
            return "No results to write or save."

        dimensions = self._detect_dimensions(result)  # detect needed dimension (x,y and possibly z and h)
        if verbose:
            print(f"[Writing results] Detected dimensions: {dimensions}")

        self.variable_configs = self._generate_variable_configs(
            dimensions
        )  # set variable long_names,short_names, and unit

        time_base, non_null_el = self._get_time_base(result)

        cubenew = self._initialize_cube(time_variable=time_base, add_date_vars=True, non_null_el=non_null_el)

        available_vars = self._detect_available_variables(non_null_el, result_quality)
        self._process_velocity_variables(cubenew, result, available_vars, time_base, smooth_res, smooth_window_size)

        if result_quality:  # if there are quality metrics
            self._process_2d_quality_metrics(cubenew, result, result_quality)

        self._set_metadata(cubenew, source, sensor, dimensions)

        if savepath:
            self._save_cube(cubenew, savepath, filename, verbose)

        return (cubenew, result) if return_result else cubenew

    def write_result_tico(
        self,
        result: list,
        source: str,
        sensor: str,
        filename: str = "Time_series_invert",
        savepath: Optional[str] = None,
        result_quality: Optional[List[str]] = None,
        return_result: bool = False,
        verbose: bool = False,
    ) -> Union["CubeDataClass", str]:
        """
        Write the result from TICOI, stored in result, in a xarray dataset matching the conventions CF-1.11
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.pdf

        :param result: [list] --- List of pd xarray, results from the TICOI method
        :param source: [str] --- Name of the source
        :param sensor: [str] --- Sensors which have been used
        :param filename: [str] [default is Time_series] --- Filename of file to saved
        :param savepath: [Optional[str]] [default is None] --- Path to save file
        :param result_quality: [list | str | None] [default is None] --- Which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
        :param return_result: [bool] [default is False] --- If True, return result
        :param verbose: [bool] [default is False] --- Print information throughout the process

        :return cubenew: [cube_data_class] --- New cube where the results are saved
        """
        if not self._validate_input(result):
            return "No results to write or save."

        dimensions = self._detect_dimensions(result)
        if verbose:
            print(f"[Writing results] Detected dimensions: {dimensions}")

        self.variable_configs = self._generate_variable_configs(dimensions)

        sample = next((r for r in result if not r.empty), None)  # first results not empty
        available_vars = self._detect_available_variables(sample, result_quality)

        reconstructed_data, time_base, ref_dates = self._vectorized_reconstruct(
            result, available_vars
        )  # reconstruct cumulative displacement time series

        cubenew = self._initialize_cube(time_variable=time_base)
        self._set_reference_date(
            cubenew, ref_dates
        )  # set reference date (i.e. the first date of the cumulative displacement time series

        final_var_map = self._build_final_var_map()

        for var_name, data_array in reconstructed_data.items():
            if var_name in final_var_map:
                config, idx = final_var_map[var_name]
                self._add_variable_to_cube(
                    cubenew,
                    var_name,
                    data_array,
                    config["long_names"][idx],
                    config["unit"],
                )

            elif verbose:
                print(f"Warning: Configuration for variable '{var_name}' not found. Skipping.")

        self._set_metadata(cubenew, source, sensor, dimensions)

        if savepath:
            self._save_cube(cubenew, savepath, filename, verbose)

        return (cubenew, result) if return_result else cubenew

    def _process_2d_quality_metrics(self, cube: "CubeDataClass", result: list, result_quality: List[str]):
        """
        Processes and adds 2D quality metrics to the data cube.
        :param cube: [CubeDataClass] --- Cube data class
        :param result: [list] --- List of pd xarray, results from the TICOI method
        :param result_quality: [list | str | None] [default is None] --- Which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
        :return:
        """

        if "Norm_residual" not in result_quality:
            return

        config = QUALITY_METRIC_CONFIGS.get("Norm_residual")
        if not config:
            return

        source_col = config["source_col"]
        sample = next((r for r in result if not r.empty and source_col in r), None)
        if sample is None:
            return

        for i, var_name in enumerate(config["vars"]):
            data_arr = np.full((self.nx, self.ny), np.nan, dtype=np.float32)

            for p_idx, df in enumerate(result):
                if not df.empty and source_col in df and df[source_col].shape[0] > i:
                    x = p_idx // self.ny
                    y = p_idx % self.ny
                    data_arr[x, y] = df[source_col][i]

            cube.ds[var_name] = xr.DataArray(data_arr, dims=["x", "y"], coords={"x": cube.ds["x"], "y": cube.ds["y"]})
            cube.ds[var_name] = cube.ds[var_name].transpose("y", "x")
            cube.ds[var_name].attrs = {
                "short_name": var_name,
                "unit": config["unit"],
                "long_name": config["long_names"][i],
                "grid_mapping": "grid_mapping",
            }

    def _build_final_var_map(self) -> Dict[str, Tuple[Dict, int]]:
        """
        Builds a mapping from a final variable name (e.g., 'dx') to its config and index.
        :return:
        """
        final_var_map = {}
        for config in self.variable_configs.values():
            for i, final_var in enumerate(config.get("final_vars", [])):
                final_var_map[final_var] = (config, i)
        return final_var_map

    def _vectorized_reconstruct(
        self, result: list, available_vars: Dict
    ) -> Tuple[Dict[str, np.ndarray], pd.Series, np.ndarray]:
        """
        A fully vectorized replacement for the original `reconstruct_common_ref` loop.
        :param result: [list] --- List of pd xarray, result from the TICOI method
        :param available_vars:[Dict] --- dictionary of available variables
        :return:
        """

        all_dates = sorted(list({date for df in result if not df.empty for date in df["date2"]}))
        time_axis = pd.Series(all_dates, dtype="datetime64[ns]")
        time_len = len(time_axis)

        vars_to_process = []
        for var_type, final_var_list in available_vars.items():
            if var_type in ["displacement", "contribution", "error"]:
                config = self.variable_configs[var_type]
                for final_var in final_var_list:
                    if final_var in config["final_vars"]:
                        idx = config["final_vars"].index(final_var)
                        vars_to_process.append(config["vars"][idx])

        if not vars_to_process:
            return {}, time_axis, np.full((self.nx, self.ny), np.nan, dtype="datetime64[ns]")

        final_var_names = {v: v.replace("result_d", "d") for v in vars_to_process}

        reconstructed_data = {
            final_name: np.full((self.nx, self.ny, time_len), np.nan, dtype=np.float32)
            for final_name in final_var_names.values()
        }  # initialize the reconstructed array as 3D array
        ref_dates_array = np.full((self.nx, self.ny), np.nan, dtype="datetime64[ns]")

        max_pixel_len = 0
        if result and any(not df.empty for df in result):
            max_pixel_len = max(len(df) for df in result if not df.empty)  # maximal temporal length of each pixel

        if max_pixel_len == 0:  # empty cube
            return {}, time_axis, ref_dates_array

        packed_data = {
            v: np.full((self.nx * self.ny, max_pixel_len), np.nan, dtype=np.float32) for v in vars_to_process
        }  # flatten spatial dimensions
        packed_dates = np.full(
            (self.nx * self.ny, max_pixel_len), np.nan, dtype="datetime64[ns]"
        )  # flatten spatial dimensions
        pixel_lengths = np.zeros(self.nx * self.ny, dtype=int)

        for i, df in enumerate(result):
            if not df.empty:
                n = len(df)
                pixel_lengths[i] = n
                ref_dates_array[i // self.ny, i % self.ny] = df["date1"].iloc[0]
                packed_dates[i, :n] = df["date2"].values
                for v in vars_to_process:
                    if v in df:
                        packed_data[v][i, :n] = df[v].values

        cumulative_data = {
            v: np.nancumsum(arr, axis=1) for v, arr in packed_data.items()
        }  # cumulative summation of displacement along time

        # put the results in a 3D array
        for i in range(self.nx * self.ny):
            n = pixel_lengths[i]
            if n > 0:
                pixel_dates = packed_dates[i, :n]
                insert_indices = np.searchsorted(time_axis, pixel_dates)

                x, y = i // self.ny, i % self.ny
                for v_orig, v_cum in cumulative_data.items():
                    v_new = final_var_names[v_orig]
                    reconstructed_data[v_new][x, y, insert_indices] = v_cum[i, :n]

        return reconstructed_data, time_axis, ref_dates_array

    def _prepare_variable_array(self, result: list, var: str, time_len: int) -> np.ndarray:
        """
        Efficiently prepares a 3D numpy array for a given variable from the result list.
        :param result: [list] ---  list with results from ticoi or tico
        :param var : [str] --- variable name
        :param time_len : [int] --- length of the time axis
        :return: 3D numpy array
        """
        final_array = np.full((self.nx, self.ny, time_len), np.nan, dtype=np.float32)
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j
                if idx < len(result) and not result[idx].empty and var in result[idx]:
                    data_slice = result[idx][var].values
                    if data_slice.shape[0] == time_len:
                        final_array[i, j, :] = data_slice
        return final_array

    def _process_velocity_variables(
        self,
        cube: "CubeDataClass",
        result: list,
        available_vars: Dict,
        time_variable: pd.Series,
        smooth_res: bool,
        smooth_window_size: int,
    ):
        """
        Process and add all detected velocity-related variables to the data cube.
        :param cube : [CubeDataClass] --- cube we are saving
        :param result: [list] --- List of pd xarray, results from the TICOI method
        :param available_vars:[Dict] --- dictionary of available variables
        :param time_variable : [pd.Series] --- centered dates for each estimation
        :param smooth_res: [bool] [default is False] --- Smooth the residuals before saving
        :param smooth_window_size:[int] [default is 3] --- Size of the smoothing kernel

        :return:
        """
        time_len = len(time_variable)

        for var_type, var_list in available_vars.items():
            if var_type not in self.variable_configs:
                continue
            config = self.variable_configs[var_type]

            for i, final_var in enumerate(config.get("final_vars", [])):
                if final_var not in var_list:
                    continue
                original_var_name = config["vars"][i]
                result_arr = self._prepare_variable_array(result, original_var_name, time_len)  # create a 3D np array

                if smooth_res and var_type == "velocity":
                    result_arr = self._smooth_array(
                        result_arr, smooth_window_size
                    )  # smooth the result by applying a spatial smoothing
                    self._update_result_list(result, original_var_name, result_arr)

                self._add_variable_to_cube(
                    cube,
                    final_var,
                    result_arr,
                    config["long_names"][i],
                    config["unit"],
                )

    def _initialize_cube(
        self, time_variable: pd.Series, add_date_vars: bool = False, non_null_el: Optional[pd.DataFrame] = None
    ) -> "CubeDataClass":
        """
        Initialize a data cube with basic coordinates and time variables.
        :param time_variable [pd.Series]: centered dates for each estimation
        :param add_date_vars [bool]: If yes, add also the two dates between each the velocity have been estimated
        :param non_null_el [Optional[pd.DataFrame]]: results which are not null
        :return:
        """
        cubenew = CubeDataClass()
        cubenew.nx = self.nx
        cubenew.ny = self.ny
        cubenew.proj4 = self.proj4

        x_attrs = {"standard_name": "projection_x_coordinate", "units": "m", "long_name": "x coordinate of projection"}
        y_attrs = {"standard_name": "projection_y_coordinate", "units": "m", "long_name": "y coordinate of projection"}

        epoch = pd.Timestamp("1970-01-01")
        time_values = (time_variable - epoch).dt.total_seconds() / (24 * 3600)
        time_attrs = {
            "standard_name": "time",
            "long_name": "center date of the velocity estimation",
            "units": "days since 1970-01-01 00:00:00",
            "calendar": "gregorian",
        }

        cubenew.ds = xr.Dataset(
            coords={
                "x": ("x", self.ds["x"].values, x_attrs),
                "y": ("y", self.ds["y"].values, y_attrs),
                "time": ("time", time_values.values, time_attrs),
            }
        )

        # Set grid mapping variable
        cubenew.ds.rio.write_crs(self.proj4, inplace=True)
        grid_mapping_attrs = cubenew.ds.coords["spatial_ref"].attrs
        cubenew.ds = cubenew.ds.drop_vars("spatial_ref")
        cubenew.ds["grid_mapping"] = xr.DataArray(0, attrs=grid_mapping_attrs)

        if add_date_vars and non_null_el is not None:
            date1_values = (non_null_el["date1"] - epoch).dt.total_seconds() / (24 * 3600)
            date2_values = (non_null_el["date2"] - epoch).dt.total_seconds() / (24 * 3600)
            time_bnds_data = np.vstack([date1_values, date2_values]).T
            cubenew.ds["time_bnds"] = (("time", "bnds"), time_bnds_data)
            cubenew.ds["time"].attrs["bounds"] = "time_bnds"

        return cubenew

    def _add_variable_to_cube(
        self,
        cube: "CubeDataClass",
        var: str,
        data: np.ndarray,
        long_name: str,
        unit: str,
    ):
        """
        Add a variable as a DataArray to the data cube.
        :param cube: [CubeDataClass] --- Cube data class
        :param var: [str] --- variable name
        :param data: [np.ndarray] --- data array to add as variable
        :param long_name: [str] --- long name of the variable
        :param unit: [str] --- unit of the variable
        :return:
        """
        data_array = xr.DataArray(
            data, dims=["x", "y", "time"], coords={"x": cube.ds["x"], "y": cube.ds["y"], "time": cube.ds["time"]}
        )
        cube.ds[var] = data_array.transpose("time", "y", "x")
        attrs = {"units": unit, "long_name": long_name, "grid_mapping": "grid_mapping"}

        attrs["short_name"] = var  # no standard_name exist for our variables
        cube.ds[var].attrs = attrs

    def _set_reference_date(self, cube: "CubeDataClass", ref_dates: np.ndarray):
        """
        Set the reference date for displacement time series.
        :param cube: [CubeDataClass] --- Cube data class
        :param ref_dates: [np.ndarray] --- reference dates
        :return:
        """
        epoch = pd.Timestamp("1970-01-01")
        # This handles NaT (Not a Time) values, which will become NaN after conversion.
        numerical_dates = (pd.to_datetime(ref_dates.flatten()) - epoch).total_seconds() / (24 * 3600)
        numerical_dates_arr = numerical_dates.values.reshape(ref_dates.shape)

        cube.ds["reference_date"] = xr.DataArray(
            numerical_dates_arr, dims=["x", "y"], coords={"x": cube.ds["x"], "y": cube.ds["y"]}
        )
        cube.ds["reference_date"].attrs = {
            "long_name": "First date of the cumulative displacement time series",
            "units": "days since 1970-01-01 00:00:00",
        }

    def _validate_input(self, result: list) -> bool:
        """
        Check if the inputs are valid
        :param result: [list] --- List of pd xarray, results from the TICOI method
        :return:
        """
        return bool(result) and any(not r.empty for r in result)

    def _get_time_base(self, result: list) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Get the centered date (time_variable) and the results which are not null
        :param result: [list] --- List of pd xarray, results from the TICOI method
        :return: entered date (time_variable) and the results which are not null
        """
        non_null_el = next((r for r in result if not r.empty), None)
        if non_null_el is None:
            return pd.Series([], dtype="datetime64[ns]"), None
        time_variable = non_null_el["date1"] + (non_null_el["date2"] - non_null_el["date1"]) / 2
        return time_variable, non_null_el

    def _detect_dimensions(self, result: list) -> List[str]:
        """
        Detect the dimension in cube result
        :param result: [list] --- List of pd xarray, results from the TICOI method
        :return: list of dimensions
        """
        sample = next((r for r in result if not r.empty), None)
        if sample is None:
            return []
        dim_map = {
            "vx": "x",
            "vy": "y",
            "vz": "z",
            "vh": "h",
            "result_dx": "x",
            "result_dy": "y",
            "result_dz": "z",
            "result_dh": "h",
        }
        return sorted(list({dim_map[col] for col in sample.columns if col in dim_map}))

    def _generate_variable_configs(self, dimensions: List[str]) -> Dict[str, Dict]:
        """
        Generate config files
        :param dimensions [List[str]]:
        :return: dict o configs
        """
        configs = {}
        for var_type, base_config in BASE_CONFIGS.items():
            vars_list, long_names, final_vars = [], [], []
            for dim in dimensions:
                if dim not in base_config["suffixes"]:
                    continue  # if the dimension is not defined
                vars_list.append(base_config["var_prefix"] + dim)
                final_vars.append(base_config["final_var_tpl"].format(dim=dim))
                direction = base_config.get("directions", [""] * len(dimensions))[base_config["suffixes"].index(dim)]
                long_names.append(base_config["long_name_tpl"].format(direction=direction, dim_upper=dim.upper()))

            if vars_list:
                configs[var_type] = {
                    "vars": vars_list,
                    "long_names": long_names,
                    "unit": base_config["unit"],
                    "final_vars": final_vars,
                    "flag": base_config.get("flag"),
                }
        return configs

    def _detect_available_variables(
        self, sample_result: pd.DataFrame, result_quality: Optional[List[str]]
    ) -> Dict[str, List[str]]:
        """
        Detect variable names inside the cube result
        :param sample_result [pd.DataFrame]: result for one particular date
        :param result_quality: [list | str | None] [default is None] --- Which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
        :return:
        """
        if sample_result is None:
            return {}

        # Get available variable
        available = {}
        for var_type, config in self.variable_configs.items():
            # Always include base types if they exist
            if var_type in ["velocity", "displacement"]:
                if any(var in sample_result for var in config["vars"]):
                    available[var_type] = config.get("final_vars", [])
                continue

            # For quality metrics, check if the flag is set
            if result_quality and config.get("flag") in result_quality:
                if any(var in sample_result for var in config["vars"]):
                    available[var_type] = config.get("final_vars", [])

        return available

    def _set_metadata(self, cube: "CubeDataClass", source: str, sensor: str, dimensions: List[str]):
        """
        Set the global attributes of the cube
        :param cube: [CubeDataClass] --- Cube data class
        :param source: [str] --- processing steps that have been applied
        :param sensor: [str] --- satellite sensors used to compute the original  displacements
        :param dimensions: List[str] -- dimensions of the cube
        """
        cube.ds.attrs = {
            "Conventions": "CF-1.11",
            "title": "Ice velocity and displacement time series",
            "institution": "Université Grenoble Alpes",
            "source": source,
            "sensor": sensor,
            "proj4": self.ds.proj4,
            "author": "L. Charrier",
            "history": f"Created on {datetime.date.today()}",
            "dimensions": f"{len(dimensions)}D ({', '.join(dimensions)})",
            "references": "Charrier, L., et al. (2025)",
        }

    def _save_cube(self, cube: "CubeDataClass", savepath: str, filename: str, verbose: bool):
        """
        Saves the data cube to a NetCDF file with appropriate encoding.
        :param cube: [CubeDataClass] --- Cube data class
        :param savepath: [Optional[str]] [default is None] --- Path to save file
        :param filename: [str] [default is Time_series] --- Filename of file to saved
        :param verbose: [bool] [default is False] --- Print information throughout the process
        :return:
        """
        encoding = {}
        for var in cube.ds.data_vars:
            if var in cube.ds.coords or var == "grid_mapping":
                continue
            encoding[var] = {"zlib": True, "complevel": 5, "dtype": "int16" if var.startswith("xcount") else "float32"}

        if "time_bnds" in cube.ds:
            encoding["time_bnds"] = {"_FillValue": None}

        filepath = f"{savepath}/{filename}.nc"
        cube.ds.to_netcdf(filepath, engine="h5netcdf", encoding=encoding)
        if verbose:
            print(f"[Writing results] Saved to {filepath}")

    def _parse_proj4_to_cf_attrs(self) -> dict:
        """convert proj4 string to CF attributes."""
        attrs = {}
        proj_map = {
            "proj": "grid_mapping_name",
            "lat_0": "latitude_of_projection_origin",
            "lon_0": "longitude_of_projection_origin",
            "lat_ts": "standard_parallel",
            "x_0": "false_easting",
            "y_0": "false_northing",
            "datum": "datum",
        }
        value_map = {"stere": "polar_stereographic"}

        # BUG FIX: Robustly parse proj4 string to handle flags without values
        params = {}
        for item in self.proj4.replace("+", "").strip().split():
            if "=" in item:
                key, value = item.split("=", 1)
                params[key] = value
            else:
                params[item] = True  # Treat flags like 'no_defs' as boolean

        for key, value in params.items():
            if key in proj_map:
                try:
                    # Attempt to convert to float, otherwise use string value
                    cf_value = float(value_map.get(value, value))
                except (ValueError, TypeError):
                    cf_value = value_map.get(value, value)
                attrs[proj_map[key]] = cf_value

        if attrs.get("datum") == "WGS84":
            attrs.update({"semi_major_axis": 6378137.0, "inverse_flattening": 298.257223563})

        attrs["crs_wkt"] = self.proj4
        return attrs

    def _smooth_array(self, array: np.ndarray, smooth_window_size: int) -> np.ndarray:
        """

        :param array: [np.ndarray] --- array to be smoothed
        :param smooth_window_size:[int] [default is 3] --- size of the smoothing kernel
        :return:
        """
        return smooth_results(array, window_size=smooth_window_size)

    def _update_result_list(self, result: list, var: str, smoothed_array: np.ndarray):
        """

        :param result: [list] --- list of pd xarray, results from the TICOI method
        :param var: [str] --- name of the variable
        :param smoothed_array: [np.ndarray] --- smoothed array
        :return:
        """
        for x in range(self.nx):
            for y in range(self.ny):
                idx = x * self.ny + y
                if idx < len(result) and not result[idx].empty:
                    result[idx][var] = smoothed_array[x, y, :]

    def write_results_ticoi_or_tico(
        self,
        result: list,
        source: str,
        sensor: str,
        filename: str = "Time_series",
        savepath: str | None = None,
        result_quality: list | None = None,
        verbose: bool = False,
    ) -> Union["CubeDataClass", str]:
        """
        Write the result from TICOI or TICO, stored in result, in a xarray dataset matching the conventions CF-1.10
        It recognizes whether the results are irregular or regular and uses the appropriate saving method
        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.pdf
        units has been changed to unit, since it was producing an error while wirtting the netcdf file

        :param result: [list] --- List of pd xarray, results from the TICOI method
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
        return self
