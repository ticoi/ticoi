"""
Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import math as m

import geopandas as gpd
import matplotlib as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import xarray as xr
from joblib import Parallel, delayed
from shapely.geometry import Point, Polygon

from ticoi.core import interpolation_to_data, inversion_core
from ticoi.cube_data_classxr import cube_data_class


def moving_average_dates(dates: np.ndarray, data: np.ndarray, v_pos: int, save_lines: bool = False) -> np.ndarray:

    """
    Compute the moving average of the velocities from data between the given dates.

    :param dates: [np array] --- An array with all the dates included in data, list
    :param data: [np array] --- An array where each line is (date1, date2, other elements) for which a velocity is computed
    :param v_pos: [int] --- Position in data of the considered variable
    :param save_lines: [bool] [default is False] --- If True, save the lines to use

    :return: [np array] --- Moving average between consecutive dates
    """

    ini = []
    # data[:, v_pos] = np.ma.array(data[:, v_pos])
    for i_date, date in enumerate(dates[:, 0]):
        i = 0
        moy = []

        while i < data.shape[0] and dates[i_date, 1] >= data[i, 0]:  # If the velocity observation is between two dates

            if dates[i_date, 0] <= data[i, 1]:
                moy.append(data[i, v_pos])
            i += 1
        if len(moy) != 0:
            ini.append(np.nanmean(moy))
        else:
            ini.append(np.nan)

    interval_output = (dates[0, 1] - dates[0, 0]) / np.timedelta64(1, "D")
    # ini = np.array([(np.ma.mean(ini[i:i + 2])) for i in range(len(ini) - 1)])
    dates_ini = dates[:, 1] - m.ceil(interval_output / 2)
    if save_lines:
        return np.array([[dates_ini[z], ini[z]] for z in range(len(dates_ini))])
    else:
        return np.array([[dates_ini[z], ini[z]] for z in range(len(dates_ini))])


def find_granule_by_point(input_dict, input_point):  # [lon,lat]

    """
    Takes an input dictionary (a geojson catalog) and a point to represent AOI and returns a list of the s3 urls corresponding to
    zarr datacubes whose footprint covers the AOI.

    :param input_dict: [dict] --- geojson catalog of data cubes
    :param input_point: [list | tuple] --- Point to choose the cube to load ([i, j])

    :return target_granule_urls: [list] --- List of the name of the cubes which cover input_point
    """

    target_granule_urls = []

    point_geom = Point(input_point[0], input_point[1])
    point_gdf = gpd.GeoDataFrame(crs="epsg:4326", geometry=[point_geom])
    for granule in input_dict["features"]:

        bbox_ls = granule["geometry"]["coordinates"][0]
        bbox_geom = Polygon(bbox_ls)
        bbox_gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[bbox_geom])

        if bbox_gdf.contains(point_gdf).all():
            target_granule_urls.append(granule["properties"]["zarr_url"])

    return target_granule_urls


# %%========================================================================= #
#                           OPTIMIZATION FUNCTIONS                            #
# =========================================================================%% #


def RMSE_TICOI_GT(
    data: list,
    mean: list | None,
    dates_range: np.ndarray | None,
    data_gt: pd.DataFrame,
    i: float | int,
    j: float | int,
    coef: int,
    inversion_kwargs: dict,
    interpolation_kwargs: dict,
    regu: int | str | None = None,
    unit: int = 365,
    visual: bool = False,
    plot_raw: bool = False,
    vminmax: list | None = None,
    savedir: str | None = None,
):

    """
    Compute the RMSE between TICOI results with a given coefficient and "ground truth" data.

    :param data: [list] --- An array where each line is (date1, date2, other elements ) for which a velocity is computed (correspond to the original displacements)
    :param mean: [list | None] --- Apriori on the average
    :param dates_range: [np array | None] --- List of np.datetime64 [D], dates of the estimated displacement in X with an irregular temporal sampling (ILF)
    :param data_gt: [pd dataframe] --- "Ground truth" data to which TICOI results are compared
    :params i, j: [float | int] --- Coordinates of the point in pixel
    :param coef: [int] --- Coef of Tikhonov regularisation
    :param inversion_kwargs: [dict] --- Inversion parameters
    :param interpolation_kwargs: [dict] --- Parameters for the interpolation to GT dates (less parameters than for core interpolation)
    :parma unit: [int] [default is 365] --- 365 if the unit is m/y, 1 if the unit is m/d
    :param visual: [bool] [default is False] --- Plot interpolated and GT velocities
    :param plot_raw: [bool] [default is False] --- Add raw data to the plot
    :param vminmax: [list | None] [default is None] --- Specify the vertical limits of the plot
    :param savedir: [str | None] [default is None] --- Save the figure to this location

    :return RMSE: Root Mean Square Error between TICOI results interpolated to "ground truth" (GT) dates, and GT data
    """

    # Proceed to inversion
    if regu is None:
        A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, mean=mean, coef=coef, **inversion_kwargs)
    else:
        A, result, dataf = inversion_core(
            data, i, j, dates_range=dates_range, mean=mean, coef=coef, regu=regu, **inversion_kwargs
        )

    if not visual or not plot_raw:
        del data
    del dates_range, mean

    # Proceed to interpolation
    dataf_lp = interpolation_to_data(result, data_gt, **interpolation_kwargs)
    del A, result, dataf

    # RMSE between TICOI result and ground truth data
    RMSE = sm.root_mean_squared_error(dataf_lp[["vx", "vy"]], data_gt[["vx", "vy"]])

    ##  Plot the interpolated velocity magnitudes along with GT velocity magnitudes
    if visual:
        data_gt = data_gt.reset_index()
        dataf_lp = dataf_lp.reset_index()

        # Magnitude of the velocities
        vv_gt = np.sqrt(data_gt["vx"] ** 2 + data_gt["vy"] ** 2)  # GT data
        vv_lp = np.sqrt(dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2)  # TICOI results interpolated to GT data
        # Offsets and central dates are the same as TICOI was interpolated to GT dates
        offset = data_gt["date2"] - data_gt["date1"]
        central_dates = data_gt["date1"] + offset // 2

        fig, ax = plt.subplots(figsize=(12, 6 / 1.8))

        # Plot raw data
        if plot_raw:
            data = pd.DataFrame(
                data={
                    "date1": data[0][:, 0],
                    "date2": data[0][:, 1],
                    "vx": data[1][:, 0],
                    "vy": data[1][:, 1],
                    "errorx": data[1][:, 2],
                    "errory": data[1][:, 3],
                    "temporal_baseline": data[1][:, 4],
                }
            )
            offset_raw = data["date2"] - data["date1"]
            central_dates_raw = data["date1"] + offset_raw / 2
            vv_raw = np.sqrt(
                (data["vx"] * unit / data["temporal_baseline"]) ** 2
                + (data["vy"] * unit / data["temporal_baseline"]) ** 2
            )

            ax.plot(
                central_dates_raw,
                vv_raw,
                linestyle="",
                color="green",
                zorder=1,
                marker="o",
                lw=0.7,
                markersize=2,
                alpha=0.7,
                label="Central date of velocity observations",
            )
            ax.errorbar(
                central_dates_raw,
                vv_raw,
                xerr=offset_raw / 2,
                color="green",
                alpha=0.2,
                fmt=",",
                zorder=1,
                label="Temporal baseline of velocity observations [days]",
            )

        # Plot interpolated velocities
        ax.plot(
            central_dates,
            vv_lp,
            linestyle="",
            marker="o",
            markersize=3,
            color="b",
            label="Central date of Interpolated velocities (TICOI results)",
        )
        ax.errorbar(
            central_dates,
            vv_lp,
            xerr=offset / 2,
            color="b",
            alpha=0.2,
            fmt=",",
            zorder=1,
            label="Temporal baseline of interpolated velocities",
        )
        # Plot "ground truth" velocities
        ax.plot(
            central_dates,
            vv_gt,
            linestyle="",
            color="orange",
            zorder=1,
            marker="o",
            lw=0.7,
            markersize=2,
            alpha=0.7,
            label="Central date of velocity observations",
        )
        ax.errorbar(
            central_dates,
            vv_gt,
            xerr=offset / 2,
            color="orange",
            alpha=0.2,
            fmt=",",
            zorder=1,
            label="Temporal baseline of velocity observations [days]",
        )
        ax.set_ylabel("Velocity magnitude [m/y]")

        if vminmax is None:
            if plot_raw:
                ax.set_ylim(
                    0.8 * min(np.nanmin(vv_gt), np.nanmin(vv_raw)), 1.2 * max(np.nanmax(vv_gt), np.nanmax(vv_raw))
                )
            else:
                ax.set_ylim(0.8 * np.nanmin(vv_gt), 1.2 * np.nanmax(vv_gt))
        else:
            ax.set_ylim(vminmax)

        ax.legend(loc="lower left", bbox_transform=fig.transFigure, fontsize=7, ncol=2)
        fig.suptitle(f"Magnitude of the velocities (ground truth and interpolated ILF) for coef={coef}", fontsize=16)

        if savedir is not None:
            fig.savefig(f"{savedir}interpol_vv_gt_{coef}.png")

        plt.show()

    del dataf_lp

    return RMSE


def optimize_coef(
    cube: cube_data_class,
    cube_gt: cube_data_class,
    i: float | int,
    j: float | int,
    obs_filt: xr.Dataset,
    load_pixel_kwargs: dict,
    inversion_kwargs: dict,
    interpolation_kwargs: dict,
    regu: dict | None = None,
    flag: xr.DataArray | None = None,
    cmin: int = 10,
    cmax: int = 1000,
    step: int = 10,
    coefs: list | None = None,
    stats: bool = False,
    parallel: bool = False,
    nb_cpu: int = 8,
    **visual_options,
):

    """
    Compute the RMSE between the velocities obtained with TICOI using velocity data and "ground truth" (GT) data for different coefficients.

    :param cube: [cube_data_class] --- Data cube used to compute TICOI at point (i, j)
    :param cube_gt: [cube_data_class] --- Data cube of "ground truth" velocities
    :params i, j: [float | int] --- Coordinates of the point where we want to optimise the coefficient
    :param obs_filt: [xr dataset] --- Filtered dataset (e.g. rolling mean)
    :param load_pixel_kwargs: [dict] --- Pixel loading parameters
    :param inversion_kwargs: [dict] --- Inversion parameters
    :param interpolation_kwargs: [dict] --- Parameters for the interpolation to GT dates (less parameters than for core interpolation)
    :parma regu: [dict | None] [default is None] --- Must be a dictionary if flags is not None, otherwise the regularisation method must be passed in the kwargs
    :param flags: [xr dataarray | None] [default is None] --- Divide the cube in several areas where the coefficient is optimized independently
    :param cmin: [int] [default is 10] --- If coefs=None, start point of the range of coefs to be tested
    :param cmax: [int] [default is 1000] --- If coefs=None, stop point of the range of coefs to be tested
    :param step: [int] [default is 10] --- If coefs=None, step for the range of coefs to be tested
    :param coefs: [list | None] [default is None] --- To specify the coefficients to be tested, if None, range(cmin, cmax, step) coefs will be tested
    :param stats: [bool] [default is False] --- Compute some statistics on raw data and GT data
    :param parallel: [bool] [default is False] --- Should the computation of the results for different coefficient be done using parallelization ?
    :param nb_cpu: [int] [default is 8] --- If parallel is True, the number of CPUs to use for parallelization
    :param visual_options: Additional options for plotting purposes during the computation of the RMSE for each coef

    :return: [pd dataframe] --- Dataframe with the studied coefficients ('coefs'), the resulting RMSEs ('RMSEs'), the standard deviation of similar original and GT data ('std'), how many of those data were used to conduct the computation ('nb_data'), their mean temporal baseline ('temporal_baseline') and their mean velocity values for both x and y components ('mean_v')
    """

    # Load data at pixel
    if flag is not None:
        if "regu" in inversion_kwargs.keys():
            inversion_kwargs.pop("regu")
        data, mean, dates_range, regu, _ = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs, flag=flag)
    else:
        data, mean, dates_range = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)
        regu = None
    dataf = pd.DataFrame(
        data={
            "date1": data[0][:, 0],
            "date2": data[0][:, 1],
            "vx": data[1][:, 0],
            "vy": data[1][:, 1],
            "errorx": data[1][:, 2],
            "errory": data[1][:, 3],
            "temporal_baseline": data[1][:, 4],
        }
    )

    # Load ground truth pixel and convert to pd dataframe
    data_gt = cube_gt.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)[0]
    data_gt = pd.DataFrame(
        data={
            "date1": data_gt[0][:, 0],
            "date2": data_gt[0][:, 1],
            "vx": data_gt[1][:, 0],
            "vy": data_gt[1][:, 1],
            "errorx": data_gt[1][:, 2],
            "errory": data_gt[1][:, 3],
            "temporal_baseline": data_gt[1][:, 4],
        }
    )
    data_gt.index = data_gt["date1"] + (data_gt["date2"] - data_gt["date1"]) // 2

    # Interpolation must be caried out in between the min and max date of the original data
    data_gt = data_gt[(data_gt["date1"] > dataf["date2"].min()) & (data_gt["date2"] < dataf["date2"].max())]

    # Must have enough data to make an interpolation
    if data_gt.shape[0] == 0 and data[0].shape[0] <= 2:
        if stats:
            return None
        return None

    # Coefficients to be tested
    if coefs is None:
        coefs = np.arange(cmin, cmax + 1, step)
    else:
        coefs = np.array(coefs)

    # Compute RMSE for every coefficient
    if parallel:
        RMSEs = Parallel(n_jobs=nb_cpu, verbose=0)(
            delayed(RMSE_TICOI_GT)(
                data,
                mean,
                dates_range,
                data_gt,
                i,
                j,
                coef,
                inversion_kwargs,
                interpolation_kwargs,
                regu=regu,
                **visual_options,
            )
            for coef in coefs
        )
    else:
        RMSEs = [
            RMSE_TICOI_GT(
                data,
                mean,
                dates_range,
                data_gt,
                i,
                j,
                coef,
                inversion_kwargs,
                interpolation_kwargs,
                regu=regu,
                **visual_options,
            )
            for coef in coefs
        ]

    if stats:
        # Average temporal baseline
        temporal_baseline = (dataf["temporal_baseline"].mean(), data_gt["temporal_baseline"].mean())
        # Mean of similar data (same acquisition dates) of raw and GT data
        mean_raw = (
            dataf.groupby(["date1", "date2"], as_index=False)[["vx", "vy", "errorx", "errory"]]
            .mean()[["vx", "vy"]]
            .mean()
        )
        mean_gt = (
            data_gt.groupby(["date1", "date2"], as_index=False)[["vx", "vy", "errorx", "errory"]]
            .mean()[["vx", "vy"]]
            .mean()
        )
        # Standard deviation of similar data (same acquisition dates) of raw and GT data
        std_raw = (
            dataf.groupby(["date1", "date2"], as_index=False)[["vx", "vy", "errorx", "errory"]]
            .std(ddof=0)[["vx", "vy"]]
            .mean()
        )
        std_gt = (
            data_gt.groupby(["date1", "date2"], as_index=False)[["vx", "vy", "errorx", "errory"]]
            .std(ddof=0)[["vx", "vy"]]
            .mean()
        )
        # Standard deviation of raw and GT data
        std_raw_all = dataf[["vx", "vy"]].std(ddof=0)
        std_gt_all = data_gt[["vx", "vy"]].std(ddof=0)

        return xr.DataArray(
            data=RMSEs,
            attrs={
                "regu": inversion_kwargs["regu"] if flag is None else regu,
                "nb_data": (dataf.shape[0], data_gt.shape[0]),
                "mean_temporal_baseline": temporal_baseline,
                "mean_v_similar_data": (mean_raw["vx"], mean_raw["vy"], mean_gt["vx"], mean_gt["vy"]),
                "std_v_similar_data": (std_raw["vx"], std_raw["vy"], std_gt["vx"], std_gt["vy"]),
                "std_raw_data": (std_raw_all["vx"], std_raw_all["vy"], std_gt_all["vx"], std_gt_all["vy"]),
            },
        )

    return xr.DataArray(
        data=RMSEs,
        attrs={
            "regu": inversion_kwargs["regu"] if flag is None else regu,
            "nb_data": (dataf.shape[0], data_gt.shape[0]),
        },
    )
