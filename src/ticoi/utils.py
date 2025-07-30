"""
Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import json
import math as m
import urllib.request

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sm
import xarray as xr
from joblib import Parallel, delayed
from pyproj import CRS
from shapely.geometry import Point, Polygon

from ticoi.core import interpolation_core, interpolation_to_data, inversion_core
from ticoi.cube_data_classxr import CubeDataClass


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


def find_granule_by_point(input_point):  # [lon,lat]
    """
    Takes an input dictionary (a geojson catalog) and a point to represent AOI and returns a list of the s3 urls corresponding to
    zarr datacubes whose footprint covers the AOI.
    Function from https://github.com/e-marshall/itslivetools.git

    :param input_point: [list | tuple] --- Point to choose the cube to load ([i, j])

    :return target_granule_urls: [list] --- List of the name of the cubes which cover input_point

    """

    with urllib.request.urlopen("https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json") as url:
        itslive_catalog = json.loads(url.read().decode())

        target_granule_urls = []

        point_geom = Point(input_point[0], input_point[1])
        point_gdf = gpd.GeoDataFrame(crs="epsg:4326", geometry=[point_geom])
        for granule in itslive_catalog["features"]:
            bbox_ls = granule["geometry"]["coordinates"][0]
            bbox_geom = Polygon(bbox_ls)
            bbox_gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[bbox_geom])

            if bbox_gdf.contains(point_gdf).all():
                target_granule_urls.append(granule["properties"]["zarr_url"])

    return target_granule_urls


def points_of_shp_line(shp_file, proj="EPSG:4326", distance=50, nb_points=None, select=None):
    geolns = gpd.read_file(shp_file)
    if geolns.geom_type.describe()["top"] not in ["LineString", "MultiLineString"]:
        raise ValueError("The shp geometries must be a LineString.")

    # The selection is given in kilometers -> convert it to meters
    if select is not None and (
        select[0] < 2 * distance
        if distance is not None
        else 100 or select[1] < 2 * distance
        if distance is not None
        else 100
    ):
        select[0] *= 1000
        select[1] *= 1000

    if nb_points is not None:
        if isinstance(select, list) and len(select) == 2:
            point_dist = np.linspace(max(0, select[0]), min(int(geolns.geometry.length.values), select[1]), nb_points)
        else:
            point_dist = np.linspace(0, int(geolns.geometry.length.values), nb_points)

    elif distance is not None:
        if isinstance(select, list) and len(select) == 2:
            point_dist = np.arange(max(0, select[0]), min(int(geolns.geometry.length.values), select[1] + 1), distance)
        else:
            point_dist = np.arange(0, int(geolns.geometry.length.values), distance)

    else:
        raise ValueError("One of 'distance' or 'nb_points' parameters must not be None.")

    # Projection of the shapeline coordinates into the cube coordinates system
    if proj is not None and CRS(proj) != geolns.crs:
        geolns = geolns.to_crs(CRS(proj))

    # Retrieve the points from the line (interpolation)
    geopnts = pd.DataFrame(
        {
            "distance": point_dist,
            "geometry": [(geolns.geometry.interpolate(z, normalized=False)).iloc[0] for z in point_dist],
        }
    )

    return geopnts


def draw_heatmap(
    line_df,
    savepath=None,
    vminmax=[False, False],
    name="heatmap",
    legend="Distance along a longitudinal profile",
    maplabel="Mean of velocity magnitude [m/y]",
    title="",
    cmap="rainbow",
    figsize=(10, 8),
    centered=False,
    x_tick_frequency=10,
    y_tick_frequency=3,
):
    """
    Draw an hovmoller diagram (heatmap).

    :param savepath: str or None, path where to save the figure. If None, the figure is not saved
    :param vminmax: [int, int] or [False, False] : min and max values used for the plot
    :param name: str, name of the figure to save
    :param legend: str, legend of the x-axis
    :param maplabel: str, label of the colormap
    :param title: str, title of the figure
    :param cmap: str, colormap e.g. coolwarm
    :param figsize: (int, int), size of the figure
    :param centered: bool, if True the colormap is centered on 0
    :param x_tick_frequency: tick frequency for x-axis, the legend will be displayed every x_thick_frequency * num meter (e.g. 100 * 10 = 1 km)
    :param y_tick_frequency: tick frequency for y-axis, the legend will be displayed every y_thick_frequency * freq heatmap (e.g. 3 * 1 = 3 months)
    """

    line_df = line_df.astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    if vminmax == [False, False]:  # bound : mean-3std; mean+3 std
        vminmax = [line_df.mean().mean() - 3 * line_df.std().std(), line_df.mean().mean() + 3 * line_df.std().std()]
        vminmax = [line_df.min().min(), line_df.max().max()]

    if centered:  # Center the colormap on 0
        ax = sns.heatmap(
            data=line_df, vmin=vminmax[0], vmax=vminmax[1], cbar_kws={"label": f"\n {maplabel}"}, cmap=cmap, center=0
        )
    else:
        ax = sns.heatmap(
            data=line_df, vmin=vminmax[0], vmax=vminmax[1], cbar_kws={"label": f"\n {maplabel}"}, cmap=cmap
        )

    # Display x-ticks
    x_ticks = range(0, len(line_df.columns), x_tick_frequency)
    x_tick_labels = [f"{line_df.columns[i]}" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=14)

    # Display y-ticks
    y_ticks = range(0, len(line_df.index), y_tick_frequency)
    y_tick_labels = [line_df.index[i].strftime("%Y-%m") for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=14)

    # Create a black line every year
    y_tick_frequency = 12
    y_ticks = range(0, len(line_df.index), y_tick_frequency)
    for year in y_ticks:
        ax.axhline(year, color="k", linewidth=1)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=14)

    ax.figure.axes[-1].yaxis.label.set_size(16)
    ax.set_title(f"{title}", pad=20, fontsize=16)
    ax.set_xlabel(f"{legend}", labelpad=8, fontsize=16)
    ax.set_ylabel("Central date", fontsize=16)
    if savepath is not None:
        fig.savefig(f"{savepath}{name}.png")

    plt.show()


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
    method: str = "stable_ground",
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
    :param method: [str] [default is 'stable_ground'] --- Method to be used to optimise the coef (among 'ground_truth' and 'stable_ground')
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
    if method == "ground_truth":
        dataf_lp = interpolation_to_data(result, data_gt, **interpolation_kwargs)
        del A, result, dataf

        # RMSE between TICOI result and ground truth data
        RMSE = np.sqrt(sm.mean_squared_error(dataf_lp[["vx", "vy"]], data_gt[["vx", "vy"]]))

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
            fig.suptitle(
                f"Magnitude of the velocities (ground truth and interpolated ILF) for coef={coef}", fontsize=16
            )

            if savedir is not None:
                fig.savefig(f"{savedir}interpol_vv_gt_{coef}.png")

            plt.show()

        return RMSE, None

    elif method == "stable_ground":
        dataf_lp = interpolation_core(result, **interpolation_kwargs)
        del A, result, dataf

        data_gt = pd.DataFrame(
            data={
                "vx": np.array([0 for _ in range(dataf_lp.shape[0])]),
                "vy": np.array([0 for _ in range(dataf_lp.shape[0])]),
            },
            index=dataf_lp.index,
        )

        # RMSE between TICOI result and ground truth data
        RMSE = np.sqrt(sm.mean_squared_error(dataf_lp[["vx", "vy"]], data_gt[["vx", "vy"]]))

        # TODO Option to visualize the results (if visual)

        return RMSE, dataf_lp.shape[0]

    else:
        raise ValueError("Please select 'ground_truth' or 'stable_ground' as method")


def VVC_TICOI(
    data: list,
    mean: list | None,
    dates_range: np.ndarray | None,
    i: float | int,
    j: float | int,
    coef: int,
    inversion_kwargs: dict,
    interpolation_kwargs: dict,
    regu: int | str | None = None,
):
    """
    Compute TICOI for one particular coefficient, and compute the VVC
    :param data:
    :param mean:
    :param dates_range:
    :param i:
    :param j:
    :param coef:
    :param inversion_kwargs:
    :param interpolation_kwargs:
    :param regu:
    :return:
    """
    # Proceed to inversion
    if regu is None:
        A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, mean=mean, coef=coef, **inversion_kwargs)
    else:
        A, result, dataf = inversion_core(
            data, i, j, dates_range=dates_range, mean=mean, coef=coef, regu=regu, **inversion_kwargs
        )

    # Interpolation
    dataf_lp = interpolation_core(result, **interpolation_kwargs)
    del A, result, dataf

    VVC = (
        np.sqrt(
            np.nansum(dataf_lp["vx"] / np.sqrt(dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2)) ** 2
            + np.nansum(dataf_lp["vy"] / np.sqrt(dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2)) ** 2
        )
        / dataf_lp.shape[0]
    )

    return VVC, dataf_lp.shape[0]


def optimize_coef(
    cube: CubeDataClass,
    cube_gt: CubeDataClass,
    i: float | int,
    j: float | int,
    obs_filt: xr.Dataset,
    load_pixel_kwargs: dict,
    inversion_kwargs: dict,
    interpolation_kwargs: dict,
    method: str = "vvc",
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
    Optimization of the regularization coefficient value for the TICOI post-processing method, either by comparing the results
    to a "ground truth" (method='ground_truth') or a zero velocity in stable ground ('stable_ground'), or by computing the
    Velocity Vector Coherence of the results (method = 'vvc').

    :param cube: [cube_data_class] --- Data cube used to compute TICOI at point (i, j)
    :param cube_gt: [cube_data_class] --- Data cube of "ground truth" velocities
    :params i, j: [float | int] --- Coordinates of the point where we want to optimise the coefficient
    :param obs_filt: [xr dataset] --- Filtered dataset (e.g. rolling mean)
    :param load_pixel_kwargs: [dict] --- Pixel loading parameters
    :param inversion_kwargs: [dict] --- Inversion parameters
    :param interpolation_kwargs: [dict] --- Parameters for the interpolation to GT dates (less parameters than for core interpolation)
    :param method: [str] [default is 'vvc'] --- Method used to optimize the coef ('ground_truth', 'stable_ground' or 'vvc')
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
    data_gt = None
    if method == "ground_truth":
        assert cube_gt is not None, "Please provide ground truth data for method 'ground_truth'"

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
        coefs = np.arange(cmin, cmax + 1, step)  # range of coef
    else:
        coefs = np.array(coefs)

    if method == "ground_truth" or method == "stable_ground":
        # Compute RMSE for every coefficient
        if parallel:
            measures = Parallel(n_jobs=nb_cpu, verbose=0)(
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
                    method=method,
                    regu=regu,
                    **visual_options,
                )
                for coef in coefs
            )
        else:
            measures = [
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
                    method=method,
                    regu=regu,
                    **visual_options,
                )
                for coef in coefs
            ]

    elif method == "vvc":
        if parallel:
            measures = Parallel(n_jobs=nb_cpu, verbose=0)(
                delayed(VVC_TICOI)(
                    data, mean, dates_range, i, j, coef, inversion_kwargs, interpolation_kwargs, regu=regu
                )
                for coef in coefs
            )
        else:
            measures = [
                VVC_TICOI(data, mean, dates_range, i, j, coef, inversion_kwargs, interpolation_kwargs, regu=regu)
                for coef in coefs
            ]

    data_gt_shape = measures[0][1]
    measures = [measures[i][0] for i in range(len(coefs))]

    if stats:
        mean_disp = (dataf["vx"].mean(), dataf["vy"].mean())  # Displacements mean
        directions = np.arctan(dataf["vy"] / dataf["vx"])
        mean_angle_to_median = np.mean(directions - np.median(directions))

        dataf["vx"] = dataf["vx"] * 365 / dataf["temporal_baseline"]
        dataf["vy"] = dataf["vy"] * 365 / dataf["temporal_baseline"]

        # Average temporal baseline
        temporal_baseline = dataf["temporal_baseline"].mean()

        # Mean of similar data (same acquisition dates) of raw and GT data
        mean_raw = (
            dataf.groupby(["date1", "date2"], as_index=False)[["vx", "vy", "errorx", "errory"]]
            .mean()[["vx", "vy"]]
            .mean()
        )
        # Standard deviation of similar data (same acquisition dates) of raw and GT data
        std_raw = (
            dataf.groupby(["date1", "date2"], as_index=False)[["vx", "vy", "errorx", "errory"]]
            .std(ddof=0)[["vx", "vy"]]
            .mean()
        )
        # Standard deviation of raw data
        std_raw_all = dataf[["vx", "vy"]].std(ddof=0)

        if method == "ground_truth":
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
                data=measures,
                attrs={
                    "regu": inversion_kwargs["regu"] if flag is None else regu,
                    "nb_data": (dataf.shape[0], data_gt.shape[0]),
                    "mean_temporal_baseline": temporal_baseline,
                    "mean_disp": mean_disp,
                    "mean_angle_to_median": mean_angle_to_median,
                    "mean_v": (dataf["vx"].mean(), dataf["vy"].mean(), data_gt["vx"].mean(), data_gt["vy"].mean()),
                    "mean_v_similar_data": (mean_raw["vx"], mean_raw["vy"], mean_gt["vx"], mean_gt["vy"]),
                    "std_v_similar_data": (std_raw["vx"], std_raw["vy"], std_gt["vx"], std_gt["vy"]),
                    "std_raw_data": (std_raw_all["vx"], std_raw_all["vy"], std_gt_all["vx"], std_gt_all["vy"]),
                },
            )

        elif method == "stable_ground" or method == "vvc":
            return xr.DataArray(
                data=measures,
                attrs={
                    "regu": inversion_kwargs["regu"] if flag is None else regu,
                    "nb_data": (dataf.shape[0], data_gt_shape),
                    "mean_temporal_baseline": temporal_baseline,
                    "mean_disp": mean_disp,
                    "mean_angle_to_median": mean_angle_to_median,
                    "mean_v": (dataf["vx"].mean(), dataf["vy"].mean()),
                    "mean_v_similar_data": (mean_raw["vx"], mean_raw["vy"]),
                    "std_v_similar_data": (std_raw["vx"], std_raw["vy"]),
                    "std_raw_data": (std_raw_all["vx"], std_raw_all["vy"]),
                },
            )

    return xr.DataArray(
        data=measures,
        attrs={
            "regu": inversion_kwargs["regu"] if flag is None else regu,
            "nb_data": (dataf.shape[0], data_gt.shape[0] if data_gt is not None else None),
        },
    )
