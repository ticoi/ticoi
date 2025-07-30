"""
Auxiliary functions to process the temporal interpolation.

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy import interpolate

from ticoi.pixel_class import PixelClass


def reconstruct_common_ref(
    result: pd.DataFrame,
    second_date_list: List[np.datetime64] | None = None,
) -> pd.DataFrame:
    """
    Build the Cumulative Displacements (CD) time series with a Common Reference (CR) from a Leap Frog time series

    :param result: [np array] --- Leap frog displacement for x-component and y-component
    :param second_date_list: [list] --- List of dates in which the leap frog displacement will be reindexed

    :return data: [pd dataframe] --- Cumulative displacement time series in x and y component, pandas dataframe
    """

    if result.empty:
        length = 1 if second_date_list is None else len(second_date_list)
        nan_list = np.full(length, np.nan)
        second_dates = [np.nan] if second_date_list is None else second_date_list
        return pd.DataFrame(
            {
                "Ref_date": nan_list,
                "Second_date": second_dates,
                "dx": nan_list,
                "dy": nan_list,
                "xcount_x": nan_list,
                "xcount_y": nan_list,
            }
        )

    # Common Reference
    data = pd.DataFrame(
        {
            "Ref_date": result["date1"][0],
            "Second_date": result["date2"],
        }
    )

    for var in result.columns.difference(["date1", "date2"]):
        if var in ["result_dx", "result_dy", "xcount_x", "xcount_y", "error_x", "error_y", "xcount_z"]:
            data[var] = result[var].values.cumsum()
    data = data.rename(columns={"result_dx": "dx", "result_dy": "dy"})

    if second_date_list is not None:
        tmp = pd.DataFrame(
            {
                "Ref_date": pd.NaT,
                "Second_date": second_date_list,
                **{var: np.nan for var in data.columns.difference(["Ref_date", "Second_date"])},
            }
        )

        positions = np.searchsorted(second_date_list, data["Second_date"].values)
        tmp.iloc[positions] = data.values

        return tmp
    return data


def set_function_for_interpolation(
    option_interpol: str, x: np.ndarray, dataf: pd.DataFrame, result_quality: list | None
) -> (
    interpolate.interp1d | interpolate.UnivariateSpline,
    interpolate.interp1d | interpolate.UnivariateSpline,
    interpolate.interp1d | interpolate.UnivariateSpline,
    interpolate.interp1d | interpolate.UnivariateSpline,
):  # type: ignore
    """
    Get the function to interpolate the each of the time series.

    :param option_interpol: [str] --- Type of interpolation, it can be 'spline', 'spline_smooth' or 'nearest'
    :param x: [int] --- Integer corresponding to the time at which a certain displacement has been estimated
    :param dataf: [pd dataframe] --- Data to interpolate
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))

    :return fdx, fdy: [functions | None] --- The functions which need to be used to interpolate dx and dy
    :return fdx_xcount, fdx_ycount: [functions | None] --- The functions which need to be used to interpolate the contributed values in X
    """

    assert type(option_interpol) is str and option_interpol in [
        "spline_smooth",
        "spline",
        "nearest",
    ], "The filepath must be a string among the options: 'spline_smooth','spline','nearest'."

    # Compute the functions used to interpolate
    # Define the interpolation functions based on the interpolation option
    interpolation_functions = {
        "spline_smooth": lambda x, y: interpolate.UnivariateSpline(x, y, k=3),
        "spline": lambda x, y: interpolate.interp1d(x, y, kind="cubic"),
        "nearest": lambda x, y: interpolate.interp1d(x, y, kind="nearest"),
    }

    # Compute the functions used to interpolate
    interpolation_func = interpolation_functions[option_interpol]

    fdx = interpolation_func(x, dataf["dx"])
    fdy = interpolation_func(x, dataf["dy"])

    fdx_xcount, fdy_xcount, fdx_error, fdy_error = None, None, None, None
    if result_quality is not None:
        if "X_contribution" in result_quality:
            fdx_xcount = interpolation_func(x, dataf["xcount_x"])
            fdy_xcount = interpolation_func(x, dataf["xcount_y"])
        if "Error_propagation" in result_quality:
            fdx_error = interpolation_func(x, dataf["error_x"])
            fdy_error = interpolation_func(x, dataf["error_y"])

    return fdx, fdy, fdx_xcount, fdy_xcount, fdx_error, fdy_error


def full_with_nan(dataf_lp: pd.DataFrame, first_date: pd.Series, second_date: pd.Series) -> pd.DataFrame:
    """

    :param dataf_lp: [pd dataframe] --- Interpolated results
    :param first_date: [pd series] --- List of first dates of the entire cube
    :param second_date: [pd series] --- List of second dates of the entire cube

    :return dataf_lp: [pd dataframe] --- Interpolated results with row of name so when there is missing estimation in comparison with the entire cube
    """

    nul_df = pd.DataFrame(
        {
            "date1": first_date,
            "date2": second_date,
            "vx": np.full(len(first_date), np.nan),
            "vy": np.full(len(first_date), np.nan),
        }
    )

    if "xcount_x" in dataf_lp.columns:
        nul_df["xcount_x"] = np.full(len(first_date), np.nan)
        nul_df["xcount_y"] = np.full(len(first_date), np.nan)
    if "error_x" in dataf_lp.columns:
        nul_df["error_x"] = np.full(len(first_date), np.nan)
        nul_df["error_y"] = np.full(len(first_date), np.nan)
    dataf_lp = pd.concat([nul_df, dataf_lp], ignore_index=True)

    return dataf_lp


def smooth_results(result: np.ndarray, window_size: int = 3):
    r"""
    Spatially smooth the data by averaging (applying a convolution filter to) each pixel with its neighborhood.
    /!\ This method only works with cubes where both starting and ending dates exactly correspond for each pixel (ie TICOI results)

    :param result: [np array] --- Results for a variable (pandas dataframe of TICOI results transformed, as in cube_data_class.write_result_ticoi)
    :param window_size: [int] [default is 3] --- Size of the window for mean filtering

    :return result: [np array] --- Smoothened result
    """

    filt = np.full((window_size, window_size), 1 / window_size**2)

    # Filter the data at each date
    for t in range(result.shape[-1]):
        result[:, :, t] = ndi.correlate(result[:, :, t], filt, mode="nearest")

    return result


def visualisation_interpolation(
    list_dataf: pd.DataFrame,
    option_visual: List = [
        "interp_xy_overlaid",
        "interp_xy_overlaid_zoom",
        "invertvv_overlaid",
        "invertvv_overlaid_zoom",
        "direction_overlaid",
        "quality_metrics",
    ],
    save: bool = False,
    show: bool = True,
    path_save: Optional[str] = None,
    colors: List[str] = ["blueviolet", "orange"],
    figsize: tuple[int] = (10, 6),
    vminmax: List[int] = None,
):
    """
    Plot some relevant information about TICOI results.

    :param list_dataf: [pd dataframe] --- Results after the interpolation in TICOI processing
    :param option_visual: [list] [default] --- List of the plots to prepare (each plot shows a different information)
    :param save: [bool] [default is False] --- If True, save the figures to path_save (if not None)
    :param show: [bool] [default is True] --- If True, plot the figures
    :param path_save: [str | None] [default is None] --- Path where the figures must be saved if save is True
    :param colors: [List<str>] [default is ["blueviolet", "orange"]] --- Colors for the plot
    :param figsize: [tuple] [default is (10, 6)] --- Size of the figures
    :param vminmax: [List[int, int]] [default is None] --- Min and max values for the y-axis of the plots
    """

    pixel_object = PixelClass()
    pixel_object.load(
        list_dataf, save=save, show=show, path_save=path_save, type_data=["obs", "interp"], figsize=figsize
    )

    dico_visual = {
        "interp_xy_overlaid": (
            lambda pix: pix.plot_vx_vy_overlaid(type_data="interp", colors=colors, zoom_on_results=False)
        ),
        "interp_xy_overlaid_zoom": (
            lambda pix: pix.plot_vx_vy_overlaid(type_data="interp", colors=colors, zoom_on_results=True)
        ),
        "inverpvv_overlaid": (
            lambda pix: pix.plot_vv_overlaid(type_data="interp", colors=colors, zoom_on_results=False, vminmax=vminmax)
        ),
        "inverpvv": (lambda pix: pix.plot_vv(type_data="interp", color=colors[1], vminmax=vminmax)),
        "inverpvv_overlaid_zoom": (
            lambda pix: pix.plot_vv_overlaid(type_data="interp", colors=colors, zoom_on_results=True, vminmax=vminmax)
        ),
        "direction_overlaid": (lambda pix: pix.plot_direction_overlaid(type_data="interp")),
        "quality_metrics": (lambda pix: pix.plot_quality_metrics()),
    }

    for option in option_visual:
        if option in dico_visual.keys():
            dico_visual[option](pixel_object)
