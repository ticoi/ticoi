"""
Auxiliary functions to process the temporal interpolation.

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import interpolate

from ticoi.pixel_class import pixel_class


def prepare_interpolation_date(
    cube: "ticoi.cube_data_classxr.cube_data_class",
) -> (np.datetime64, np.datetime64):  # type: ignore

    """
    Define the first and last date required for the interpolation, as the first date and last in the observations.
    The purpose is to have homogenized results

    :param cube: dataset

    :return: first and last date required for the interpolation
    """

    # Prepare interpolation dates
    cube_date1 = cube.date1_().tolist()
    cube_date1.remove(np.min(cube_date1))
    first_date_interpol = np.min(cube_date1)
    last_date_interpol = np.max(cube.date2_())

    return first_date_interpol, last_date_interpol


def reconstruct_common_ref(
    result: pd.DataFrame, result_quality: list | str | None = None, result_dz: pd.DataFrame | None = None
) -> pd.DataFrame:

    """
    Build the Cumulative Displacements (CD) time series with a Common Reference (CR) from a Leap Frog time series

    :param result: [np array] --- Leap frog displacement for x-component and y-component
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param result_dz: [pd dataframe | None] [default is None] --- Vertical displacement component

    :return data: [pd dataframe] --- Cumulative displacement time series in x and y component, pandas dataframe
    """

    if result.empty:
        return pd.DataFrame(
            {
                "Ref_date": [np.nan],
                "Second_date": [np.nan],
                "dx": [np.nan],
                "dy": [np.nan],
                "xcount_x": [np.nan],
                "xcount_y": [np.nan],
            }
        )

    # Common Reference
    data = pd.DataFrame(
        {
            "Ref_date": np.full(result.shape[0], result["date1"][0]),
            "Second_date": result["date2"],
            "dx": np.cumsum(result["result_dx"]),
            "dy": np.cumsum(result["result_dy"]),
        }
    )

    if result_quality is not None and "X_contribution" in result_quality:
        data["xcount_x"] = np.cumsum(result["xcount_x"])
        data["xcount_y"] = np.cumsum(result["xcount_y"])

    if result_quality is not None and "Error_propagation" in result_quality:
        data["error_x"] = np.cumsum(result["error_x"])
        data["error_y"] = np.cumsum(result["error_y"])

    if result_dz is not None:
        data["dz"] = np.cumsum(result["dz"])
        if result_quality is not None and "X_contribution" in result_quality:
            data["xcount_z"] = np.cumsum(result["xcount_z"])

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

    # Compute the functions used to interpolate
    # Define the interpolation functions based on the interpolation option
    interpolation_functions = {
        "spline_smooth": lambda x, y: interpolate.UnivariateSpline(x, y, k=3),
        "spline": lambda x, y: interpolate.interp1d(x, y, kind="cubic"),
        "nearest": lambda x, y: interpolate.interp1d(x, y, kind="nearest"),
    }

    # Compute the functions used to interpolate
    if option_interpol in interpolation_functions:
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

    return None, None, None, None  # Return default values if interpolation option is not valid


def full_with_nan(dataf_lp: pd.DataFrame, first_date: pd.Series, second_date: pd.Series) -> pd.DataFrame:

    """

    :param dataf_lp: interpolated results
    :param first_date: list of first dates of the entire cube
    :param second_date: list of second dates of the entire cube

    :return: interpolated with row of name so when there is missing estimation in comparison with the entire cube
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


def visualisation_interpolation(
    list_dataf: pd.DataFrame,
    option_visual: List = [
        "interp_xy_overlaid",
        "interp_xy_overlaid_zoom",
        "invertvv_overlaid",
        "invertvv_overlaid_zoom",
        "direction_overlaid",
    ],
    save: bool = False,
    show: bool = True,
    path_save: Optional[str] = None,
    colors: List[str] = ["blueviolet", "orange"],
    figsize: tuple[int, int] = (10, 6),
):

    """

    :param list_dataf:
    :param option_visual:
    :param save:
    :param show:
    :param path_save:
    :param colors:
    :param figsize:

    :return:
    """

    pixel_object = pixel_class()
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
        "invertvv_overlaid": (
            lambda pix: pix.plot_vv_overlaid(type_data="interp", colors=colors, zoom_on_results=False)
        ),
        "invertvv_overlaid_zoom": (
            lambda pix: pix.plot_vv_overlaid(type_data="interp", colors=colors, zoom_on_results=True)
        ),
        "direction_overlaid": (lambda pix: pix.plot_direction_overlaid(type_data="interp")),
    }

    for option in option_visual:
        dico_visual[option](pixel_object)
