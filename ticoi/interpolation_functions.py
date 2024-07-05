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
import scipy.ndimage as ndi
from scipy import interpolate
from numba import jit
from intervaltree import IntervalTree
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
    result: pd.DataFrame, result_quality: list | str | None = None, drop_nan: bool = True, result_dz: pd.DataFrame | None = None
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
    
    if drop_nan:
        data = data[~data["dx"].isna()]

    return data

def optimize_result_assignment(result, date_range):
    result2 = pd.DataFrame(
        {
            "date1": date_range[:-1],
            "date2": date_range[1:],
            **{col: None for col in result.columns.difference(['date1', 'date2'])},
            "covered": False,
        }
    )
    result["covered"] = False
    merged_result = pd.merge(result, result2[['date1', 'date2']], on=['date1', 'date2'], how='left', indicator=True)
    merged_result2 = pd.merge(result, result2[['date1', 'date2']], on=['date1', 'date2'], how='right', indicator=True)
    result1_cover = result[merged_result['_merge'] == 'left_only']
    result2_cover = result2[merged_result2['_merge'] == 'right_only']
    merged_result = merged_result[merged_result['_merge'] == 'both'].drop(columns=['_merge'])

    for _, row1 in result1_cover.iterrows():
        mask = (result2_cover['date1'] >= row1['date1']) & (result2_cover['date2'] <= row1['date2'])
        columns_to_update = result2_cover.columns.difference(['date1', 'date2'])
        if mask.any():
            last_index = result2_cover.index[mask][-1]
            result2_cover.loc[last_index, columns_to_update] = row1[columns_to_update]
            result2_cover.loc[mask, "covered"] = True
    merged_result["covered"] = False        
    result1 = pd.concat([merged_result, result2_cover], ignore_index=True).sort_values(by='date1', ignore_index=True)
    
    return result1

def reconstruct_common_ref_new(
    result: pd.DataFrame,
    result_quality: list | str | None = None,
    second_date_list: List[np.datetime64] | None = None,
    result_dz: pd.DataFrame | None = None,
) -> pd.DataFrame:

    """
    Build the Cumulative Displacements (CD) time series with a Common Reference (CR) from a Leap Frog time series

    :param result: [np array] --- Leap frog displacement for x-component and y-component
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param result_dz: [pd dataframe | None] [default is None] --- Vertical displacement component

    :return data: [pd dataframe] --- Cumulative displacement time series in x and y component, pandas dataframe
    """

    if result.empty:
        length = 1 if second_date_list is None else len(second_date_list)
        nan_list = np.full(length, np.nan)
        second_dates = [np.nan] if second_date_list is None else second_date_list
        return pd.DataFrame({
            "Ref_date": nan_list,
            "Second_date": second_dates,
            "dx": nan_list,
            "dy": nan_list,
            "xcount_x": nan_list,
            "xcount_y": nan_list,
        })

    # Common Reference
    data = pd.DataFrame(
        {
            "Ref_date": result["date1"][0],
            "Second_date": result["date2"],
        }
    )
    
    # # result_arr = {var: np.cumsum(result[var]) for var in result.columns.difference(["date1", "date2"])}
    for var in result.columns.difference(["date1", "date2"]):
        data[var] = result[var].values.cumsum()
    data = data.rename(columns={"result_dx": "dx", "result_dy": "dy"})    
    # if result_quality is not None:
    #     if "X_contribution" in result_quality:
    #         data["xcount_x"] = result["xcount_x"].values.cumsum()
    #         data["xcount_y"] = result["xcount_y"].values.cumsum()

    #     if "Error_propagation" in result_quality:
    #         data["error_x"] = result["error_x"].values.cumsum()
    #         data["error_y"] = result["error_y"].values.cumsum()

    # if result_dz is not None:
    #     data["dz"] = result["dz"].cumsum()
    #     if "X_contribution" in result_quality:
    #         data["xcount_z"] = result["xcount_z"].values.cumsum()
    if second_date_list is not None:
        tmp = pd.DataFrame(
                {
                    "Ref_date": pd.NaT,
                    "Second_date": second_date_list,
                    **{var: np.nan for var in data.columns.difference(["Ref_date", "Second_date"])}
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


def smooth_results(result: np.ndarray, window_size: int = 3, filt: np.ndarray | None = None):

    r"""
    Spatially smooth the data by averaging (applying a convolution filter to) each pixel with its neighborhood.
    /!\ This method only works with cubes where both starting and ending dates exactly correspond for each pixel (ie TICOI results)

    :param result: [np array] --- Results for a variable (pandas dataframe of TICOI results transformed, as in cube_data_class.write_result_ticoi)
    :param window_size: [int] [default is 3] --- Size of the window for mean filtering
    :param filt: [np array | None] [default is None] --- Customized filter to apply on the data (ex: Gaussian filter)

    :return result: [np array] --- Smoothened result
    """

    if filt is None:  # Apply a mean filter
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
    ],
    save: bool = False,
    show: bool = True,
    path_save: Optional[str] = None,
    colors: List[str] = ["blueviolet", "orange"],
    figsize: tuple[int, int] = (10, 6),
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
        if option in dico_visual.keys():
            dico_visual[option](pixel_object)
