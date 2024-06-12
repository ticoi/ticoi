"""
Auxillary functions to process the temporal interpolation.

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import numpy as np
from scipy import interpolate
import pandas as pd
from ticoi.pixel_class import pixel_class
from typing import List, Optional, Union

def prepare_interpolation_date(cube:"ticoi.cube_data_classxr.cube_data_class")-> (np.datetime64,np.datetime64):
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
    return first_date_interpol,last_date_interpol

def reconstruct_common_ref(result: pd.DataFrame, result_quality: list | str | None = None,
                           result_dz: pd.DataFrame | None = None) -> pd.DataFrame:

    """
    Build the Cumulative Displacements (CD) time series with a Common Reference (CR) from a Leap Frog time series

    :param result: [np array] --- Leap frog displacement for x-component and y-component
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param result_dz: [pd dataframe | None] [default is None] --- Vertical displacement component

    :return data: [pd dataframe] --- Cumulative displacement time series in x and y component, pandas dataframe
    """

    if result.empty:
        return pd.DataFrame(
            {'Ref_date': [np.nan], 'date2': [np.nan],
             'dx': [np.nan], 'dy': [np.nan],'xcount_x': [np.nan], 'xcount_y': [np.nan]})

    
    # Common Reference
    data = pd.DataFrame({'Ref_date': np.full(result.shape[0], result['date1'][0]), 'date2': result['date2'],
         'dx': np.cumsum(result['result_dx']), 'dy': np.cumsum(result['result_dy'])})

    if result_quality is not None and 'X_contribution' in result_quality :
        data['xcount_x']=np.cumsum(result['xcount_x'])
        data['xcount_y']=np.cumsum(result['xcount_y'])

    if result_quality is not None and 'Error_propagation' in result_quality:
        data['error_x'] = np.cumsum(result['error_x'])
        data['error_y'] = np.cumsum(result['error_y'])

    if result_dz is not None:
        data['dz'] = np.cumsum(result['dz'])
        if result_quality is not None and 'X_contribution' in result_quality :data['xcount_z']= np.cumsum(result['xcount_z'])

    return data


def set_function_for_interpolation(option_interpol: str, x: np.ndarray, dataf: pd.DataFrame, result_quality: list | None) -> (
            interpolate.interp1d | interpolate.UnivariateSpline, interpolate.interp1d | interpolate.UnivariateSpline,
            interpolate.interp1d | interpolate.UnivariateSpline, interpolate.interp1d | interpolate.UnivariateSpline):

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
        'spline_smooth': lambda x, y: interpolate.UnivariateSpline(x, y, k=3),
        'spline': lambda x, y: interpolate.interp1d(x, y, kind='cubic'),
        'nearest': lambda x, y: interpolate.interp1d(x, y, kind='nearest')
    }

    # Compute the functions used to interpolate
    if option_interpol in interpolation_functions:
        interpolation_func = interpolation_functions[option_interpol]

        fdx = interpolation_func(x, dataf['dx'])
        fdy = interpolation_func(x, dataf['dy'])

        fdx_xcount, fdy_xcount, fdx_error, fdy_error = None, None, None,None
        if result_quality is not None:
            if 'X_contribution' in result_quality:
                fdx_xcount = interpolation_func(x, dataf['xcount_x'])
                fdy_xcount = interpolation_func(x, dataf['xcount_y'])
            if 'Error_propagation' in result_quality:
                fdx_error = interpolation_func(x, dataf['error_x'])
                fdy_error = interpolation_func(x, dataf['error_y'])

        return fdx, fdy, fdx_xcount, fdy_xcount, fdx_error,fdy_error

    return None, None, None, None  # Return default values if interpolation option is not valid

def full_with_nan(dataf_lp:pd.DataFrame,first_date:pd.Series,second_date:pd.Series)->pd.DataFrame:
    """

    :param dataf_lp: interpolated results
    :param first_date: list of first dates of the entire cube
    :param second_date: list of second dates of the entire cube
    :return: interpolated with row of name so when there is missing estimation in comparison with the entire cube
    """
    nul_df = pd.DataFrame(
        {'date1': first_date, 'date2': second_date,
         'vx': np.full(len(first_date), np.nan), 'vy': np.full(len(first_date), np.nan)})
    if 'xcount_x' in dataf_lp.columns:
        nul_df['xcount_x'] = np.full(len(first_date), np.nan)
        nul_df['xcount_y'] = np.full(len(first_date), np.nan)
    if 'error_x' in dataf_lp.columns:
        nul_df['error_x'] = np.full(len(first_date), np.nan)
        nul_df['error_y'] = np.full(len(first_date), np.nan)
    dataf_lp = pd.concat([nul_df, dataf_lp], ignore_index=True)
    return dataf_lp

def visualisation_interpolation (list_dataf: pd.DataFrame, option_visual: List=['interp_xy_overlayed','interp_xy_overlayed_zoom','invertvv_overlayed','invertvv_overlayed_zoom','direction_overlayed'], save: bool = False, show: bool = True, path_save: Optional[str] = None, colors: List[str] = ['blueviolet', 'orange'],figsize:tuple[int, int] = (10,6)):
    pixel_object = pixel_class()
    pixel_object.load(list_dataf, save=save, show=show, path_save=path_save,type_data=['obs','interp'],figsize=figsize)

    dico_visual = {'interp_xy_overlayed': pixel_object.plot_vx_vy_overlayed(type_data='interp', colors=colors, zoom_on_results=False),
                   'interp_xy_overlayed_zoom': pixel_object.plot_vx_vy_overlayed(type_data='interp',
                                                                            colors=colors,
                                                                            zoom_on_results=True),
                   'invertvv_overlayed': pixel_object.plot_vv_overlayed(type_data='interp', colors=colors, zoom_on_results=False),
                   'invertvv_overlayed_zoom': pixel_object.plot_vv_overlayed(type_data='interp', colors=colors,
                                                                        zoom_on_results=True),
                   'direction_overlayed': pixel_object.plot_direction_overlayed(type_data='interp')}
    for option in option_visual:
        dico_visual[option]


# def visualisation_interpolation(dataf_lp: pd.DataFrame, data: pd.DataFrame, path_save: str, show_temp: bool = True, unit='m/y',
#                                 vmax=None, interval_output=30, figsize=(12, 6)):
#
#     """
#     Plot some figures to analyse the results from the interpolation.
#
#     :param dataf_lp: [pd dataframe] --- Results from the inversion
#     :param data: [pd dataframe] --- Original data
#     :param path_save: [str] --- Where to save the figures
#     :param show_temp: [bool] [default is True] --- If True, show the temporal baseline on the plot
#     :param unit: [str] [default is 'm/y'] --- 'm/y' or 'm/d'
#     :param vmax: [list] [default is [False, False]] --- [min,max] where min,max correspond to the ylim of the figures
#     :param interval_output: [int] [default is 30] --- Period between two dates of the obtained RLF
#     :param figsize: [tuple] [default is (12, 6)] --- (width, height) where width and height are the size of the figures
#     """
#
#     if vmax is None: vmax = [False, False]
#     offset = (dataf_lp['date2'] - dataf_lp['date1'])
#     offset_bar = data['date2'] - data['date1']
#     date_cori = data['date2'] - offset_bar / 2
#     delta = offset_bar.dt.days
#
#     # Vizualisation of the original velocity x and y [m/an]
#     fig1, ax1 = plt.subplots(2, 1, figsize=figsize)
#     ymin = np.min(dataf_lp['vx']) - 50
#     ymax = np.max(dataf_lp['vx']) + 50
#     ax1[0].set_ylim(ymin, ymax)
#     if show_temp:
#         ax1[0].errorbar(date_cori, data["vx"], xerr=offset_bar / 2, color='orange', alpha=0.2, fmt=',', zorder=1)
#         ax1[0].errorbar(dataf_lp['date1'] + offset[0] / 2, dataf_lp['vx'], xerr=offset / 2, color='b',
#                         alpha=0.2, fmt=',', zorder=1)
#     ax1[0].plot(date_cori, data["vx"], linestyle='', zorder=1, marker='o', color='orange', markersize=3,
#                 alpha=0.3)
#     ax1[0].plot(dataf_lp['date1'] + offset[0] / 2, dataf_lp['vx'], linestyle='', marker='o', markersize=3,
#                 color='b')
#     ax1[0].set_ylabel('Vx [m/y]', fontsize=16)
#     ymin = np.min(dataf_lp['vy']) - 50
#     ymax = np.max(dataf_lp['vy']) + 50
#     ax1[1].set_ylim(ymin, ymax)
#     if show_temp:
#         ax1[1].errorbar(date_cori, data["vy"], xerr=offset_bar / 2, color='orange', alpha=0.2, fmt=',', zorder=1,
#                         label='Regular LF velocities [m/y]')
#         ax1[1].errorbar(dataf_lp['date1'] + offset[0] / 2, dataf_lp['vy'], xerr=offset / 2, color='b',
#                         alpha=0.2,
#                         fmt=',', zorder=1)
#     ax1[1].plot(date_cori, data["vy"], linestyle='', zorder=1, marker='o', color='orange', markersize=3,
#                 alpha=0.7,
#                 label='Velocity observations [m/y]')
#     ax1[1].plot(dataf_lp['date1'] + offset[0] / 2, dataf_lp['vy'], linestyle='', marker='o', markersize=3,
#                 color='b', label=f'Temporal Baselines of {interval_output} days')
#
#     ax1[1].set_ylabel('Vy [m/y]', fontsize=16)
#     # dataf_lp.plot(x='date1', y='vx', style='.')
#     plt.subplots_adjust(bottom=0.20)
#     ax1[1].legend(loc='lower left', bbox_to_anchor=(0.15, 0), bbox_transform=fig1.transFigure, fontsize=12)
#     plt.show()
#     # dataf_lp.plot(x='date1', y='vx', style='.')
#     plt.show()
#     fig1.savefig(f'{path_save}interpol_vx_vy.png')
#
#     vv = np.sqrt((data["vx"] ** 2 + data["vy"] ** 2).astype('float'))
#
#     fig1, ax1 = plt.subplots(figsize=figsize)
#     ax1.plot(dataf_lp['date1'] + offset[0] / 2, np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2),
#              linestyle='',
#              marker='o', markersize=3, color='b')
#     if show_temp:
#         ax1.errorbar(dataf_lp['date1'] + offset[0] / 2, np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2),
#                      xerr=offset / 2, color='b', alpha=0.2, fmt=',', zorder=1)
#     ax1.plot(date_cori, vv, linestyle='', color='orange', zorder=1, marker='o', lw=0.7, markersize=2, alpha=0.7)
#     ax1.errorbar(date_cori, vv, xerr=offset_bar / 2, color='orange', alpha=0.2, fmt=',', zorder=1)
#     plt.show()
#     fig1.savefig(f'{path_save}interpol_vv')
#
#     fig1, ax1 = plt.subplots(figsize=figsize)
#     if vmax == [False, False]:
#         ymin = np.min(np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2))
#         ymax = np.max(np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2))
#     else:
#         ymin = vmax[0]
#         ymax = vmax[1]
#     ax1.set_ylim(ymin, ymax)
#     ax1.plot(dataf_lp['date1'] + offset[0] / 2, np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2),
#              linestyle='',
#              marker='o', markersize=3, color='b', label=f'Temporal baseline of Regular LF velocity[{unit}]')
#     if show_temp: ax1.errorbar(dataf_lp['date1'] + offset[0] / 2,
#                                np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2),
#                                xerr=offset / 2, color='b', fmt=',', zorder=1,
#                                label=f'Central date of Regular LF velocity[{unit}]')
#     ax1.plot(date_cori, vv, linestyle='', color='orange', zorder=1, marker='o', lw=0.7, markersize=2, alpha=0.7,
#              label=f'Central date of velocity observations [{unit}]')
#     if show_temp: ax1.errorbar(date_cori, vv, xerr=offset_bar / 2, color='orange', alpha=0.2, fmt=',', zorder=1,
#                                label=f'Temporal baseline of velocity observations [{unit}]')
#     plt.subplots_adjust(bottom=0.2)
#     ax1.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12, ncol=2)
#     plt.show()
#     fig1.savefig(f'{path_save}interpol_vv_zoom.png')
#
#     fig1, ax1 = plt.subplots(2, 1, figsize=figsize)
#     ymin = np.min(data["vx"])
#     ymax = np.max(data["vx"])
#     ax1[0].set_ylim(ymin, ymax)
#     if show_temp:
#         ax1[0].errorbar(date_cori, data["vx"], xerr=offset_bar / 2, color='orange', alpha=0.2, fmt=',', zorder=1)
#         ax1[0].errorbar(dataf_lp['date1'] + offset[0] / 2, dataf_lp['vx'], xerr=offset / 2, color='b',
#                         alpha=0.2,
#                         fmt=',', zorder=1)
#     ax1[0].plot(date_cori, data["vx"], linestyle='', zorder=1, marker='o', color='orange', markersize=3,
#                 alpha=0.7)
#     ax1[0].plot(dataf_lp['date1'] + offset[0] / 2, dataf_lp['vx'], linestyle='', marker='o', markersize=3,
#                 color='b')
#     ax1[0].set_ylabel('Vx [m/y]', fontsize=16)
#     ymin = np.min(data["vy"])
#     ymax = np.max(data["vy"])
#     ax1[1].set_ylim(ymin, ymax)
#     if show_temp:
#         ax1[1].errorbar(date_cori, data["vy"], xerr=offset_bar / 2, color='orange', alpha=0.2, fmt=',', zorder=1,
#                         label='Regular LF velocities [m/y]')
#         ax1[1].errorbar(dataf_lp['date1'] + offset[0] / 2, dataf_lp['vy'], xerr=offset / 2, color='b',
#                         alpha=0.2,
#                         fmt=',', zorder=1)
#     ax1[1].plot(date_cori, data["vy"], linestyle='', zorder=1, marker='o', color='orange', markersize=3,
#                 alpha=0.7,
#                 label='Velocity observations [m/y]')
#
#     ax1[1].plot(dataf_lp['date1'] + offset[0] / 2, dataf_lp['vy'], linestyle='', marker='o', markersize=3,
#                 color='b',
#                 label=f'Temporal Baselines of {interval_output} days')
#     ax1[1].set_ylabel('Vy [m/y]', fontsize=16)
#     plt.subplots_adjust(bottom=0.20)
#     ax1[1].legend(loc='lower left', bbox_to_anchor=(0.15, 0), bbox_transform=fig1.transFigure, fontsize=12)
#     plt.show()
#     fig1.savefig(f'{path_save}interpol_vy_vx_zoom')
#
#     # Compute the averaged direction, and the directions of the observations and the results
#     directionr = np.arctan2(dataf_lp['vy'], dataf_lp['vx'])
#     directionr[directionr < 0] += 2 * np.pi
#     directionm = np.arctan2(data["vy"].astype('float32'), data["vx"].astype('float32'))
#     directionm[directionm < 0] += 2 * np.pi
#     directionm_mean = np.arctan2(np.mean(data["vy"]), np.mean(data["vx"]))
#     if directionm_mean < 0: directionm_mean += 2 * np.pi
#
#     # Convert to degrees
#     directionr *= 360 / (2 * np.pi)
#     directionm *= 360 / (2 * np.pi)
#     directionm_mean *= 360 / (2 * np.pi)
#
#     fig1, ax1 = plt.subplots(figsize=(12, 6))
#     ax1.plot(dataf_lp['date1'] + offset[0] / 2, directionr, linestyle='', marker='o', markersize=3,
#              color='b',
#              label='Direction of the RLF velocities')
#     ax1.plot(date_cori, directionm, linestyle='', color='orange', zorder=1, marker='o', lw=0.7, markersize=2,
#              alpha=0.7,
#              label='Direction of the observed velocities')
#     ax1.hlines(directionm_mean, np.min(dataf_lp['date1'] + offset[0] / 2),
#                np.max(dataf_lp['date1'] + offset[0] / 2),
#                label='Mean direction of the observed velocities')
#     ax1.set_ylim(0, 360)
#     ax1.set_ylabel('Direction [°]')
#     ax1.set_xlabel('Central Dates')
#     ax1.legend(loc='lower left', bbox_to_anchor=(0.15, 0), bbox_transform=fig1.transFigure, ncol=3, fontsize=9)
#     fig1.suptitle('Direction of the velocity vectors (observations and RLF)', fontsize=20)
#     plt.show()
#     fig1.savefig(f'{path_save}direction_vv')
