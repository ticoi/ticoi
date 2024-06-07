'''
Coefficient optimization of the TICOI post-processing method, according to a "ground truth" given data cube (GPS, 
more precise satellitarian data...). A range of coefficients is tested for a given regularisation method, by computing
the RMSE between TICOI results for the tested coefficient, interpolated to the ground truth dates, and compared
to those ground truth datas computing the Root Mean Square Error (RMSE). A RMSE-coefficient curve is then plotted.
'''

import os
import time
import warnings
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from ticoi.core import inversion_core, interpolation_to_data
from ticoi.cube_data_classxr import cube_data_class


# %% ======================================================================== #
#                                FUNCTIONS                                    #
# =========================================================================%% #

def RMSE_TICOI_GT(data: list, mean: list | None, dates_range: np.ndarray | None, data_gt: pd.DataFrame, i: float | int, 
                  j: float | int, coef: int, inversion_kwargs: dict, interpolation_kwargs: dict,
                  visual: bool = False, plot_raw: bool = False, vminmax: list | None = None, savedir: str | None = None):
    
    '''
    Compute the RMSE between TICOI results with a given coefficient and "ground truth" data.
    
    :param data: [list] --- An array where each line is (date1, date2, other elements ) for which a velocity is computed (correspond to the original displacements)
    :param mean: [list | None] --- Apriori on the average
    :param dates_range: [np array | None] --- List of np.datetime64 [D], dates of the estimated displacement in X with an irregular temporal sampling (ILF)
    :param data_gt: [pd dataframe] --- "Ground truth" data to which TICOI results are compared
    :params i, j: [float | int] --- Coordinates of the point in pixel
    :param coef: [int] --- Coef of Tikhonov regularisation
    :param inversion_kwargs: [dict] --- Inversion parameters
    :param interpolation_kwargs: [dict] --- Parameters for the interpolation to GT dates (less parameters than for core interpolation)
    :param visual: [bool] [default is False] --- Plot interpolated and GT velocities
    :param plot_raw: [bool] [default is False] --- Add raw datas to the plot
    :param vminmax: [list | None] [default is None] --- Specify the vertical limits of the plot
    :param savedir: [str | None] [default is None] --- Save the figure to this location
    
    :return RMSE: Root Mean Square Error between TICOI results interpolated to "ground truth" (GT) dates, and GT datas
    '''
    
    # Proceed to inversion
    A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, mean=mean, coef=coef, **inversion_kwargs)
    if not visual or not plot_raw:
        del data
    del dates_range, mean
    
    # Proceed to interpolation
    dataf_lp = interpolation_to_data(result, data_gt, **interpolation_kwargs)
    del A, result, dataf

    # RMSE between TICOI result and ground truth data
    RMSE = sm.root_mean_squared_error(dataf_lp[['vx', 'vy']], data_gt[['vx', 'vy']])
    
    ##  Plot the interpolated velocity magnitudes along with GT velocity magnitudes
    if visual:
        data_gt = data_gt.reset_index()
        dataf_lp = dataf_lp.reset_index()
        
        # Magnitude of the velocities
        vv_gt = np.sqrt((data_gt['vx'] ** 2 + data_gt['vy'] ** 2)) # GT data
        vv_lp = np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2) # TICOI results interpolated to GT data
        # Offsets and central dates are the same as TICOI was interpolated to GT dates
        offset = data_gt['date2'] - data_gt['date1']
        central_dates = data_gt['date1'] + offset // 2
        
        fig, ax = plt.subplots(figsize=(12, 6/1.8))
        
        # Plot raw data
        if plot_raw:
            data = pd.DataFrame(data={'date1': data[0][:, 0], 'date2': data[0][:, 1],
                                      'vx': data[1][:, 0], 'vy': data[1][:, 1],
                                      'errorx': data[1][:, 2], 'errory': data[1][:, 3],
                                      'temporal_baseline': data[1][:, 4]})
            offset_raw = data['date2'] - data['date1']
            central_dates_raw = data['date1'] + offset_raw / 2
            vv_raw = np.sqrt((data['vx'] * unit / data['temporal_baseline']) ** 2 + (data['vy'] * unit / data['temporal_baseline']) ** 2)
        
            ax.plot(central_dates_raw, vv_raw, linestyle='', color='green', zorder=1, marker='o', lw=0.7, markersize=2, alpha=0.7,
                      label='Central date of velocity observations')
            ax.errorbar(central_dates_raw, vv_raw, xerr=offset_raw / 2, color='green', alpha=0.2, fmt=',', zorder=1,
                          label='Temporal baseline of velocity observations [days]')
            
        # Plot interpolated velocities
        ax.plot(central_dates, vv_lp, linestyle='', marker='o', markersize=3, color='b', 
                  label='Central date of Interpolated velocities (TICOI results)')
        ax.errorbar(central_dates, vv_lp, xerr=offset / 2, color='b', alpha=0.2, 
                      fmt=',', zorder=1, label='Temporal baseline of interpolated velocities')
        # Plot "ground truth" velocities
        ax.plot(central_dates, vv_gt, linestyle='', color='orange', zorder=1, marker='o', lw=0.7, markersize=2, alpha=0.7,
                  label='Central date of velocity observations')
        ax.errorbar(central_dates, vv_gt, xerr=offset / 2, color='orange', alpha=0.2, fmt=',', zorder=1,
                      label='Temporal baseline of velocity observations [days]')
        ax.set_ylabel('Velocity magnitude [m/y]')
        
        if vminmax is None:
            if plot_raw: ax.set_ylim(0.8*min(np.nanmin(vv_gt), np.nanmin(vv_raw)), 1.2*max(np.nanmax(vv_gt), np.nanmax(vv_raw)))
            else: ax.set_ylim(0.8*np.nanmin(vv_gt), 1.2*np.nanmax(vv_gt))
        else:
            ax.set_ylim(vminmax)
        
        ax.legend(loc='lower left', bbox_transform=fig.transFigure, fontsize=7, ncol=2)
        fig.suptitle(f'Magnitude of the velocities (ground truth and interpolated ILF) for coef={coef}', fontsize=16)
    
        if savedir is not None:
            fig.savefig(f'{savedir}interpol_vv_gt_{coef}.png')
        
        plt.show()

    del dataf_lp
    
    return RMSE


def optimize_coef(cube: cube_data_class, cube_gt: cube_data_class, i: float | int, j: float | int, obs_filt: xr.Dataset, 
                  load_pixel_kwargs: dict, inversion_kwargs: dict, interpolation_kwargs: dict,
                  cmin: int = 10, cmax: int = 1000, step: int = 10, coefs : list | None = None, 
                  stats: bool = False, **visual_options):
    
    '''
    Compute the RMSE between the velocities obtained with TICOI using Sentinel-2 data and velocities available in Pleiade data
    for different coefficients.
    
    :param cube: [cube_data_class] --- Data cube used to compute TICOI at point (i, j)
    :param cube_gt: [cube_data_class] --- Data cube of "ground truth" velocities
    :params i, j: [float | int] --- Coordinates of the point where we want to optimise the coefficient
    :param obs_filt: [xr dataset] --- Filtered dataset (e.g. rolling mean)
    :param load_pixel_kwargs: [dict] --- Pixel loading parameters
    :param inversion_kwargs: [dict] --- Inversion parameters
    :param interpolation_kwargs: [dict] --- Parameters for the interpolation to GT dates (less parameters than for core interpolation)
    :param cmin: [int] [default is 10] --- If coefs=None, start point of the range of coefs to be tested
    :param cmax: [int] [default is 1000] --- If coefs=None, stop point of the range of coefs to be tested
    :param step: [int] [default is 10] --- If coefs=None, step for the range of coefs to be tested
    :param coefs: [list | None] [default is None] --- To specify the coefficients to be tested, if None, range(cmin, cmax, step) coefs will be tested
    :param stats: [bool] [default is False] --- Compute some statistics on raw data and GT data
    :param visual_options: Additionnal options for plotting purposes during the computation of the RMSE for each coef
    
    :return: [pd dataframe] --- Dataframe with the studied coefficients ('coefs'), the resulting RMSEs ('RMSEs'), the standard deviation of similar Sentinel-2 and Pleiade data ('std'), how many of those data were used to conduct the computation ('nb_data'), their mean temporal baseline ('temporal_baseline') and their mean velocity values for both x and y components ('mean_v')
    '''
    
    # Load data at pixel
    data, mean, dates_range = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)
    dataf = pd.DataFrame(data={'date1': data[0][:, 0], 'date2': data[0][:, 1],
                              'vx': data[1][:, 0], 'vy': data[1][:, 1],
                              'errorx': data[1][:, 2], 'errory': data[1][:, 3],
                              'temporal_baseline': data[1][:, 4]})
    
    # Load ground truth pixel and convert to pd dataframe
    data_gt, _, _ = cube_gt.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)
    data_gt = pd.DataFrame(data={'date1': data_gt[0][:, 0], 'date2': data_gt[0][:, 1],
                                 'vx': data_gt[1][:, 0], 'vy': data_gt[1][:, 1],
                                 'errorx': data_gt[1][:, 2], 'errory': data_gt[1][:, 3],
                                 'temporal_baseline': data_gt[1][:, 4]})
    data_gt.index = data_gt['date1'] + (data_gt['date2'] - data_gt['date1']) // 2
    
    # Interpolation must be caried out in between the min and max date of the original data
    data_gt = data_gt[(data_gt['date1'] > dataf['date2'].min()) & (data_gt['date2'] < dataf['date2'].max())]
    
    # Must have enough data to make an interpolation
    if data_gt.shape[0] == 0 and data[0].shape[0] <= 2:
        if stats:
            return pd.DataFrame({'coefs':[], 'RMSEs':[], 'nb_data':[], 'temporal_baseline':[], 'std':[], 'mean_v':[], 'position':[]}) 
        return pd.DataFrame({'coefs':[], 'RMSEs':[]})
    
    # Coefficients to be tested
    if coefs is None:
        coefs = np.arange(cmin, cmax+1, step)
        
    # Compute RMSE for every coefficient
    RMSEs = [RMSE_TICOI_GT(data, mean, dates_range, data_gt, i, j, coef, inversion_kwargs, interpolation_kwargs, **visual_options) for coef in coefs]
    
    if stats:
        # Statistics on raw and GT data
        nb_data = (dataf.shape[0], data_gt.shape[0]) # Amount of data in each dataset
        # Mean of similar data (same acquisition dates) of raw and GT data
        mean_raw = dataf.groupby(['date1', 'date2'], as_index=False)[['vx', 'vy', 'errorx', 'errory']].mean()[['vx', 'vy']].mean()
        mean_gt = data_gt.groupby(['date1', 'date2'], as_index=False)[['vx', 'vy', 'errorx', 'errory']].mean()[['vx', 'vy']].mean()
        # Standard deviation of similar data (same acquisition dates) of raw and GT data
        std_raw = dataf.groupby(['date1', 'date2'], as_index=False)[['vx', 'vy', 'errorx', 'errory']].std(ddof=0)[['vx', 'vy']].mean()
        std_gt = data_gt.groupby(['date1', 'date2'], as_index=False)[['vx', 'vy', 'errorx', 'errory']].std(ddof=0)[['vx', 'vy']].mean()
        # Standard deviation of raw and GT data
        std_raw_all = dataf[['vx', 'vy']].std(ddof=0)
        std_gt_all = data_gt[['vx', 'vy']].std(ddof=0)
        temporal_baseline = (dataf['temporal_baseline'].mean(), data_gt['temporal_baseline'].mean()) # Average temporal baseline
        
        return pd.DataFrame({'coefs': coefs,
                             'RMSEs': RMSEs,
                             'nb_data':[nb_data for _ in range(len(coefs))],
                             'temporal_baseline':[temporal_baseline for _ in range(len(coefs))],
                             'std': [(std_raw['vx'], std_raw['vy'], std_gt['vx'], std_gt['vy']) for _ in range(len(coefs))],
                             'std_all': [(std_raw_all['vx'], std_raw_all['vy'], std_gt_all['vx'], std_gt_all['vy']) for _ in range(len(coefs))],
                             'mean_v': [(mean_raw['vx'], mean_raw['vy'], mean_gt['vx'], mean_gt['vy']) for _ in range(len(coefs))],
                             'position': [(i, j) for _ in range(len(coefs))]})
    
    return pd.DataFrame({'coefs': coefs, 'RMSEs': RMSEs})

if __name__ == '__main__':
    
    # %%===================================================================== #
    #                               PARAMETERS                                #
    # =====================================================================%% #
    
    warnings.filterwarnings('ignore')
    
    ## ---------------------------- Data selection ------------------------- ##
    # List of the paths where the data cubes are stored
    cube_names = ['nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01470_y03675_all_filt-multi.nc',]
                   # 'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc']
    cube_authors = None # Specify the author of the data cubes (list), keep None if the authors are specified in the dataset
    path_save = 'nathan/Tests_MB/' # Path where to store the results
    proj = 'EPSG:32632'  # EPSG system of the given coordinates

    i, j = 338913.8, 5081510.3 # Point (pixel) where to carry on the computation

    ## ------------------------- Main parameters --------------------------- ##
    regu = '1accelnotnull' # Regularization method to be used
    solver = 'LSMR_ini' # Solver for the inversion
    unit = 365 # 1 for m/d, 365 for m/y
    result_quality = 'X_contribution' # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')

    ## --------------------- Visualization parameters ---------------------- ##
    verbose = False # Print information throughout TICOI processing
    visual = False # Plot informations along the way
    save = False # Save the results or not
    # Visualisation options for 
    option_visual = ['original_velocity_xy', 'original_magnitude',
                     'X_magnitude_zoom', 'X_magnitude', 'X_zoom', 'X',
                     'vv_quality', 'vxvy_quality',
                     'Residu_magnitude', 'Residu',
                     'X_z', 'Y_contribution',
                     'direction']
    vmax = [False, False] # Vertical limits for the plots
    
    ## --------------------- Optimization parameters ----------------------- ##
    # Path to the "ground truth" cube used to optimize the regularisation
    cube_gt_name = 'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc' 
    # Specify the coefficients you want to test
    coefs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 450, 500, 550, 600, 700, 800, 900, 1000] # To specify the coefficients to be tested, if None, range(cmin, cmax, step) coefs will be tested
    coef_min = 10 # If coefs=None, start point of the range of coefs to be tested 
    coef_max = 1000 # If coefs=None, stop point of the range of coefs to be tested
    step = 10 # If coefs=None, step for the range of coefs to be tested
    stats = True # Compute some statistics on raw data and GT data
    # Visualization during optimization (for each coefficient) /!\ Can generate a lot of figures...
    visual_opt = False # Plot interpolated and GT velocities
    plot_raw = True # Add raw datas to the plot
    vminmax = None # Specify the vertical limits of the plot
    savedir = path_save # Save the figure to this location

    ## ------------------------ Loading parameters ------------------------- ##
    load_kwargs = {'chunks': {}, 
                   'conf': False, # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
                   'buffer': [i, j, 500], # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
                   'pick_date': ['2015-01-01', '2023-01-01'], # Select dates ([min, max] or None to select all)
                   'pick_sensor': None, # Select sensors (None to select all)
                   'pick_temp_bas': None, # Select temporal baselines ([min, max] in days or None to select all)
                   'proj': proj, # EPSG system of the given coordinates
                   'verbose': verbose} # Print information throughout the loading process 

    ## ------------------- Data preparation parameters --------------------- ##
    preData_kwargs = {'smooth_method': 'gaussian', # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
                      's_win': 3, # Size of the spatial window
                      't_win': 90, # Time window size for 'ewma' smoothing
                      'sigma': 3, # Standard deviation for 'gaussian' filter
                      'order': 3, # Order of the smoothing function
                      'unit': unit, # 365 if the unit is m/y, 1 if the unit is m/d
                      'delete_outliers': 'vvc_angle', # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle') 
                      'regu': regu, # Regularization method to be used
                      'solver': solver, # Solver for the inversion
                      'proj': proj, # EPSG system of the given coordinates
                      'velo_or_disp': 'velo', # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
                      'verbose': verbose} # Print information throughout the filtering process 
                      
    ## -------------- Parameters for the pixel loading part ---------------- ##
    load_pixel_kwargs = {'regu': regu, # Regularization method to be used
                         'solver': solver, # Solver for the inversion
                         'proj': proj, # EPSG system of the given coordinates
                         'interp': 'nearest', # Interpolation method used to load the pixel when it is not in the dataset
                         'visual': visual, # Plot results along the way
                         'verbose':verbose} # Print information throughout TICOI processing
                         
    ## ----------------------- Inversion parameters ------------------------ ##
    inversion_kwargs = {'regu': regu, # Regularization method to be used
                        'solver': solver, # Solver for the inversion
                        'conf': False, # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
                        'unit': unit, # 365 if the unit is m/y, 1 if the unit is m/d

                        'iteration': True, # Allow the inversion process to make several iterations
                        'nb_max_iteration': 10, # Maximum number of iteration during the inversion process
                        'threshold_it': 0.1, # Threshold to test the stability of the results between each iteration, used to stop the process
                        'weight': True, # If True, use apriori weights
                        'detect_temporal_decorrelation': True, # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
                        'linear_operator': None, # Perform the inversion using this specific linear operator
                        'result_quality': result_quality, # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                        
                        'visual': visual, # Plot results along the way
                        'verbose': verbose} # Print information throughout TICOI processing
                        
    ## --------------------- Interpolation parameters ---------------------- ##
    interpolation_kwargs = {'option_interpol': 'spline', # Type of interpolation ('spline', 'spline_smooth', 'nearest')                
                            'result_quality': result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                            'unit': unit} # 365 if the unit is m/y, 1 if the unit is m/d
    
    # Create a subfolder if it doesnt exist
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    
    
    # %% ======================================================================== #
    #                                DATA LOADING                                 #
    # =========================================================================%% #
    
    start = [time.time()]

    # In the first place, we load the data
    cube = cube_data_class()
    cube.load(cube_names[0], author=cube_authors[0] if type(cube_authors) == list else cube_authors, **load_kwargs)

    # Several cubes have to be merged together
    if len(cube_names) > 1:
        for n in range(1, len(cube_names)):
            cube2 = cube_data_class()
            cube2.load(cube_names[n], author=cube_authors[n] if type(cube_authors) == list else cube_authors, **load_kwargs)
            cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
            cube.merge_cube(cube2)

    # Then we load the "ground truth"
    cube_gt = cube_data_class()
    cube_gt.load(cube_gt_name, **load_kwargs)

    stop = [time.time()]
    print(f'[Data loading] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s')
    print(f'[Data loading] Data cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ')
    print(f'[Data loading] Ground Truth cube of dimension (nz,nx,ny) : ({cube_gt.nz}, {cube_gt.nx}, {cube_gt.ny})')

    start.append(time.time())
    
    # Filter the data cube (compute rolling_mean for regu=1accelnotnull)
    obs_filt = cube.filter_cube(**preData_kwargs)

    stop.append(time.time())
    print(f'[Data loading] Filtering the cube took {round((stop[1] - start[1]), 4)} s')
    
    
    # %%========================================================================= #
    #                              COMPUTE COEFS                                  #
    # =========================================================================%% #
    
    start.append(time.time())

    # Optimize the regularisation coefficient
    result = optimize_coef(cube, cube_gt, i, j, obs_filt, load_pixel_kwargs, inversion_kwargs, interpolation_kwargs,
                           cmin=coef_min, cmax=coef_max, step=step, coefs=coefs, stats=stats, visual=visual_opt, 
                           plot_raw=plot_raw, vminmax=vminmax, savedir=savedir)
    
    stop.append(time.time())
    print(f'[Coef optimization] Coefficient optimization took {round((stop[1] - start[1]), 1)} s')
    
    
    # %%========================================================================= #
    #                                 PLOT COEFS                                  #
    # =========================================================================%% #
    
    start.append(time.time())
    
    coefs = np.array(result['coefs'])
    RMSEs = result['RMSEs']
    mean_nb_data = result['nb_data'][0]
    mean_baselines = result['temporal_baseline'][0]
    mean_std = result['std'][0]
    mean_std_gt = np.sqrt(mean_std[2] ** 2 + mean_std[3] ** 2)
    mean_std_all = result['std_all'][0]
    mean_v = result['mean_v'][0]
    
    best_coef = coefs[np.argmin(RMSEs)]
    best_RMSE = np.min(RMSEs)
    good_RMSE = max(1.05 * best_RMSE, best_RMSE + mean_std_gt)
    good_coefs = coefs[RMSEs < good_RMSE]
    
    if verbose: 
        print(f'[Coef optimization] Best RMSE {best_RMSE} obtained for coef = {best_coef}')
    
    # Plot result
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(coefs[~(RMSEs < good_RMSE)], RMSEs[~(RMSEs < good_RMSE)], marker='x', markersize=7, linestyle='', markeredgecolor='darkred')
    ax.plot(coefs[RMSEs < good_RMSE], RMSEs[RMSEs < good_RMSE], marker='x', markersize=7, linestyle='', markeredgecolor='darkgreen')
    ax.plot([coefs[0], coefs[-1]], [good_RMSE, good_RMSE], linestyle='--', color='midnightblue')
    ax.set_xlabel('Regularisation coefficient value')
    if coefs is None:
        ax.set_xlim(coef_min - int(step/2), coef_max + int(step/2))
    else:
        ax.set_xlim(min(coefs), max(coefs))
    ax.set_ylabel('RMSE between TICOI results from Sentinel-2 data and Pleiades data [m/y]')
    fig.suptitle(f'RMSE between TICOI results and Pleiades data when changing the regularisation coefficient (regu={regu}) at point ({round(i, 5)}, {round(j, 5)})\nBest for coef = {best_coef} (RMSE = {best_RMSE})')
    plt.show()
    
    if save: fig.savefig(f'{path_save}RMSE_coef_{round(i,5)}_{round(j,5)}_{regu}.png')

    stop.append(time.time())
    print(f'[Overall] Overall processing took {round((stop[-1] - start[0]), 1)} s')