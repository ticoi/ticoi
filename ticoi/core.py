'''
Main functions to process the temporal inversion of glacier's surface velocity using the TICOI method. The inversion is solved using an Iterative Reweighted Least Square, and a robust downweighted function (Tukey's biweight).
    - mu_regularisation: Build the regularisation matrix
    - weight_for_inversion: Initialisation of the weights used in the IRLS approach
    - inversion_iteration: Compute an iteration of the inversion (weights are updated using the residuals)
    - inversion: Main function to be called, makes the temporal inversion with an IRLS approach using a given solver, it returns leap frog velocities (velcoties between consecutive dates) with an irregular temporal sampling.
    - interpolation_post: Interpolate Irregular Leap Frog time series (result of an inversion) to Regular LF time series using Cumulative Displacement times series.
    - process: Launch the entire process, data loading, inversion and interpolation
    - visualisation: Different figures can be shown in this function, according to what the user wants

Author : Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import time
import itertools
import asyncio
import warnings
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
import scipy.sparse as sp
import pandas as pd

from tqdm import tqdm
from scipy import stats
from typing import Union
from joblib import Parallel, delayed

from ticoi.cube_data_classxr import cube_data_class
from ticoi.inversion_functions import construction_a_lf, class_linear_operator, find_date_obs, inversion_one_component, \
    inversion_two_components, TukeyBiweight, weight_for_inversion, mu_regularisation, construction_dates_range_np
from ticoi.interpolation_functions import reconstruct_common_ref, set_function_for_interpolation, \
    visualisation_interpolation

warnings.filterwarnings("ignore")


# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #

def inversion_iteration(data: np.ndarray, A: np.ndarray, dates_range: np.ndarray, solver: str, coef: int,
                        Weight: np.ndarray, result_dx: np.ndarray, result_dy: np.ndarray, mu: np.ndarray,
                        regu: int | str = 1, accel: np.ndarray | None = None, linear_operator=Union["class_linear_operator", None],
                        result_quality: list | str | None = None, ini: np.ndarray | None = None, verbose: bool = False) -> (
                        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None):
    
    '''
    Compute an iteration of the inversion : update the weights using the weights from the previous iteration and the studentized residual, update the results in consequence
    and compute the residu's norm if required.

    :param data: [np array] --- Data at a given point
    :param A: [np array] --- Design matrix linking X (vector containing the velocity observations) to Y
    :param dates_range: [list] --- Dates of the displacements in X
    :param solver: [str] --- Solver of the inversion: 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
    :param coef: [int] --- Coef of Tikhonov regularisation
    :param Weight: [np array] --- Weight to give to the inversion
    :param result_dx: [np array] --- Estimated time series vx at the given iteration
    :param result_dy: [np array] --- Estimated time series vx at the given iteration
    :param mu: [np array] --- Regularization matrix
    :param regu: [int | str] [default is 1] --- Type of regularization
    :param accel: [np array | None] [default is None] --- Apriori on the acceleration
    :param linear_operator: [bool] [default is False] --- If linear operator, the inversion is performed using a linear operator (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)
    :param result_quality: [list | str | None] [default is None] --- Which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param ini: [np array | None]
    :param verbose: [bool] [default is False] --- Print informations along the way

    :return result_dx, result_dy: [np arrays] --- Obtained results (velocities) for this iteration along x and y axis
    :return weightx, weighty: [np arrays] --- Newly computed weights along x and y axis
    :return residu_normx, residu_normy: [np arrays | None] --- Norm of the residu along x and y axis (when showing the L curve)
    '''

    def compute_residual(A: np.ndarray, v: np.ndarray, X: np.ndarray) -> np.ndarray:
        Residu = (v - A.dot(X))
        return Residu

    def weightf(residu: np.ndarray, Weight: np.ndarray) -> np.ndarray:
        
        '''
        Compute weight according to the residual
        
        :param residu: [np array] Residual vector
        :param Weight: [np array | None] Apriori weight
        
        :return weight: [np array] Weight for the inversion
        '''
        
        r_std = residu / (stats.median_abs_deviation(residu) / 0.6745)
        if Weight is not None:  # The weight is a combination of apriori weight and the studentized residual
            weight = TukeyBiweight((Weight * r_std), 4.685)
        else:
            weight = TukeyBiweight((r_std), 4.685)

        return weight

    weightx = weightf(compute_residual(A, data[:, 0], result_dx), Weight[0])
    weighty = weightf(compute_residual(A, data[:, 1], result_dy), Weight[1])

    if A.shape[0] < A.shape[1]:
        if verbose: print(
            f'If the number of row is lower than the number of colomns, the results are not updated {A.shape}')
        return result_dx, result_dy, weightx, weighty, None, None

    if regu == 'directionxy':
        if solver == 'LSMR_ini':
            result_dx, result_dy, residu_normx, residu_normy = inversion_two_components(A, dates_range, 0, data, solver,
                                                                    np.concatenate([weightx, weighty]), mu, coef=coef,
                                                                    ini=np.concatenate([result_dx, result_dy]))
        else:
            result_dx, result_dy, residu_normx, residu_normy = inversion_two_components(A, dates_range, 0, data, solver,
                                                                    np.concatenate([weightx, weighty]), mu, coef=coef)
            
    elif solver == 'LSMR_ini':
        if ini == None:  # Initialization with the result from the previous inversion
            result_dx, residu_normx = inversion_one_component(A, dates_range, 0, data, solver, weightx, mu, coef=coef,
                                                              ini=result_dx, result_quality=result_quality, regu=regu,
                                                              accel=accel, linear_operator=linear_operator)
            result_dy, residu_normy = inversion_one_component(A, dates_range, 1, data, solver, weighty, mu, coef=coef,
                                                              ini=result_dy, result_quality=result_quality, regu=regu,
                                                              accel=accel, linear_operator=linear_operator)
        else:  # Initialization with the list ini, which can be a moving average
            result_dx, residu_normx = inversion_one_component(A, dates_range, 0, data, solver, weightx, mu, coef=coef,
                                                              ini=ini[0], result_quality=result_quality, regu=regu,
                                                              accel=accel, linear_operator=linear_operator)
            result_dy, residu_normy = inversion_one_component(A, dates_range, 1, data, solver, weighty, mu, coef=coef,
                                                              ini=ini[1], result_quality=result_quality, regu=regu,
                                                              accel=accel, linear_operator=linear_operator)
            
    else:  # No initialization
        result_dx, residu_normx = inversion_one_component(A, dates_range, 0, data, solver, weightx, mu, coef=coef,
                                                          result_quality=result_quality, regu=regu, accel=accel,
                                                          linear_operator=linear_operator)
        result_dy, residu_normy = inversion_one_component(A, dates_range, 1, data, solver, weighty, mu, coef=coef,
                                                          result_quality=result_quality, regu=regu, accel=accel,
                                                          linear_operator=linear_operator)

    return result_dx, result_dy, weightx, weighty, residu_normx, residu_normy


def inversion_core(data: list, i: float | int, j: float | int, dates_range: np.ndarray | None = None, solver: str = 'LSMR', 
                   regu: int | str = 1, coef: int = 100, weight: bool = False, iteration: bool = True, threshold_it: float = 0.1, 
                   unit: int = 365, conf: bool = False,  mean: list | None = None, detect_temporal_decorrelation: bool = True, 
                   linear_operator: bool = False, result_quality: list | str | None = None, nb_max_iteration: int = 10,
                   visual: bool = True, verbose: bool = False) -> (np.ndarray, pd.DataFrame, pd.DataFrame):
    
    """
    Computes A in AX = Y and does the inversion using a given solver.

    :param data: [list] --- An array where each line is (date1, date2, other elements ) for which a velocity is computed (correspond to the original displacements)
    :params i, j: [float | int] --- Coordinates of the point in pixel
    :param dates_range: [np array | None] [default is None] --- List of np.datetime64 [D], dates of the estimated displacement in X with an irregular temporal sampling (ILF)
    :param solver: [str] [default is 'LSMR'] --- Solver of the inversion: 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
    :param regu: [int | str] [default is 1] --- Type of regularization
    :param coef: [int] [default is 100] --- Coef of Tikhonov regularisation
    :param weight: [bool] [default is False] --- If True use of aprori weight
    :param iteration: [bool] [default is True] --- If True, use of iterations
    :param threshold_it: [float] [default is 0.1] --- Threshold to test the stability of the results between each iteration, use to stop the process
    :param unit: [int] [default is 365] --- 1 for m/d, 365 for m/y
    :param conf: [bool] [default is False] --- If True means that the error corresponds to confidence intervals between 0 and 1, otherwise it corresponds to errors in m/y or m/d
    :param mean: [list | None] [default is None] --- Apriori on the average
    :param detect_temporal_decorrelation: [bool] [default is True] --- If True the first inversion is solved using only velocity observations with small temporal baselines, to detect temporal decorelation
    :param linear_operator: [bool] [default is False] --- If linear operator, the inversion is performed using a linear operator (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param nb_max_iteration: [int] [default is 10] --- Maximum number of iterations
    :param visual: [bool] [default is True] --- Keep the weights for future plots
    :param verbose: [bool] [default is False] --- Print informations along the way

    :return A: [np array | None] --- Design matrix in AX = Y
    :return result: [pd dataframe | None] --- DF with dates, computed displacements and number of observations used to compute each displacement
    :return dataf: [pd dataframe | None] --- Complete DF with dates, velocities, errors, residus, weights, xcount, normr... for further visual purposes (directly depends on param visual and result_quality)
    """

    if data[0].size:  # If there are available data on this pixel
    
        # Split the data, which one dtype per array
        if len(data) == 3:
            data_dates, data_values, data_str = data
        else:
            data_dates, data_values = data

        if dates_range is None: dates_range = construction_dates_range_np(
            data_dates)  # 652 µs ± 3.24 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        ####  Build A (design matrix in AX = Y)
        if not linear_operator:
            A = construction_a_lf(data_dates,
                                  dates_range)  # 1.93 ms ± 219 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
        else:  # use a linear operator to solve the inversion
            linear_operator = class_linear_operator()
            linear_operator.load(find_date_obs(data_dates[:, :2], dates_range), dates_range,
                                 coef)  # load parameter of the linear operator
            A = sp.linalg.LinearOperator((data_values.shape[0], len(dates_range) - 1),
                                         matvec=linear_operator.matvec,
                                         rmatvec=linear_operator.rmatvec)  # build A
            mu = None

        # Set a weight of 0, for large temporal baseline in the first inversion
        # 115 µs ± 1.2 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
        aprio_weight = np.where(data_values[:, 4] > 200, 0, 1) if detect_temporal_decorrelation else None
        # First weight of the inversion
        Weightx = weight_for_inversion(weight, conf, data_values, 2, inside_Tukey=False, apriori_weight=aprio_weight)
        Weighty = weight_for_inversion(weight, conf, data_values, 3, inside_Tukey=False, apriori_weight=aprio_weight)
        del aprio_weight
        if not visual: data_values = np.delete(data_values, [2, 3],
                                               1)  # Delete quality indicator, which are not needed anymore
        # Compute regularisation matrix
        # 493 µs ± 2.35 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
        if not linear_operator:
            if regu == 'directionxy':
                # Constrain according to the vectorial product, the magnitude of the vector corresponds to mean2, the magnitude of a rolling mean
                mu = mu_regularisation(regu, A, dates_range, ini=mean)
            else:
                mu = mu_regularisation(regu, A, dates_range, ini=mean)

        #### Initialisation (depending on apriori and solver)
        # # Apriori on acceleration (following)
        if regu == '1accelnotnull':
            accel = [np.diff(mean[0]), np.diff(
                mean[1])]  # compute acceleration based on the moving average, computing using a given kernel
            mean_ini = [np.multiply(mean[i], np.diff(dates_range) / np.timedelta64(1, 'D')) for i in
                        range(
                            len(mean))]  # compute what should be the displacement in X according to the moving average, computing using a given kernel

        elif mean is not None and solver == 'LSMR_ini':  # initilization is set according the average of the whole time series
            mean_ini = [np.multiply(mean[i], np.diff(dates_range) / np.timedelta64(1, 'D') / unit) for i in
                        range(len(mean))]
            accel = None
        else:
            mean_ini = None
            accel = None

        #### Inversion
        # 87.5 ms ± 668 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        if regu == 'directionxy':
            result_dx, result_dy, residu_normx, residu_normy = inversion_two_components(A, dates_range, 0, data_values,
                                                                                        solver,
                                                                                        np.concatenate(
                                                                                            [Weightx, Weighty]),
                                                                                        mu, coef=coef,
                                                                                        ini=mean_ini)
        else:
            result_dx, residu_normx = inversion_one_component(A, dates_range, 0, data_values, solver, Weightx, mu,
                                                              coef=coef,
                                                              ini=mean_ini, result_quality=None,
                                                              regu=regu,
                                                              linear_operator=linear_operator, accel=accel)
            result_dy, residu_normy = inversion_one_component(A, dates_range, 1, data_values, solver, Weighty, mu,
                                                              coef=coef,
                                                              ini=mean_ini, result_quality=None,
                                                              regu=regu,
                                                              linear_operator=linear_operator, accel=accel)

        if not visual: del Weighty, Weightx

        if regu == 'directionxy':
            mu = mu_regularisation(regu, A, dates_range, ini=[mean[0], mean[1], result_dx, result_dy])
            # coef = coef * 1000

        # Second Iteration
        # 1.11 s ± 17.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        if iteration:
            result_dx_i, result_dy_i, weight_2x, weight_2y, residu_normx, residu_normy = inversion_iteration(
                data_values, A,
                dates_range,
                solver, coef,
                [None, None],
                result_dx,
                result_dy, mu=mu,
                verbose=verbose,
                regu=regu,
                linear_operator=linear_operator, ini=None, accel=accel,result_quality=result_quality)
            # print('nb_max_iteration',nb_max_iteration)
            # Continue to iterate until the difference between two results is lower than threshold_it or the number of iteration larger than 10
            # 6 sec
            i = 2
            while (np.mean(abs(result_dx_i - result_dx)) > threshold_it or np.mean(
                    abs(result_dy_i - result_dy)) > threshold_it) and i < nb_max_iteration:
                result_dx = result_dx_i
                result_dy = result_dy_i
                result_dx_i, result_dy_i, weight_ix, weight_iy, residu_normx, residu_normy = inversion_iteration(
                    data_values, A,
                    dates_range,
                    solver, coef,
                    [None, None],
                    result_dx,
                    result_dy, mu,
                    verbose=verbose, regu=regu,
                    linear_operator=linear_operator, ini=None, accel=accel,result_quality=result_quality)

                i += 1

                if verbose: print(i, 'dx', np.mean(abs(result_dx_i - result_dx)), 'dy',
                                  np.mean(abs(result_dy_i - result_dy)))

            if verbose:
                print('end loop', i, np.mean(abs(result_dy_i - result_dy)))
                print('nb iteration', i)

            if i == 2:
                weight_iy = weight_2y
                weight_ix = weight_2x

            del result_dx, result_dy
            if not visual: del data_values, data_dates

        else:  # If not iteration
            result_dy_i = result_dy
            result_dx_i = result_dx

        if np.isnan(result_dx_i).all():  # no results
            return None, None, None

        # compute the number of observations which have contributed to each estimated displacement
        if result_quality is not None and 'X_contribution' in result_quality:
            xcount_x = A.T.dot(weight_ix)
            xcount_y = A.T.dot(weight_iy)
        else:
            xcount_x = xcount_y = np.ones(result_dx_i.shape[0])

        # propagate the error
        # TODO terminate propgation of errors
        if result_quality is not None and 'Error_propagation' in result_quality:
            def Prop_weight(weight, Residu):
                W = np.diag(weight_ix.astype('float32'))
                FTWF = F.T * W @ F
                N = np.linalg.inv(FTWF + coef * mu.T @ mu)
                Prop_weight = N @ F.T @ W @ F @ N
                sigma0_weight = np.sum(Residu ** 2 * weight) / (F.shape[0] - F.shape[1] + 1)
                prop_wieght_diag = np.diag(Prop_weight)
                return prop_wieght_diag, sigma0_weight

            # if not 'GCV' in result_quality:
            F = sp.csc_matrix(A, dtype='float32')
            Residux = (data_values[:, 0] - F @ result_dx_i)  # has a normal distribution
            prop_wieght_diagx, sigma0_weightx = Prop_weight(weight_ix, Residux)
            Residuy = (data_values[:, 1] - F @ result_dy_i)  # has a normal distribution
            prop_wieght_diagy, sigma0_weighty = Prop_weight(weight_iy, Residuy)
            # error = np.identity(data_values.shape[0])*data_values[:,2].astype('float32')/data_values[:, -1]
            # Prop_error = N @ F.T @ W @ error @ F @ N
            #
            # t=sigma0 * np.linalg.inv(F.T @ error @ F)
            # sigma02 = np.sum(Residux ** 2) / (F.shape[0] - F.shape[1] + 1)
            #
            #
            # t=sigma0 * np.linalg.inv(A.T @  np.identity(data_values.shape[0])*weight_ix @ A)
            #
            # sigma0 = np.sum(Residux ** 2) / (F.shape[0] - F.shape[1]+1)
            # t=sigma0 * np.linalg.inv(A.T @ A)

        # If visual, save the velocity observation, the errors, the initial weights (weightini), the last weights (weightlast), the residuals from the last inversion, the sensors, and the authors
        if visual:
            vx = data_values[:, 0] / data_values[:, -1] * unit
            vy = data_values[:, 1] / data_values[:, -1] * unit
            Residux = (data_values[:, 0] - A.dot(result_dx_i))
            Residuy = (data_values[:, 1] - A.dot(result_dy_i))
            dataf = pd.DataFrame(
                {'date1': data_dates[:, 0], 'date2': data_dates[:, 1], 'vx': vx, 'vy': vy, 'errorx': data_values[:, 2],
                 'errory': data_values[:, 3], 'weightinix': Weightx,
                 'weightiniy': Weighty, 'weightlastx': weight_ix,
                 'weightlasty': weight_iy, 'residux': Residux,
                 'residuy': Residuy,
                 'sensor': data_str[:, 0],
                 'author': data_str[:, 1]})
            if residu_normx is not None:  # save the L2-norm from the last inversion, of the term AXY and the regularization term for the x- and y-component
                NormR = np.zeros(data_values.shape[0])
                NormR[:4] = np.hstack([residu_normx,
                                       residu_normy])  # the order is: AXY and regularization term L2-norm for x-component, and AXY and regularization term L2-norm for y-component
                dataf['NormR'] = NormR
                del NormR
        else:
            dataf, A = None, None

    else:  # If there is no data over this pixel
        if verbose: print(f'NO DATA TO INVERSE AT POINT {i, j}')
        return None, None, None

    # pandas dataframe with the saved results
    result = pd.DataFrame({
        'date1': dates_range[:-1],
        'date2': dates_range[1:],
        'result_dx': result_dx_i,
        'result_dy': result_dy_i,
        'xcount_x': xcount_x,
        'xcount_y': xcount_y
    })
    if residu_normx is not None:  # add the norm of the residual
        normr = np.zeros(result.shape[0])
        if normr.shape[0]>3:
            normr[:4] = np.hstack([residu_normx, residu_normy])
        else:
            normr[:normr.shape[0]]= np.full(normr.shape[0],np.nan)
        result['NormR'] = normr
        del normr
    if result_quality is not None:  # add the error propagation
        if 'Error_propagation' in result_quality:
            result['error_x'] = prop_wieght_diagx
            result['error_y'] = prop_wieght_diagy
            sigma = np.zeros(result.shape[0])
            sigma[:2] = np.hstack([sigma0_weightx, sigma0_weighty])
            result['sigma0'] = sigma
    return A, result, dataf


# %% ======================================================================== #
#                               INTERPOLATION                                 #
# =========================================================================%% #

def interpolation_core(result: pd.DataFrame, interval_output: int, path_save: str, option_interpol: str = 'spline', 
                       first_date_interpol: np.datetime64 | str | None = None, last_date_interpol: np.datetime64 | str | None = None, 
                       unit: int = 365, redundancy: int | None = None, result_quality: list | None = None, 
                       verbose=False, visual: bool = False, data: pd.DataFrame | None = None, vmax = [False, False],
                       figsize = (12, 6), show_temp : bool = True):
    
    '''
    Interpolate Irregular Leap Frog time series (result of an inversion) to Regular LF time series using Cumulative Displacement times series.

    :param result: [pd dataframe] --- Leap frog displacement for x-component and y-component
    :param interval_output: [int] --- Period between two dates of the obtained RLF
    :param path_save: [str] --- Where to save the figures
    :param option_interpol: [str] [default is 'spline'] --- Type of interpolation, it can be 'spline', 'spline_smooth' or 'nearest'
    :param first_date_interpol: [np.datetime64 | str | None] [default is None] --- First date of the interpolation
    :param last_date_interpol: [np.datetime64 | str | None] [default is None] --- Last date of the interpolation
    :param unit: [int] [default is 365] --- 1 for m/d, 365 for m/y
    :param redundancy: [int | None] [default is None] --- If None there is no redundancy between two velocity in the interpolated time-series, else the overlap between two velocities is redundancy days
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param verbose: [bool] [default is False] --- Print informations along the way
    :param visual: [bool] [default is True] --- Keep the weights for future plots
    :param data: [pd dataframe | None] [default is None] --- Each line is (date1, date2, vx, vy, errorx, errory) for which a velocity is computed
    :param vmax: [list] [default is [False, False]] --- [min,max] where min,max correspond to the ylim of the figures
    :param figsize: [tuple] [default is (12, 6)] --- (width, height) where width and height are the size of the figures
    :param show_temp: [bool] [default is True] --- If True, show the temporal baseline on the plot

    :return dataf_lp: [pd dataframe] --- Result of the temporal interpolation
    '''
    
    ##  Reconstruction of COMMON REF TIME SERIES, e.g. cumulative displacement time series
    dataf = reconstruct_common_ref(result, result_quality) # Build cumulative displacement time series
    if first_date_interpol is None:
        start_date = dataf['Ref_date'][0] # First date at the considered pixel
    else:
        start_date = pd.to_datetime(first_date_interpol)

    x = np.array((dataf['Second_date'] - np.datetime64(start_date)).dt.days)  # Number of days according to the start_date
    if len(x) <= 1 or (np.isin('spline', option_interpol) and len(x) <= 3):  # It is not possible to interpolate, because too few estimation
        return pd.DataFrame({'First_date': [], 'Second_date': [], 'vx': [], 'vy': [], 'xcount_x': [], 'xcount_y': [], 'dz': [],
             'vz': [], 'xcount_z': [], 'NormR': []})

    # Compute the functions used to interpolate
    fdx, fdy, fdx_xcount, fdy_xcount, fdx_error, fdy_error = set_function_for_interpolation(option_interpol, x, dataf, result_quality)

    if redundancy is None: # No redundancy between two interpolated velocity
        x_regu = np.arange(np.min(x) + (interval_output - np.min(x) % interval_output), np.max(x), interval_output)
    else: # The overlap between two velocities corresponds to redundancy
        x_regu = np.arange(np.min(x) + (redundancy - np.min(x) % redundancy), np.max(x),
                           redundancy) # To make sure that the first element of x_regu is multiple of redundancy

    if len(x_regu) <= 1: # No interpolation
        return pd.DataFrame({'First_date': [], 'Second_date': [], 'vx': [], 'vy': [], 'xcount_x': [], 'xcount_y': [], 'dz': [],
             'vz': [], 'xcount_z': [], 'NormR': []})

    ##  Reconstruct a time series with a given temporal sampling, and a given overlap
    step = interval_output if redundancy is None else int(interval_output / redundancy)
    if step >= len(x_regu):
        return pd.DataFrame({'First_date': [], 'Second_date': [], 'vx': [], 'vy': [], 'xcount_x': [], 'xcount_y': [], 'dz': [],
             'vz': [], 'xcount_z': [], 'NormR': []})

    x_shifted = x_regu[step:]
    dx = fdx(x_shifted) - fdx(x_regu[:-step]) # Equivalent to [fdx(x_regu[i + step]) - fdx(x_regu[i]) for i in range(len(x_regu) - step)]
    dy = fdy(x_shifted) - fdy(x_regu[:-step]) # Equivalent to [fdy(x_regu[i + step]) - fdy(x_regu[i]) for i in range(len(x_regu) - step)]
    if result_quality is not None:
        if 'X_contribution' in result_quality:
            xcount_x = fdx_xcount(x_shifted) - fdx_xcount(x_regu[:-step])
            xcount_y = fdy_xcount(x_shifted) - fdy_xcount(x_regu[:-step])
        if 'Error_propagation' in result_quality:
            error_x = fdx_error(x_shifted) - fdx_error(x_regu[:-step])
            error_y = fdy_error(x_shifted) - fdy_error(x_regu[:-step])
    vx = dx * unit / interval_output # Convert to velocity in m/d or m/y
    vy = dy * unit / interval_output # Convert to velocity in m/d or m/

    First_date = start_date + pd.to_timedelta(x_regu[:-step], unit='D') # Equivalent to [start_date + pd.Timedelta(x_regu[i], 'D') for i in range(len(x_regu) - step)]
    Second_date = start_date + pd.to_timedelta(x_shifted, unit='D')

    dataf_lp = pd.DataFrame({'First_date': First_date, 'Second_date': Second_date, 'vx': vx, 'vy': vy})
    if result_quality is not None:
        if 'X_contribution' in result_quality:
            dataf_lp['xcount_x'] = xcount_x
            dataf_lp['xcount_y'] = xcount_y
        if 'Error_propagation' in result_quality:
            dataf_lp['error_x'] = error_x * unit / interval_output
            dataf_lp['error_y'] = error_y * unit / interval_output
            dataf_lp['sigma0'] = np.concatenate([result['sigma0'][:2],np.full(dataf_lp.shape[0]-2, np.nan)])
    del x_regu, First_date, Second_date, vx, vy

    # Fill with nan values if the first date of the cube which will be interpolated is lower than the first date interpolated for this pixel
    if first_date_interpol is not None and dataf_lp['First_date'].iloc[0] > pd.Timestamp(first_date_interpol):
        first_date = np.arange(first_date_interpol, dataf_lp['First_date'].iloc[0], np.timedelta64(redundancy, 'D'))
        # dataf_lp = full_with_nan(dataf_lp, first_date=first_date,
        #                          second_date=first_date + np.timedelta64(interval_output, 'D'))
        nul_df = pd.DataFrame({'First_date': first_date, 'Second_date': first_date + np.timedelta64(interval_output, 'D'),
             'vx': np.full(len(first_date), np.nan), 'vy': np.full(len(first_date), np.nan)})
        if result_quality is not None:
            if 'X_contribution' in result_quality:
                nul_df['xcount_x'] = np.full(len(first_date), np.nan)
                nul_df['xcount_y'] = np.full(len(first_date), np.nan)
            if 'Error_propagation' in result_quality:
                nul_df['error_x'] = np.full(len(first_date), np.nan)
                nul_df['error_y'] = np.full(len(first_date), np.nan)
        dataf_lp = pd.concat([nul_df, dataf_lp], ignore_index=True)

    # Fill with nan values if the last date of the cube which will be interpolated is higher than the last date interpolated for this pixel
    if last_date_interpol is not None and dataf_lp['Second_date'].iloc[-1] < pd.Timestamp(last_date_interpol):
        first_date = np.arange(dataf_lp['Second_date'].iloc[-1] + np.timedelta64(redundancy, 'D'), last_date_interpol,
                               np.timedelta64(redundancy, 'D'))
        # dataf_lp = full_with_nan(dataf_lp, first_date=(first_date - np.timedelta64(interval_output, 'D')),
        #                          second_date=first_date)
        nul_df = pd.DataFrame({'First_date': first_date - np.timedelta64(interval_output, 'D'), 'Second_date': first_date,
             'vx': np.full(len(first_date), np.nan), 'vy': np.full(len(first_date), np.nan)})
        dataf_lp = pd.concat([dataf_lp, nul_df], ignore_index=True)

    ##  Visualisation
    if visual: visualisation_interpolation(dataf_lp, data, path_save, interval_output=interval_output, show_temp=show_temp,
                                           unit='m/y' if unit == 365 else 'm/d', vmax=vmax, figsize=figsize,)

    return dataf_lp

def interpolation_to_data(result: pd.DataFrame, data: pd.DataFrame, option_interpol: str = 'spline', unit: int = 365, 
                          result_quality: list | None = None):
    
    '''
    Interpolate Irregular Leap Frog time series (result of an inversion) to the dates of given data (usefull to compare
    TICOI results to a "ground truth").
    
    :param result: [pd dataframe] --- Leap frog displacement for x-component and y-component
    :param data: [pd dataframe] --- Ground truth data which the interpolation must fit along the temporal axis
    :param option_interpol: [str] [default is 'spline'] --- Type of interpolation, it can be 'spline', 'spline_smooth' or 'nearest'
    :param unit: [int] [default is 365] --- 1 for m/d, 365 for m/y
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    '''
    
    ##  Reconstruction of COMMON REF TIME SERIES, e.g. cumulative displacement time series
    dataf = reconstruct_common_ref(result, result_quality) # Build cumulative displacement time series
    start_date = dataf['Ref_date'][0] # First date at the considered pixel
    x = np.array((dataf['Second_date'] - np.datetime64(start_date)).dt.days)  # Number of days according to the start_date
    
    # Interpolation must be caried out in between the min and max date of the original data
    if data['date1'].min() < result['date2'].min() or data['date2'].max() > result['date2'].max():
        data = data[(data['date1'] > result['date2'].min()) & (data['date2'] < result['date2'].max())]
        
    # Ground truth first and second dates
    x_gt_date1 = np.array((data['date1'] - start_date).dt.days)
    x_gt_date2 = np.array((data['date2'] - start_date).dt.days)
    
    ##  Interpolate the displacements and convert to velocities
    # Compute the functions used to interpolate
    fdx, fdy, fdx_xcount, fdy_xcount, fdx_error, fdy_error = set_function_for_interpolation(option_interpol, x, dataf, result_quality)
    
    # Interpolation
    dx = fdx(x_gt_date2) - fdx(x_gt_date1) # Equivalent to [fdx(x_regu[i + step]) - fdx(x_regu[i]) for i in range(len(x_regu) - step)]
    dy = fdy(x_gt_date2) - fdy(x_gt_date1) # Equivalent to [fdy(x_regu[i + step]) - fdy(x_regu[i]) for i in range(len(x_regu) - step)]
    
    # Convertion
    vx = dx * unit / data['temporal_baseline']  # Convert to velocity in m/d or m/y
    vy = dy * unit / data['temporal_baseline']  # Convert to velocity in m/d or m/y
    
    # Fill dataframe
    First_date = start_date + pd.to_timedelta(x_gt_date1, unit='D')  # Equivalent to [start_date + pd.Timedelta(x_regu[i], 'D') for i in range(len(x_regu) - step)]
    Second_date = start_date + pd.to_timedelta(x_gt_date2, unit='D')
    data_dict = {'First_date': First_date, 'Second_date': Second_date, 'vx': vx, 'vy': vy}
    dataf_lp = pd.DataFrame(data_dict)
    
    return dataf_lp


# %% ======================================================================== #
#                               GLOBAL PROCESS                                #
# =========================================================================%% #

def process(cube: cube_data_class, i: float | int, j: float | int, path_save, solver: str = 'LSMR', regu: int | str = 1, coef: int = 100, 
            flags: list | None = None, apriori_weight: bool = False, returned: list | str = 'interp', obs_filt: xr.Dataset = None, 
            interpolation_load_pixel: str = 'nearest', iteration: bool = True, interval_output: int = 1, first_date_interpol: np.datetime64 | None = None, 
            last_date_interpol: np.datetime64 | None = None, proj='EPSG:4326', threshold_it: float = 0.1, conf: bool = True, 
            option_interpol: str = 'spline', redundancy: int | None = None, detect_temporal_decorrelation: bool = True, unit: int = 365,
            result_quality: list | str | None = None, nb_max_iteration: int = 10, delete_outliers: int | str | None = None, 
            linear_operator: bool = False, visual: bool = False, verbose: bool = False):
    
    '''
    :params i, j: [float | int] --- Coordinates of the point in pixel
    :param solver: [str] [default is 'LSMR'] --- Solver of the inversion: 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
    :param regu: [int | str] [default is 1] --- Type of regularization
    :param coef: [int] [default is 100] --- Coef of Tikhonov regularisation
    :param flags: [list | None] [default is None] --- List which can contain 'linear_operator', which is the linear operator to use in the inversion (e.g. the covariance matrix of the observation errors), and 'mean', which is the mean of the observations
    :param apriori_weight: [bool] [default is False] --- If True use of aprori weight
    :param returned: [list | str] [default is 'interp'] --- What results must be returned ('raw', 'invert' and/or 'interp')
    :param obs_filt: [xr dataset] [default is None] --- Filtered dataset (e.g. rolling mean)
    :param interpolation_load_pixel: [str] [default is 'nearest'] --- Type of interpolation to load the previous pixel in the temporal interpolation ('nearest' or 'linear')
    :param iteration: [bool] [default is True] --- If True, use of iterations
    :param interval_output: [int] [default is 1] --- Temporal sampling of the leap frog time series
    :param first_date_interpol: [np.datetime64 | None] --- First date at wich the time series are interpolated
    :param last_date_interpol: [np.datetime64 | None] --- Last date at wich the time series are interpolated
    :param proj: [str] [default is 'EPSG:4326'] --- Projection of the cube 
    :param threshold_it: [float] [default is 0.1] --- Threshold to test the stability of the results between each iteration, use to stop the process
    :param conf: [bool] [default is False] --- If True means that the error corresponds to confidence intervals between 0 and 1, otherwise it corresponds to errors in m/y or m/d
    :param option_interpol: [str] [default is 'spline'] --- Type of interpolation, it can be 'spline', 'spline_smooth' or 'nearest'
    :param redundancy: [int | None] [default is None] --- If None there is no redundancy between two velocity in the interpolated time-series, else the overlap between two velocities is redundancy days
    :param detect_temporal_decorrelation: [bool] [default is True] --- If True the first inversion is solved using only velocity observations with small temporal baselines, to detect temporal decorelation
    :param unit: [int] [default is 365] --- 1 for m/d, 365 for m/y
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param nb_max_iteration: [int] [default is 10] --- Maximum number of iterations
    :param delete_outliers: [int | str | None] [default is None] --- Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle') 
    :param linear_operator: [bool] [default is False] --- If linear operator, the inversion is performed using a linear operator (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)
    :param visual: [bool] [default is False] --- Keep the weights for future plots
    :param verbose: [bool] [default is False] --- Print informations along the way
    
    :return dataf_list: [pd dataframe] Result of the temporal inversion + interpolation at point (i, j) if inversion was successful, an empty dataframe if not
    '''
    
    returned_list = []
    
    # Loading data at pixel location
    data = cube.load_pixel(i, j, proj=proj, interp=interpolation_load_pixel, solver=solver, coef=coef, regu=regu, 
                           rolling_mean=obs_filt, flags=flags)

    if 'raw' in returned:
        returned_list.append(data)
    
    if 'invert' in returned or 'interp' in returned:
        if flags is not None:
            regu, coef = data[3], data[4]
        
        # Inversion
        if delete_outliers == 'median_angle': conf = True  # set conf to True, because the errors have been replaced by confidence indicators based on the cos of the angle between the vector of each observation and the median vector
        result = inversion_core(data[0], i, j, dates_range=data[2], solver=solver, coef=coef, weight=apriori_weight, unit=unit, conf=conf, 
                                regu=regu, mean=data[1], iteration=iteration, threshold_it=threshold_it, 
                                detect_temporal_decorrelation=detect_temporal_decorrelation, linear_operator=linear_operator, 
                                result_quality=result_quality, nb_max_iteration=nb_max_iteration, visual=visual, verbose=verbose)
        
        if 'invert' in returned and 'interp' not in returned:
            if result[1] is not None:
                returned_list.append(result[1])
            else:
                returned_list.append(pd.DataFrame(
                    {'date1': [], 'date2': [], 'result_dx': [], 'result_dy': [], 'xcount_x': [], 'xcount_y': []}))

        if 'interp' in returned:   
            # Interpolation
            if result[1] is not None:  # if inversion have been performed
                dataf_list = interpolation_core(result[1], interval_output, path_save, option_interpol=option_interpol,
                                                first_date_interpol=first_date_interpol, last_date_interpol=last_date_interpol,
                                                visual=visual, data=data, unit=unit, redundancy=redundancy, 
                                                result_quality=result_quality, verbose=verbose)
        
                if result_quality is not None and 'Norm_residual' in result_quality: 
                    dataf_list['NormR'] = result[1]['NormR']  # store norm of the residual from the inversion
                returned_list.append(dataf_list)
            else:
                if result_quality is not None and 'Norm_residual' in result_quality: 
                    returned_list.append(pd.DataFrame({'First_date': [], 'Second_date': [], 'vx': [], 'vy': [], 'xcount_x': [], 'xcount_y': [], 'NormR': []}))
                else:
                    returned_list.append(pd.DataFrame({'First_date': [], 'Second_date': [], 'vx': [], 'vy': [], 'xcount_x': [], 'xcount_y': []}))
    
    if len(returned_list) == 1:
        return returned_list[0]
    return returned_list if len(returned_list) > 0 else None


def process_blocks_refine(cube: cube_data_class, nb_cpu: int = 8, block_size: float = 0.5, returned: list | str = 'interp', 
                          preData_kwargs: dict = None, inversion_kwargs: dict = None, verbose: bool = False):

    '''
    Separate the cube in several blocks computed synchronously one after the other by loading one block while the other is computed (with 
    parallelization) in order to avoid memory overconsumption and kernel crashing, and benefit from smaller computation time.
    
    :param cube: [cube_data_class] --- Cube of raw data to be processed
    :param nb_cpu: [int] [default is 8] --- Number of processing unit to use for parallel processing
    :param block_size: [float] [default is 0.5] --- Maximum size of the blocks (in GB)
    :param returned: [list | str] [default is 'interp'] --- What results must be returned ('raw', 'invert' and/or 'interp')
    :param preData_kwargs: [dict] [default is None] --- Pre-processing parameters (see cube_data_classxr.filter_cube)
    :param inversion_kwargs: [dict] [default is None] --- Inversion (and interpolation) parameters (see core.process)
    :param verbose: [bool] [default is False] --- Print informations along the way
    
    :return: [pd dataframe] Resulting estimated time series after inversion (and interpolation)
    '''    

    def chunk_to_block(cube, block_size=1, verbose=False):
        GB = 1073741824
        blocks = []
        if cube.ds.nbytes > block_size * GB:
            num_elements = np.prod([cube.ds.chunks[dim][0] for dim in cube.ds.chunks.keys()])
            chunk_bytes = num_elements * cube.ds['vx'].dtype.itemsize
            
            nchunks_block = int(block_size * GB // chunk_bytes)
            
            x_step = int(np.sqrt(nchunks_block))
            y_step = nchunks_block // x_step
            
            nblocks_x = int(np.ceil(len(cube.ds.chunks['x']) / x_step))
            nblocks_y = int(np.ceil(len(cube.ds.chunks['y']) / y_step))
            
            nblocks = nblocks_x * nblocks_y
            if verbose: print(f'Divide into {nblocks} blocks\n blocks size: {x_step * cube.ds.chunks["x"][0]} x {y_step * cube.ds.chunks["y"][0]}')

            for i in range(nblocks_y):
                for j in range(nblocks_x):
                    x_start = j * x_step * cube.ds.chunks['x'][0]
                    y_start = i * y_step * cube.ds.chunks['y'][0]
                    x_end = x_start + x_step * cube.ds.chunks['x'][0] if j != nblocks_x - 1 else cube.ds.dims['x']
                    y_end = y_start + y_step * cube.ds.chunks['y'][0] if i != nblocks_y - 1 else cube.ds.dims['y']
                    blocks.append([x_start, x_end, y_start,y_end])
        else:
            blocks.append([0, cube.ds.dims['x'], 0, cube.ds.dims['y']])
            if verbose: print(f'Cube size smaller than {block_size}GB, no need to divide')

        return blocks

    def load_block(cube: "cube_data_class", x_start: int, x_end: int, y_start: int, y_end: int,
                   flags: None | xr.Dataset):
        
        """
        Persist a block in memory, i.e. load it in a distributed way.
        """
        
        start = time.time()
        block = cube_data_class()
        block.ds = cube.ds.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
        block.ds = block.ds.persist()
        block.update_dimension()

        if flags is not None:
            flags_block = flags.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
        else:
            flags_block = None
        duration = time.time() - start

        return block, flags_block, duration


    async def process_block(block, returned=returned, nb_cpu=8, verbose=False): 

        xy_values = itertools.product(block.ds['x'].values, block.ds['y'].values)
        xy_values_tqdm = tqdm(xy_values, total=(block.nx * block.ny))

        if 'raw' in returned and (type(returned) == str or len(returned) == 1): # Only load the raw data
            result_block = Parallel(n_jobs=nb_cpu, verbose=0)(
                              delayed(block.load_pixel)(i, j, proj=inversion_kwargs['proj'], interp=inversion_kwargs['interpolation_load_pixel'],
                                      solver=inversion_kwargs['solver'], regu=inversion_kwargs['regu'], rolling_mean=None,
                                      visual=inversion_kwargs['visual'], verbose=inversion_kwargs['verbose'])
                              for i, j in xy_values_tqdm)
            return result_block
        
        obs_filt = block.filter_cube(**preData_kwargs)
        
        # There is no data on the whole block (masked data)
        if obs_filt is None and 'interp' in returned:
            if inversion_kwargs['result_quality'] is not None and 'Norm_residual' in inversion_kwargs['result_quality']: 
                return [pd.DataFrame({'First_date': [], 'Second_date': [], 'vx': [], 'vy': [], 'xcount_x': [], 'xcount_y': [], 'NormR': []})]
            else:
                return [pd.DataFrame({'First_date': [], 'Second_date': [], 'vx': [], 'vy': [], 'xcount_x': [], 'xcount_y': []})]
 
        result_block = Parallel(n_jobs=nb_cpu, verbose=0)(
        delayed(process)(block,
            i, j, obs_filt=obs_filt,returned=returned,**inversion_kwargs)
        for i, j in xy_values_tqdm)

        # result_block = [process(block, i, j, obs_filt=obs_filt,**inversion_kwargs)
        #                 for i, j in xy_values_tqdm]
        return result_block
    
    async def process_blocks_main(cube, nb_cpu=8, block_size=0.5, returned='interp', preData_kwargs=None, inversion_kwargs=None, verbose=False):
        
        # Get the parameters
        # if isinstance(preData_kwargs, dict) and isinstance(inversion_kwargs, dict):
        #     for key, value in preData_kwargs.items():
        #         globals()[key] = value
        #     for key, value in inversion_kwargs.items():
        #         globals()[key] = value
        # else:
        #     raise ValueError('preData_kwars and inversion_kwars must be a dict')

        # blocks = cube_split(cube, block_size=block_size, verbose=True)
        flags = preData_kwargs['flags']
        blocks = chunk_to_block(cube, block_size=block_size, verbose=True)
        
        dataf_list = [None] * ( cube.nx * cube.ny )

        loop = asyncio.get_event_loop()

        for n in range(len(blocks)):
            print(f'Processing block {n+1}/{len(blocks)}')

            # load the first block and start the loop
            if n == 0:
                x_start, x_end, y_start, y_end = blocks[0]
                future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end, flags)

            block, flags_block, duration = await future
            preData_kwargs['flags'] = flags_block
            inversion_kwargs['flags'] = flags_block
            print(f'Block {n+1} loaded in {duration:.2f} s')
            # if verbose: print(f'Block {n+1} loaded in {duration:.2f} s')

            if n < len(blocks) - 1:
                # load the next block while processing the current block
                x_start, x_end, y_start, y_end = blocks[n+1]
                future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end, flags)

            block_result = await process_block(block, returned=returned, nb_cpu=nb_cpu, verbose=verbose)

            for i in range(len(block_result)):
                row = i % block.ny + blocks[n][2]
                col = np.floor( i / block.ny ) + blocks[n][0]
                idx = int( col * cube.ny + row )

                dataf_list[idx] = block_result[i]

            del block_result, block, flags_block

        return dataf_list
      
    return asyncio.run(process_blocks_main(cube, nb_cpu=nb_cpu, block_size=block_size, returned=returned, preData_kwargs=preData_kwargs, 
                                           inversion_kwargs=inversion_kwargs, verbose=verbose))


# %% ======================================================================== #
#                               VISUALISATION                                 #
# =========================================================================%% #

def visualisation(data: pd.DataFrame, result: np.ndarray, option_visual: list, path_save: str, interval_output: int = 1, 
                  interval_inputMax: int | None = None, A: np.ndarray | None = None, dataf: pd.DataFrame | None = None, 
                  unit: str = 'm/y', figsize: tuple = (12, 6), show: bool = True):
    
    """
    Visualize the data (original datas and results from inversion and interpolation).
    
    :param data: [pd dataframe] --- An array where each line is (date1, date2, other elements ) for which a velocity is computed (correspond to the original displacements)
    :param result: [np array] --- Estimated velocity time series
    :param option_visual: [list of str] --- Among ['orginal_velocity_xy','original_magnitude','error','vv_good_quality','vv_quality','vxvy_quality','X','X_vxvy','X_magnitude','X_magnitude_Zoom','X_filter','X_filterZoom','X_magnitude_filter','Y_contribution','Residu','Residu_magnitude']
    :param path_save: [str] --- Path where to save the figures
    :param interval_output: [int] [default is 1] --- Temporal sampling of the leap frog time series
    :param interval_inputMax: [int | None] [default is None] --- Maximal temporal baseline
    :param A: [np array | None] [default is None] --- Matrix of the temporal invserion system AX=Y or AX=BY
    :param dataf: [pd dataframe | None] [default is None] --- Dataframe containing the observations data, and some information about the inversion
    :param unit: [str] [default is 'm/y'] --- 'm/d' or 'm/y'
    :param figsize: [tuple] [default is (12, 6)] --- Size of the figures (int1, int2)
    :param show: [bool] [default is True] --- If True the figures are showed
    """
    
    if data is not None and data.size:  # If there are datas at the given point
        conversion = unit
        unit = 'm/y' if unit == 365 else 'm/d'


        ## --------------- Vizualisation of the original data -------------- ##

        delta = dataf['date2'] - dataf['date1']  # temporal baseline of the observations
        date_cori = np.asarray(dataf['date1'] + delta // 2).astype('datetime64[D]')  # central date
        delta = np.asarray((delta).dt.days).astype('int')  # temporal basline as an integer
        offset_bar = delta // 2  # to plot the temporal baseline of the plots

        ####  Vizualisation of the original velocity x and y [m/y]
        if 'original_velocity_xy' in option_visual:
            fig1, ax1 = plt.subplots(2, 1, figsize=figsize)
            ax1[0].set_title('Original velocities', pad=10, fontsize=16)
            ymin = int(dataf['vx'].min())
            ymax = int(dataf['vx'].max())
            ax1[0].set_ylim(ymin, ymax)
            ax1[0].plot(date_cori, dataf['vx'], linestyle='', marker='o', markersize=3,
                        color='blueviolet')  # Display the vx components
            ax1[0].errorbar(date_cori, dataf['vx'], xerr=offset_bar, color='b', alpha=0.5, fmt=',', zorder=1)
            ax1[0].set_ylabel(f'Vx [{unit}]', fontsize=16)
            ymin = dataf['vy'].min()
            ymax = dataf['vy'].max()
            ax1[1].set_ylim(ymin, ymax)
            ax1[1].plot(date_cori, dataf['vy'], linestyle='', marker='o', markersize=3, color='blueviolet',
                        label='center between first and second date of acquisition')  # Display the vx components
            ax1[1].errorbar(date_cori, dataf['vy'], xerr=offset_bar, color='b', alpha=0.2, fmt=',', zorder=1,
                            label=f'Temporal Baselines [days]')
            ax1[1].set_ylabel(f'Vy [{unit}]', fontsize=16)
            plt.subplots_adjust(bottom=0.15)
            ax1[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12)
            if show: plt.show()
            fig1.savefig(f'{path_save}vx_vy.png')

        ####  Vizualisation of the original velocity magnitude [m/y]
        if 'original_magnitude' in option_visual:
            vv = np.round(np.sqrt((dataf['vx'] ** 2 + dataf['vy'] ** 2).astype('float')),
                          2)  # Compute the magnitude of the velocity
            ymin = vv.min() - 1
            ymax = vv.max() + 1
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(f'Velocity magnitude  [{unit}]')
            p = ax.plot(date_cori, vv, linestyle='', zorder=1, marker='o', lw=0.7, markersize=2)
            ax.errorbar(date_cori, vv, xerr=offset_bar, color='b', alpha=0.2, fmt=',', zorder=1)
            plt.title('Original velocity magnitude')
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}vv.png')

        ####  Velocity magnitude overlayed by quality/error values
        if 'vv_quality' in option_visual:
            vv = np.round(np.sqrt((dataf['vx'] ** 2 + dataf['vy'] ** 2).astype('float')),
                          2)  # Compute the magnitude of the velocity
            quality = dataf['errorx']
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylabel(f'Velocity magnitude  [{unit}]')
            scat = ax.scatter(date_cori, vv, c=quality, s=5, cmap='rainbow', label=f'Error [{unit}]')
            legend1 = ax.legend(*scat.legend_elements(num=10), loc="upper right", title=f"Error [{unit}]")
            ax.add_artist(legend1)
            plt.title('Original velocity magnitude')
            if show: plt.show(block=False)
            plt.legend()
            fig.savefig(f'{path_save}vv_quality.png')

        ####  Original velocity magnitude with quality higher than 0.5
        if 'vv_good_quality' in option_visual:
            vv = np.round(np.sqrt((dataf['vx'] ** 2 + dataf['vy'] ** 2).astype('float')),
                          2)  # Compute the magnitude of the velocity
            quality = data['errorx']
            vv_g = np.where(quality > 0.5, vv, np.nan)
            # date_cori_g = np.where(quality>0.5,date_cori,np.nan)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylabel(f'Velocity magnitude [{unit}]')
            p = ax.plot(date_cori, vv_g, linestyle='', zorder=1, marker='o', lw=0.7, markersize=2)
            ax.errorbar(date_cori, vv_g, xerr=offset_bar, color='b', alpha=0.2, fmt=',', zorder=1)
            plt.title('Original velocity magnitude with quality higher than 0.5')
            if show: plt.show(block=False)
            plt.legend()
            fig.savefig(f'{path_save}vv_good_quality.png')

        ####  vx and vy overlayed by quality/error values
        if 'vxvy_quality' in option_visual:
            qualityx = dataf['errorx']
            qualityy = dataf['errory']
            fig, ax = plt.subplots(2, 1, figsize=figsize)
            ax[0].set_ylabel(f'Velocity x [{unit}]')
            scat = ax[0].scatter(date_cori, dataf['vx'], c=qualityx, s=5, cmap='rainbow', label=f'Error [{unit}]')
            ax[1].set_ylabel(f'Velocity x [{unit}]')
            scat = ax[1].scatter(date_cori, dataf['vy'], c=qualityy, s=5, cmap='rainbow', label=f'Error [{unit}]')
            legend1 = ax[1].legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0, -0.7), ncol=10,
                                   title='Confidence')
            ax[1].add_artist(legend1)
            plt.subplots_adjust(bottom=0.20)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.15, 0), bbox_transform=fig.transFigure, fontsize=12)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}vxvy_quality_bas.png')


        ## ------------------ Vizualisation of the results ----------------- ##

        delta_r = result['date2'] - result['date1']
        offset_bar_r = delta_r / 2
        delta_r = delta_r.dt.days
        dates_deplacement_inv = (result['date2'] - offset_bar_r).tolist()  # Central date
        print('Averaged temporal baseline of Irregular Time Series', np.mean(delta_r))
        result['result_vx'] = result['result_dx'] / delta_r * conversion
        result['result_vy'] = result['result_dy'] / delta_r * conversion

        ####  Velocity observation overlayed by the results for vx, vy components (min, max are from the results)
        if 'X' in option_visual:
            fig1, ax1 = plt.subplots(2, 1, figsize=figsize)
            ax1[0].set_title('Results of velocity x and y')
            ymin = np.nanmin(result['result_vx']) - 50
            ymax = np.nanmin(result['result_vx']) + 50
            ax1[0].set_ylim(ymin, ymax)
            ax1[0].plot(date_cori, dataf['vx'], linestyle='', marker='o', markersize=3,
                        color='orange', label=f'Velocity observations [{unit}]', alpha=0.7)  # Display the vx components
            ax1[0].errorbar(date_cori, dataf['vx'], xerr=offset_bar, color='orange', alpha=0.2, fmt=',', zorder=1)
            ax1[0].plot(dates_deplacement_inv, result['result_vx'], color='b', linestyle='', marker='o',
                        markersize=3)  # Display the vx components
            ax1[0].errorbar(dates_deplacement_inv, result['result_vx'], xerr=offset_bar_r, color='b', alpha=0.2,
                            fmt=',', zorder=1)
            ax1[0].set_ylabel(f'Velocity x [{unit}]')
            ax1[0].set_xlabel('Centrale Dates')
            label = 'interval output: {} days\n interval used : [ {} : {} ] days'.format(interval_output,
                                                                                         interval_output,
                                                                                         interval_inputMax)
            ymin = np.nanmin(result['result_vy']) - 50
            ymax = np.nanmax(result['result_vy']) + 50
            ax1[1].set_ylim(ymin, ymax)
            ax1[1].plot(date_cori, dataf['vy'], linestyle='', marker='o', markersize=3,
                        color='orange', label=f'Velocity observations [{unit}]', alpha=0.7)  # Display the vy components
            ax1[1].errorbar(date_cori, dataf['vy'], xerr=offset_bar, color='orange', alpha=0.2, fmt=',', zorder=1)
            ax1[1].plot(dates_deplacement_inv, result['result_vy'], color='b', linestyle='', marker='o', label=label,
                        markersize=3)  # Display the vy components
            ax1[1].errorbar(dates_deplacement_inv, result['result_vy'], xerr=offset_bar_r, color='b', alpha=0.2,
                            fmt=',', zorder=1, label=f'Temporal Baselines of {interval_output} days')
            ax1[1].set_ylabel(f'Velocity y [{unit}]')
            ax1[1].set_xlabel('Centrale Dates')
            plt.subplots_adjust(bottom=0.20)
            ax1[1].legend(loc='lower left', bbox_to_anchor=(0.15, 0), bbox_transform=fig1.transFigure, fontsize=12,
                          ncol=2)
            if show: plt.show()
            fig1.savefig(f'{path_save}X_velocity.png')

        ####  Velocity observation overlayed by the results for vx, vy components (min,max are from the observation)
        if 'X_zoom' in option_visual:
            fig1, ax1 = plt.subplots(2, 1, figsize=figsize)
            ax1[0].set_title('Resulting velocities', pad=10, fontsize=18)
            ymin = data['vx'].min()
            ymax = data['vx'].max()
            ax1[0].set_ylim(ymin, ymax)
            ax1[0].plot(date_cori, dataf['vx'], linestyle='', marker='o', markersize=3,
                        color='orange', label=f'Velocity observations  [{unit}]',
                        alpha=0.7)  # Display the vx components
            ax1[0].errorbar(date_cori, dataf['vx'], xerr=offset_bar, color='orange', alpha=0.2, fmt=',', zorder=1)
            ax1[0].plot(dates_deplacement_inv, result['result_vx'], linestyle='', marker='o', markersize=3,
                        color='r', label=f'Irregular LF velocities  [{unit}]')  # Display the vx components
            ax1[0].set_ylabel(f'Vx  [{unit}]', fontsize=18)
            ax1[0].errorbar(dates_deplacement_inv, result['result_vx'], xerr=offset_bar_r, color='r', alpha=0.2,
                            fmt=',', zorder=1)
            ymin = data['vy'].min()
            ymax = data['vy'].max()
            ax1[1].set_ylim(ymin, ymax)
            ax1[1].plot(date_cori, dataf['vy'], linestyle='', marker='o', markersize=3,
                        color='orange', label=f'Velocity observations  [{unit}]',
                        alpha=0.7)  # Display the vy components
            ax1[1].errorbar(date_cori, dataf['vy'], xerr=offset_bar, color='orange', alpha=0.2, fmt=',', zorder=1)
            ax1[1].plot(dates_deplacement_inv, result['result_vy'], linestyle='', marker='o', markersize=3, color='r',
                        label=f'Irregular LF velocities  [{unit}]')  # Display the vy components
            ax1[1].errorbar(dates_deplacement_inv, result['result_vy'], xerr=offset_bar_r, color='r', alpha=0.2,
                            fmt=',', zorder=1, label=f'Temporal Baselines [days]')
            ax1[1].set_ylabel(f'Vy  [{unit}]', fontsize=18)
            plt.subplots_adjust(bottom=0.2)
            ax1[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12)
            if show: plt.show()
            fig1.savefig(f'{path_save}X_velocity_Zoom.png')

        ####  Vizualisation of the original velocity magnitude [m/an] overlayed with the velocity magnitude of the result, the min and max are from the observations
        if 'X_magnitude_zoom' in option_visual:
            vv = np.round(np.sqrt((result['result_vx'] ** 2 + result['result_vy'] ** 2).astype('float')),
                          2)  # Compute the magnitude of the velocity
            vv_ori = np.round(np.sqrt((dataf['vx'] ** 2 + dataf['vy'] ** 2).astype('float')),
                              2)  # Compute the magnitude of the velocity
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylabel(f'Velocity magnitude [{unit}]', fontsize=16)
            ax.set_ylim(vv_ori.min(), vv_ori.max())
            p = ax.plot(date_cori, vv_ori, color='orange', linestyle='', zorder=1, marker='o', lw=0.7, markersize=2,
                        label=f'Central date of velocity observations [{unit}]')
            ax.errorbar(date_cori, vv_ori, xerr=offset_bar, color='orange', alpha=0.2, fmt=',', zorder=1,
                        label=f'Temporal baseline of velocity observations [{unit}]')
            p = ax.plot(dates_deplacement_inv, vv, linestyle='', color='b', zorder=1, marker='o', lw=0.7, markersize=2,
                        label=f'Central date of Irregular LF velocities  [{unit}]')
            ax.errorbar(dates_deplacement_inv, vv, xerr=offset_bar_r, color='b', alpha=0.2, fmt=',', zorder=1,
                        label=f'Temporal sampling of Irregular LF velocities  [{unit}]')
            plt.subplots_adjust(bottom=0.2)
            ax.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12, ncol=2)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}Xvv_Zoom_bis.png')

        ####  Vizualisation of the original velocity magnitude [m/an] overlayed with the velocity magnitude of the result, the min and max are from the results
        if 'X_magnitude' in option_visual:
            vv = np.round(np.sqrt((result['result_vx'] ** 2 + result['result_vy'] ** 2).astype('float')),
                          2)  # Compute the magnitude of the velocity
            vv_ori = np.round(np.sqrt((data['vx'] ** 2 + data['vy'] ** 2).astype('float')), 2)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylabel(f'Velocity magnitude [{unit}]')
            ax.set_ylim(vv.min(), vv.max())
            p = ax.plot(dates_deplacement_inv, vv, linestyle='', zorder=1, marker='o', lw=0.7, markersize=2, color='b')
            ax.errorbar(dates_deplacement_inv, vv, xerr=offset_bar_r, color='b', alpha=0.2, fmt=',', zorder=1)
            p = ax.plot(date_cori, vv_ori, color='orange', linestyle='', zorder=1, marker='o', lw=0.7, markersize=2)
            ax.errorbar(date_cori, vv_ori, xerr=offset_bar, color='orange', alpha=0.2, fmt=',', zorder=1)
            plt.title('Resulting velocity magnitude')
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}Xvv.png')

        ####  How many displacement in Y have contributed to each displacement in X
        if 'Y_contribution' in option_visual:
            fig, ax = plt.subplots(2, 1, figsize=figsize)
            ax[0].set_ylabel(f'Velocity x [{unit}]')
            scat = ax[0].scatter(dates_deplacement_inv, result['result_vx'], c=result['xcount_x'], s=4,
                                 cmap='rainbow', label='Y_contribution')
            ax[1].set_ylabel(f'Velocity x [{unit}]')
            scat = ax[1].scatter(dates_deplacement_inv, result['result_vy'], c=result['xcount_y'], s=4,
                                 cmap='rainbow', label='Y_contribution')
            legend1 = ax[1].legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.1, 0), ncol=5,
                                   bbox_transform=fig.transFigure,
                                   title="Y_contribution")
            plt.subplots_adjust(bottom=0.2)
            ax[1].add_artist(legend1)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}X_dates_contribution_vx_vy.png')

            vv = np.round(np.sqrt((result['result_vx'] ** 2 + result['result_vy'] ** 2).astype('float')),
                          2)  # compute the magnitude of the velocity
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylabel(f'Velocity magnitude [{unit}]', fontsize=16)
            scat = ax.scatter(dates_deplacement_inv, vv, c=(result['xcount_x'] + result['xcount_y']) / 2, s=4, vmin=0,
                              vmax=100,
                              cmap='viridis_r', label='Y_contribution')
            legend1 = ax.legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.1, 0), ncol=5,
                                bbox_transform=fig.transFigure,
                                title="Number of overlap from which the velocity is reconstructed", fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax.add_artist(legend1)
            plt.setp(legend1.get_title(), fontsize=16)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}X_dates_contribution_vv.png')

        ####  Residual from the inversion
        if 'Residu' in option_visual:

            # Reconstruct the observation velocity according to A and the estimated velocities
            Y_reconstruct_x = np.dot(A, result['result_vx'] * delta_r / conversion) / delta * conversion
            Y_reconstruct_y = np.dot(A, result['result_vy'] * delta_r / conversion) / delta * conversion

            # Velocity observation compared to reconstructed velocity from the inversion
            fig1, ax1 = plt.subplots(2, 1, figsize=figsize)
            ax1[0].plot(date_cori, dataf['vx'], linestyle='', marker='o', color='b', markersize=3,
                        alpha=0.3)  # Display the vx components
            ax1[0].errorbar(date_cori, dataf['vx'], xerr=offset_bar, color='b', fmt=',', zorder=1, alpha=0.5)
            ax1[0].plot(date_cori, Y_reconstruct_x, linestyle='', marker='o', color='r', markersize=3,
                        alpha=0.2)  # Display the vx components
            ax1[0].errorbar(date_cori, Y_reconstruct_x, xerr=offset_bar, color='r', alpha=0.2, fmt=',', zorder=1)
            ax1[0].set_ylabel(f'Vx [{unit}]', fontsize=18)
            ax1[1].plot(date_cori, dataf['vy'], linestyle='', marker='o', color='b', markersize=3, alpha=0.3,
                        label='Original data')  # Display the vy components
            ax1[1].errorbar(date_cori, dataf['vy'], xerr=offset_bar, color='b', fmt=',', zorder=1, alpha=0.3)
            ax1[1].plot(date_cori, Y_reconstruct_y, linestyle='', marker='o', color='r', markersize=3, alpha=0.2,
                        label='Reconstructed Data')  # Display the vy components
            ax1[1].errorbar(date_cori, Y_reconstruct_y, xerr=offset_bar, color='r', alpha=0.3, fmt=',', zorder=1)
            ax1[1].set_ylabel(f'Vy [{unit}]', fontsize=18)
            ax1[1].legend(bbox_to_anchor=(0.55, -0.3), ncol=3, fontsize=15)
            if show: plt.show()
            fig1.savefig(f'{path_save}vx_vy_mismatch.png')

            # Plot the residual from the last inversion
            Final_residux = dataf['residux']
            Final_residuy = dataf['residuy']

            fig, ax = plt.subplots(2, 1, figsize=(8, 4))
            ax[0].set_ylabel(f'Vx [{unit}]')
            scat1 = ax[0].scatter(date_cori, dataf['vx'], c=abs(Final_residux), s=5, cmap='plasma_r', edgecolors='k',
                                  linewidth=0.1)
            ax[1].set_ylabel(f'Vy [{unit}]')
            scat2 = ax[1].scatter(date_cori, dataf['vy'], c=abs(Final_residuy), s=5, cmap='plasma_r', edgecolors='k',
                                  linewidth=0.1)
            plt.subplots_adjust(bottom=0.3)
            legend1 = ax[1].legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                                   bbox_transform=fig.transFigure,
                                   ncol=3, title="Absolute residual Vx")
            legend2 = ax[1].legend(*scat2.legend_elements(num=5), loc='lower right', bbox_to_anchor=(0.95, 0),
                                   bbox_transform=fig.transFigure,
                                   ncol=3, title="Absolute residual Vy")
            ax[1].add_artist(legend1)
            ax[1].add_artist(legend2)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}vx_vy_final_residual.png')

            # Plot the first weight used in the inversion
            fig, ax = plt.subplots(2, 1, figsize=(8, 4))
            ax[0].set_ylabel(f'Vx [{unit}]')
            scat1 = ax[0].scatter(date_cori, dataf['vx'], c=abs(dataf['weightinix']), s=5, cmap='plasma_r',
                                  edgecolors='k', linewidth=0.1)
            ax[1].set_ylabel(f'Vy [{unit}]')
            scat2 = ax[1].scatter(date_cori, dataf['vy'], c=abs(dataf['weightiniy']), s=5, cmap='plasma_r',
                                  edgecolors='k', linewidth=0.1)
            plt.subplots_adjust(bottom=0.3)
            legend1 = ax[1].legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                                   bbox_transform=fig.transFigure,
                                   ncol=3, title="Weight ini Vx")
            legend2 = ax[1].legend(*scat2.legend_elements(num=5), loc='lower right', bbox_to_anchor=(0.95, 0),
                                   bbox_transform=fig.transFigure,
                                   ncol=3, title="Weight ini Vy")
            ax[1].add_artist(legend1)
            ax[1].add_artist(legend2)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}vx_vy_weightini.png')

            # Plot the last weight of the inversion
            fig, ax = plt.subplots(2, 1, figsize=(8, 4))
            ax[0].set_ylabel(f'Vx [{unit}]')
            scat1 = ax[0].scatter(date_cori, dataf['vx'], c=abs(dataf['weightlastx']), s=5, cmap='plasma_r',
                                  edgecolors='k', linewidth=0.1)
            ax[1].set_ylabel(f'Vy [{unit}]')
            scat2 = ax[1].scatter(date_cori, dataf['vy'], c=abs(dataf['weightlasty']), s=5, cmap='plasma_r',
                                  edgecolors='k', linewidth=0.1)
            plt.subplots_adjust(bottom=0.3)
            legend1 = ax[1].legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                                   bbox_transform=fig.transFigure,
                                   ncol=3, title="Last weight Vx")
            legend2 = ax[1].legend(*scat2.legend_elements(num=5), loc='lower right', bbox_to_anchor=(0.95, 0),
                                   bbox_transform=fig.transFigure,
                                   ncol=3, title="Last weight Vy")
            ax[1].add_artist(legend1)
            ax[1].add_artist(legend2)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}vx_vy_weightlast.png')

            # Comparison between authors and sensors of the residual
            dataf[dataf['author'] == 'L. Charrier, J. Mouginot, R.Millan, A.Derkacheva']['author'] = 'IGE'
            dataf = dataf.replace('L. Charrier, J. Mouginot, R.Millan, A.Derkacheva', 'IGE')
            dataf = dataf.replace('S. Leinss, L. Charrier', 'Leinss')

            dataf["abs_residux"] = abs(dataf["residux"])
            dataf["abs_residuy"] = abs(dataf["residuy"])

            dataf = dataf.rename(columns={"author": 'Author'})
            dataf.to_csv(f'{path_save}dataf.csv')

            ax = sns.catplot(data=dataf, x="sensor", y="abs_residux", hue="Author", kind="box")
            ax.set(xlabel='Sensor', ylabel='Absolute residual vx [m/y]')
            if show: plt.show()
            plt.savefig(f'{path_save}vx_residual_author_abs.png')

            ax = sns.catplot(data=dataf, x="sensor", y="abs_residuy", hue="Author", kind="box")
            ax.set(xlabel='Sensor', ylabel='Absolute residual vy [m/y]')
            if show: plt.show()
            plt.savefig(f'{path_save}vy_residual_author_abs.png')

            ax = sns.catplot(data=dataf, x="sensor", y="residux", hue="Author", kind="box")
            ax.set(xlabel='Sensor', ylabel='Residual vx [m/y]')
            if show: plt.show()
            plt.savefig(f'{path_save}vx_residual_author.png')

            ax = sns.catplot(data=dataf, x="sensor", y="residuy", hue="Author", kind="box")
            ax.set(xlabel='Sensor', ylabel='Residual vy [m/y]')
            if show: plt.show()
            plt.savefig(f'{path_save}vy_residual_author.png')

            fig, ax = plt.subplots(2, 1, figsize=figsize)
            color_list = ['b', 'm', 'k', 'g', 'm']
            for i, auth in enumerate(dataf['Author'].unique()):
                ax[0].plot(dataf[dataf["Author"] == auth]['weightinix'], dataf[dataf["Author"] == auth]['residux'],
                           linestyle='', marker='o', color=color_list[i], markersize=3)
                ax[1].plot(dataf[dataf["Author"] == auth]['weightiniy'], dataf[dataf["Author"] == auth]['residuy'],
                           linestyle='', marker='o', color=color_list[i], markersize=3, label=auth)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
            ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
            ax[1].set_xlabel(f'Quality indicator', fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                         ncol=5)
            if show: plt.show()
            fig.savefig(f'{path_save}residu_qualitylog.png')

            fig, ax = plt.subplots(2, 1, figsize=figsize)
            color_list = ['b', 'm', 'k', 'g', 'm']
            for i, auth in enumerate(dataf['Author'].unique()):
                ax[0].plot(dataf[dataf["Author"] == auth]['weightinix'], dataf[dataf["Author"] == auth]['residux'],
                           linestyle='', marker='o', color=color_list[i], markersize=3)
                ax[1].plot(dataf[dataf["Author"] == auth]['weightiniy'],
                           dataf[dataf["Author"] == auth]['residuy'], linestyle='', marker='o',
                           color=color_list[i], markersize=3, label=auth)
            ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
            ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
            ax[1].set_xlabel(f'Quality indicator', fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                         ncol=5)
            if show: plt.show()
            fig.savefig(f'{path_save}residu_quality.png')

            fig, ax = plt.subplots(2, 1, figsize=figsize)
            color_list = ['b', 'm', 'k', 'g', 'm']
            for i, auth in enumerate(dataf['Author'].unique()):
                ax[0].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2, dataf[dataf["Author"] == auth]['residux'],
                           linestyle='', marker='o', color=color_list[i], markersize=3)
                ax[1].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2, dataf[dataf["Author"] == auth]['residuy'],
                           linestyle='', marker='o', color=color_list[i], markersize=3, label=auth)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
            ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
            ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                         ncol=5)
            if show: plt.show()
            fig.savefig(f'{path_save}residu_tempbaseline_log.png')

            fig, ax = plt.subplots(2, 1, figsize=figsize)
            color_list = ['b', 'm', 'k', 'g', 'm']
            for i, auth in enumerate(dataf['Author'].unique()):
                ax[0].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2,
                           abs(dataf[dataf["Author"] == auth]['residux']), linestyle='', marker='o',
                           color=color_list[i],
                           markersize=3)
                ax[1].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2,
                           abs(dataf[dataf["Author"] == auth]['residuy']), linestyle='', marker='o',
                           color=color_list[i], markersize=3, label=auth)
            ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
            ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
            ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                         ncol=5)
            if show: plt.show()
            fig.savefig(f'{path_save}residu_tempbaselineabs.png')

            fig, ax = plt.subplots(2, 1, figsize=figsize)
            color_list = ['b', 'm', 'k', 'g', 'm']
            for i, auth in enumerate(dataf['Author'].unique()):
                ax[0].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2,
                           dataf[dataf["Author"] == auth]['residux'], linestyle='', marker='o', color=color_list[i],
                           markersize=3)
                ax[1].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2,
                           dataf[dataf["Author"] == auth]['residuy'], linestyle='', marker='o',
                           color=color_list[i], markersize=3, label=auth)
            ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
            ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
            ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                         ncol=5)
            if show: plt.show()
            fig.savefig(f'{path_save}residu_tempbaseline.png')

            fig, ax = plt.subplots(2, 1, figsize=figsize)
            color_list = ['b', 'm', 'k', 'g', 'm']
            for i, auth in enumerate(dataf['Author'].unique()):
                ax[0].plot(dataf[dataf["Author"] == auth]['sensor'],
                           dataf[dataf["Author"] == auth]['residux'], linestyle='', marker='o', color=color_list[i],
                           markersize=3)
                ax[1].plot(dataf[dataf["Author"] == auth]["sensor"],
                           dataf[dataf["Author"] == auth]['residuy'], linestyle='', marker='o',
                           color=color_list[i], markersize=3, label=auth)
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
            ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
            ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.1, 0), bbox_transform=fig1.transFigure, fontsize=12,
                         ncol=4)
            if show: plt.show()
            fig.savefig(f'{path_save}residu_sensor_log.png')

            fig, ax = plt.subplots(2, 1, figsize=figsize)
            color_list = ['b', 'm', 'k', 'g', 'm']
            for i, auth in enumerate(dataf['Author'].unique()):
                ax[0].plot(dataf[dataf["Author"] == auth]['sensor'],
                           abs(dataf[dataf["Author"] == auth]['residux']), linestyle='', marker='o',
                           color=color_list[i],
                           markersize=3)
                ax[1].plot(dataf[dataf["Author"] == auth]["sensor"],
                           abs(dataf[dataf["Author"] == auth]['residuy']), linestyle='', marker='o',
                           color=color_list[i], markersize=3, label=auth)
            ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
            ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
            ax[1].set_xlabel(f'Sensor', fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.1, 0), bbox_transform=fig1.transFigure, fontsize=12,
                         ncol=4)
            if show: plt.show()
            fig.savefig(f'{path_save}residu_sensorabs.png')

            fig, ax = plt.subplots(2, 1, figsize=figsize)
            color_list = ['b', 'm', 'k', 'g', 'm']
            for i, auth in enumerate(dataf['Author'].unique()):
                ax[0].plot(dataf[dataf["Author"] == auth]['sensor'],
                           dataf[dataf["Author"] == auth]['residux'], linestyle='', marker='o', color=color_list[i],
                           markersize=3)
                ax[1].plot(dataf[dataf["Author"] == auth]["sensor"],
                           dataf[dataf["Author"] == auth]['residuy'], linestyle='', marker='o',
                           color=color_list[i], markersize=3, label=auth)
            ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
            ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
            ax[1].set_xlabel(f'Sensor', fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax[1].legend(loc='lower left', bbox_to_anchor=(0.1, 0), bbox_transform=fig1.transFigure, fontsize=12,
                         ncol=4)
            if show: plt.show()
            fig.savefig(f'{path_save}residu_sensor.png')

        print('Residu vxvy')
        print('MAE x', sm.mean_absolute_error(Y_reconstruct_x, dataf['vx']))
        print('MAE y', sm.mean_absolute_error(Y_reconstruct_y, dataf['vy']))

        print('MAD x', np.median(np.abs(Y_reconstruct_x - dataf['vx'])))
        print('MAD y', np.median(np.abs(Y_reconstruct_y - dataf['vy'])))

        print('RMSE x', sm.mean_squared_error(Y_reconstruct_x, dataf['vx'], squared=False))
        print('RMSE y', sm.mean_squared_error(Y_reconstruct_y, dataf['vy'], squared=False))

        if 'Residu_magnitude' in option_visual:

            # Reconstruct the velocity magnitude from the inversion according to A, and the estimated velocity
            vv = np.round(np.sqrt((dataf['vx'] ** 2 + dataf['vy'] ** 2).astype('float')), 2)
            Y_reconstruct_x = np.dot(A, result['result_dx']) / delta * conversion
            Y_reconstruct_y = np.dot(A, result['result_dy']) / delta * conversion
            Y_reconstruct_vv = np.round(np.sqrt((Y_reconstruct_x ** 2 + Y_reconstruct_y ** 2).astype('float')), 2)

            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(date_cori, vv, linestyle='', marker='o', color='b', markersize=3,
                     alpha=0.3)  # Display the vx components
            ax1.errorbar(date_cori, vv, xerr=offset_bar, color='b', fmt=',', zorder=1, alpha=0.3)
            ax1.plot(date_cori, Y_reconstruct_vv, linestyle='', marker='o', color='r', markersize=3,
                     alpha=0.2)  # Display the vx components
            ax1.set_ylabel(f'Vv [{unit}]', fontsize=18)
            ax1.legend(bbox_to_anchor=(0.55, -0.3), ncol=3, fontsize=15)
            if show: plt.show()
            fig1.savefig(f'{path_save}vv_reconstruct.png')

            # Residual from the last inversion
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_ylabel(f'Vv [{unit}]')
            scat1 = ax.scatter(date_cori, vv, c=[np.sqrt(dataf['residux'][z] ** 2 + dataf['residuy'][z] ** 2) for z in
                                                 range(dataf.shape[0])],
                               s=10, cmap='plasma_r', edgecolors='k', linewidth=0.1)

            plt.subplots_adjust(bottom=0.2)
            legend1 = ax.legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                                bbox_transform=fig.transFigure,
                                ncol=10, title="Residual")
            ax.add_artist(legend1)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}vv_residu.png')

            # First weight
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_ylabel(f'Vv [{unit}]')
            scat1 = ax.scatter(date_cori, vv,
                               c=[np.sqrt(dataf['weightinix'][z] ** 2 + dataf['weightiniy'][z] ** 2) for z in
                                  range(dataf.shape[0])],
                               s=10, cmap='plasma_r', edgecolors='k', linewidth=0.1)

            plt.subplots_adjust(bottom=0.2)
            legend1 = ax.legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                                bbox_transform=fig.transFigure,
                                ncol=10, title="Initial weight")
            ax.add_artist(legend1)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}vv_weightini.png')

            # Last weight
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_ylabel(f'Vv [{unit}]')
            scat1 = ax.scatter(date_cori, vv,
                               c=[np.sqrt(dataf['weightlastx'][z] ** 2 + dataf['weightlasty'][z] ** 2) for z in
                                  range(dataf.shape[0])],
                               s=10, cmap='plasma_r', edgecolors='k', linewidth=0.1)

            plt.subplots_adjust(bottom=0.2)
            legend1 = ax.legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                                bbox_transform=fig.transFigure,
                                ncol=10, title="Last weight")
            ax.add_artist(legend1)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}vv_weightlast.png')

            print('MAE relatif x', sm.mean_absolute_error(Y_reconstruct_x, dataf['vx']) / np.mean(dataf['vx']))
            print('MAE relatif y', sm.mean_absolute_error(Y_reconstruct_y, dataf['vy']) / np.mean(dataf['vy']))

            print('RMSE y', sm.mean_squared_error(Y_reconstruct_x, dataf['vx'], squared=False))
            print('RMSE y', sm.mean_squared_error(Y_reconstruct_y, dataf['vy'], squared=False))
        if 'Error_propagation' in option_visual:
            fig, ax = plt.subplots(2, 1, figsize=figsize)
            ax[0].set_ylabel(f'Velocity x [{unit}]')
            scat = ax[0].scatter(dates_deplacement_inv, result['result_vx'], c=result['Error_x'] / delta_r * 365, s=4,
                                 cmap='rainbow', label='errorx')
            ax[1].set_ylabel(f'Velocity x [{unit}]')
            scat = ax[1].scatter(dates_deplacement_inv, result['result_vy'], c=result['Error_y'] / delta_r * 365, s=4,
                                 cmap='rainbow', label='errory')
            legend1 = ax[1].legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.1, 0), ncol=5,
                                   bbox_transform=fig.transFigure,
                                   title="errory")
            plt.subplots_adjust(bottom=0.2)
            ax[1].add_artist(legend1)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}X_dates_error_prop.png')

            vv = np.round(np.sqrt((result['result_vx'] ** 2 + result['result_vy'] ** 2).astype('float')),
                          2)  # compute the magnitude of the velocity
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_ylabel(f'Velocity magnitude [{unit}]', fontsize=16)
            scat = ax.scatter(dates_deplacement_inv, vv, c=(result['xcount_x'] + result['xcount_y']) / 2, s=4, vmin=0,
                              vmax=100,
                              cmap='viridis_r', label='Y_contribution')
            legend1 = ax.legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.1, 0), ncol=5,
                                bbox_transform=fig.transFigure,
                                title="Number of overlap from which the velocity is reconstructed", fontsize=16)
            plt.subplots_adjust(bottom=0.2)
            ax.add_artist(legend1)
            plt.setp(legend1.get_title(), fontsize=16)
            if show: plt.show(block=False)
            fig.savefig(f'{path_save}X_dates_contribution_vv.png')

    else:
        print(f'NO DATA')
