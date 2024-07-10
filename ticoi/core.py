#!/usr/bin/env python
"""
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
"""

import asyncio
import itertools
import time
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

from ticoi.cube_data_classxr import cube_data_class
from ticoi.interpolation_functions import (
    reconstruct_common_ref,
    set_function_for_interpolation,
)
from ticoi.inversion_functions import (
    TukeyBiweight,
    class_linear_operator,
    construction_a_lf,
    construction_dates_range_np,
    find_date_obs,
    inversion_one_component,
    inversion_two_components,
    mu_regularisation,
    weight_for_inversion,
)
from ticoi.pixel_class import pixel_class

warnings.filterwarnings("ignore")

# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #


def inversion_iteration(
    data: np.ndarray,
    A: np.ndarray,
    dates_range: np.ndarray,
    solver: str,
    coef: int,
    Weight: np.ndarray,
    result_dx: np.ndarray,
    result_dy: np.ndarray,
    mu: np.ndarray,
    regu: int | str = 1,
    accel: np.ndarray | None = None,
    linear_operator=Union["class_linear_operator", None],
    result_quality: list | str | None = None,
    ini: np.ndarray | None = None,
    verbose: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None):

    """
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
    :param verbose: [bool] [default is False] --- Print information along the way

    :return result_dx, result_dy: [np arrays] --- Obtained results (velocities) for this iteration along x and y axis
    :return weightx, weighty: [np arrays] --- Newly computed weights along x and y axis
    :return residu_normx, residu_normy: [np arrays | None] --- Norm of the residu along x and y axis (when showing the L curve)
    """

    def compute_residual(A: np.ndarray, v: np.ndarray, X: np.ndarray) -> np.ndarray:
        Residu = v - A.dot(X)
        return Residu

    def weightf(residu: np.ndarray, Weight: np.ndarray) -> np.ndarray:

        """
        Compute weight according to the residual

        :param residu: [np array] Residual vector
        :param Weight: [np array | None] Apriori weight

        :return weight: [np array] Weight for the inversion
        """

        r_std = residu / (stats.median_abs_deviation(residu) / 0.6745)
        if Weight is not None:  # The weight is a combination of apriori weight and the studentized residual
            # Weight = Weight / (stats.median_abs_deviation(Weight) / 0.6745)
            weight = Weight * TukeyBiweight(r_std, 4.685)
        else:
            weight = TukeyBiweight((r_std), 4.685)

        return weight

    weightx = weightf(compute_residual(A, data[:, 0], result_dx), Weight[0])
    weighty = weightf(compute_residual(A, data[:, 1], result_dy), Weight[1])

    if A.shape[0] < A.shape[1]:
        if verbose:
            print(
                f"[Inversion] If the number of row is lower than the number of columns, the results are not updated {A.shape}"
            )
        return result_dx, result_dy, weightx, weighty, None, None

    if regu == "directionxy":
        if solver == "LSMR_ini":
            result_dx, result_dy, residu_normx, residu_normy = inversion_two_components(
                A,
                dates_range,
                0,
                data,
                solver,
                np.concatenate([weightx, weighty]),
                mu,
                coef=coef,
                ini=np.concatenate([result_dx, result_dy]),
            )
        else:
            result_dx, result_dy, residu_normx, residu_normy = inversion_two_components(
                A, dates_range, 0, data, solver, np.concatenate([weightx, weighty]), mu, coef=coef
            )

    elif solver == "LSMR_ini":
        if ini == None:  # Initialization with the result from the previous inversion
            result_dx, residu_normx = inversion_one_component(
                A,
                dates_range,
                0,
                data,
                solver,
                weightx,
                mu,
                coef=coef,
                ini=result_dx,
                result_quality=result_quality,
                regu=regu,
                accel=accel,
                linear_operator=linear_operator,
            )
            result_dy, residu_normy = inversion_one_component(
                A,
                dates_range,
                1,
                data,
                solver,
                weighty,
                mu,
                coef=coef,
                ini=result_dy,
                result_quality=result_quality,
                regu=regu,
                accel=accel,
                linear_operator=linear_operator,
            )
        else:  # Initialization with the list ini, which can be a moving average
            result_dx, residu_normx = inversion_one_component(
                A,
                dates_range,
                0,
                data,
                solver,
                weightx,
                mu,
                coef=coef,
                ini=ini[0],
                result_quality=result_quality,
                regu=regu,
                accel=accel,
                linear_operator=linear_operator,
            )
            result_dy, residu_normy = inversion_one_component(
                A,
                dates_range,
                1,
                data,
                solver,
                weighty,
                mu,
                coef=coef,
                ini=ini[1],
                result_quality=result_quality,
                regu=regu,
                accel=accel,
                linear_operator=linear_operator,
            )

    else:  # No initialization
        result_dx, residu_normx = inversion_one_component(
            A,
            dates_range,
            0,
            data,
            solver,
            weightx,
            mu,
            coef=coef,
            result_quality=result_quality,
            regu=regu,
            accel=accel,
            linear_operator=linear_operator,
        )
        result_dy, residu_normy = inversion_one_component(
            A,
            dates_range,
            1,
            data,
            solver,
            weighty,
            mu,
            coef=coef,
            result_quality=result_quality,
            regu=regu,
            accel=accel,
            linear_operator=linear_operator,
        )

    return result_dx, result_dy, weightx, weighty, residu_normx, residu_normy


def inversion_core(
    data: list,
    i: float | int,
    j: float | int,
    dates_range: np.ndarray | None = None,
    solver: str = "LSMR",
    regu: int | str = 1,
    coef: int = 100,
    apriori_weight: bool = False,
    iteration: bool = True,
    threshold_it: float = 0.1,
    unit: int = 365,
    conf: bool = False,
    mean: list | None = None,
    detect_temporal_decorrelation: bool = True,
    linear_operator: bool = False,
    result_quality: list | str | None = None,
    nb_max_iteration: int = 10,
    apriori_weight_in_second_iteration: bool = False,
    visual: bool = True,
    verbose: bool = False,
) -> (np.ndarray, pd.DataFrame, pd.DataFrame):  # type: ignore

    """
    Computes A in AX = Y and does the inversion using a given solver.

    :param data: [list] --- An array where each line is (date1, date2, other elements ) for which a velocity is computed (correspond to the original displacements)
    :params i, j: [float | int] --- Coordinates of the point in pixel
    :param dates_range: [np array | None] [default is None] --- List of np.datetime64 [D], dates of the estimated displacement in X with an irregular temporal sampling (ILF)
    :param solver: [str] [default is 'LSMR'] --- Solver of the inversion: 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
    :param regu: [int | str] [default is 1] --- Type of regularization
    :param coef: [int] [default is 100] --- Coef of Tikhonov regularisation
    :param apriori_weight: [bool] [default is False] --- If True use of aprori weight, based on the provided observation errors
    :param iteration: [bool] [default is True] --- If True, use of iterations
    :param threshold_it: [float] [default is 0.1] --- Threshold to test the stability of the results between each iteration, use to stop the process
    :param unit: [int] [default is 365] --- 1 for m/d, 365 for m/y
    :param conf: [bool] [default is False] --- If True means that the error corresponds to confidence intervals between 0 and 1, otherwise it corresponds to errors in m/y or m/d
    :param mean: [list | None] [default is None] --- Apriori on the average
    :param detect_temporal_decorrelation: [bool] [default is True] --- If True the first inversion is solved using only velocity observations with small temporal baselines, to detect temporal decorelation
    :param linear_operator: [bool] [default is False] --- If linear operator, the inversion is performed using a linear operator (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param nb_max_iteration: [int] [default is 10] --- Maximum number of iterations
    :param apriori_weight_in_second_iteration: [bool] [default is False] --- it True use the error to weight each of the iterations, if not use it only in the first iteration
    :param visual: [bool] [default is True] --- Keep the weights for future plots
    :param verbose: [bool] [default is False] --- Print information along the way

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

        if dates_range is None:
            dates_range = construction_dates_range_np(
                data_dates
            )  # 652 µs ± 3.24 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        ####  Build A (design matrix in AX = Y)
        if not linear_operator:
            A = construction_a_lf(
                data_dates, dates_range
            )  # 1.93 ms ± 219 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
        else:  # use a linear operator to solve the inversion
            linear_operator = class_linear_operator()
            linear_operator.load(
                find_date_obs(data_dates[:, :2], dates_range), dates_range, coef
            )  # load parameter of the linear operator
            A = sp.linalg.LinearOperator(
                (data_values.shape[0], len(dates_range) - 1),
                matvec=linear_operator.matvec,
                rmatvec=linear_operator.rmatvec,
            )  # build A
            mu = None

        # Set a weight of 0, for large temporal baseline in the first inversion
        # 115 µs ± 1.2 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
        weight_temporal_decorrelation = (
            np.where(data_values[:, 4] > 200, 0, 1) if detect_temporal_decorrelation else None
        )
        # First weight of the inversion
        Weightx = weight_for_inversion(
            weight_origine=apriori_weight,
            conf=conf,
            data=data_values,
            pos=2,
            inside_Tukey=False,
            temporal_decorrelation=weight_temporal_decorrelation,
        )
        Weighty = weight_for_inversion(
            weight_origine=apriori_weight,
            conf=conf,
            data=data_values,
            pos=3,
            inside_Tukey=False,
            temporal_decorrelation=weight_temporal_decorrelation,
        )
        del weight_temporal_decorrelation
        if not visual and not apriori_weight_in_second_iteration:
            data_values = np.delete(data_values, [2, 3], 1)  # Delete quality indicator, which are not needed anymore
        # Compute regularisation matrix
        # 493 µs ± 2.35 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
        if not linear_operator:
            if regu == "directionxy":
                # Constrain according to the vectorial product, the magnitude of the vector corresponds to mean2, the magnitude of a rolling mean
                mu = mu_regularisation(regu, A, dates_range, ini=mean)
            else:
                mu = mu_regularisation(regu, A, dates_range, ini=mean)

        ##  Initialisation (depending on apriori and solver)
        # # Apriori on acceleration (following)
        if regu == "1accelnotnull":
            accel = [
                np.diff(mean[0]),
                np.diff(mean[1]),
            ]  # compute acceleration based on the moving average, computing using a given kernel
            mean_ini = [
                np.multiply(mean[i], np.diff(dates_range) / np.timedelta64(1, "D")) for i in range(len(mean))
            ]  # compute what should be the displacement in X according to the moving average, computing using a given kernel

        elif (
            mean is not None and solver == "LSMR_ini"
        ):  # initialization is set according the average of the whole time series
            mean_ini = [
                np.multiply(mean[i], np.diff(dates_range) / np.timedelta64(1, "D") / unit) for i in range(len(mean))
            ]
            accel = None
        else:
            mean_ini = None
            accel = None

        ##  Inversion
        # 87.5 ms ± 668 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        if regu == "directionxy":
            result_dx, result_dy, residu_normx, residu_normy = inversion_two_components(
                A, dates_range, 0, data_values, solver, np.concatenate([Weightx, Weighty]), mu, coef=coef, ini=mean_ini
            )
        else:
            result_dx, residu_normx = inversion_one_component(
                A,
                dates_range,
                0,
                data_values,
                solver,
                Weightx,
                mu,
                coef=coef,
                ini=mean_ini,
                result_quality=None,
                regu=regu,
                linear_operator=linear_operator,
                accel=accel,
            )
            result_dy, residu_normy = inversion_one_component(
                A,
                dates_range,
                1,
                data_values,
                solver,
                Weighty,
                mu,
                coef=coef,
                ini=mean_ini,
                result_quality=None,
                regu=regu,
                linear_operator=linear_operator,
                accel=accel,
            )

        if not visual:
            del Weighty, Weightx

        if regu == "directionxy":
            mu = mu_regularisation(regu, A, dates_range, ini=[mean[0], mean[1], result_dx, result_dy])
            # coef = coef * 1000

        # Second Iteration
        # 1.11 s ± 17.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        if iteration:
            if (
                apriori_weight_in_second_iteration
            ) and apriori_weight:  # use apriori weight based on the error or quality indicator, Tukeybiweight(error/MAD(error)/ 0.6745)
                Weightx2 = weight_for_inversion(
                    weight_origine=apriori_weight, conf=conf, data=data_values, pos=2, inside_Tukey=False
                )
                Weighty2 = weight_for_inversion(
                    weight_origine=apriori_weight, conf=conf, data=data_values, pos=3, inside_Tukey=False
                )
            else:
                Weightx2, Weighty2 = None, None

            result_dx_i, result_dy_i, weight_2x, weight_2y, residu_normx, residu_normy = inversion_iteration(
                data_values,
                A,
                dates_range,
                solver,
                coef,
                [Weightx2, Weighty2],
                result_dx,
                result_dy,
                mu=mu,
                verbose=verbose,
                regu=regu,
                linear_operator=linear_operator,
                ini=None,
                accel=accel,
                result_quality=result_quality,
            )
            # Continue to iterate until the difference between two results is lower than threshold_it or the number of iteration larger than 10
            i = 2
            while (
                np.mean(abs(result_dx_i - result_dx)) > threshold_it
                or np.mean(abs(result_dy_i - result_dy)) > threshold_it
            ) and i < nb_max_iteration:
                result_dx = result_dx_i
                result_dy = result_dy_i
                result_dx_i, result_dy_i, weight_ix, weight_iy, residu_normx, residu_normy = inversion_iteration(
                    data_values,
                    A,
                    dates_range,
                    solver,
                    coef,
                    [Weightx2, Weighty2],
                    result_dx,
                    result_dy,
                    mu,
                    verbose=verbose,
                    regu=regu,
                    linear_operator=linear_operator,
                    ini=None,
                    accel=accel,
                    result_quality=result_quality,
                )

                i += 1

                if verbose:
                    print(
                        "[Inversion] ",
                        i,
                        "dx",
                        np.mean(abs(result_dx_i - result_dx)),
                        "dy",
                        np.mean(abs(result_dy_i - result_dy)),
                    )

            if verbose:
                print("[Inversion] End loop", i, np.mean(abs(result_dy_i - result_dy)))
                print("[Inversion] Nb iteration", i)

            if i == 2:
                weight_iy = weight_2y
                weight_ix = weight_2x

            del result_dx, result_dy
            if not visual and not "Error_propagation" in result_quality:
                del data_values, data_dates

        else:  # If not iteration
            result_dy_i = result_dy
            result_dx_i = result_dx

        if np.isnan(result_dx_i).all():  # no results
            return None, None, None

        # compute the number of observations which have contributed to each estimated displacement
        if result_quality is not None and "X_contribution" in result_quality:
            xcount_x = A.T.dot(weight_ix)
            xcount_y = A.T.dot(weight_iy)
        else:
            xcount_x = xcount_y = np.ones(result_dx_i.shape[0])

        # propagate the error
        # TODO terminate propgation of errors
        if result_quality is not None and "Error_propagation" in result_quality:

            def Prop_weight(weight, Residu):
                W = np.diag(weight_ix.astype("float32"))
                FTWF = F.T * W @ F
                N = np.linalg.inv(FTWF + coef * mu.T @ mu)
                Prop_weight = N @ F.T @ W @ F @ N
                sigma0_weight = np.sum(Residu**2 * weight) / (F.shape[0] - F.shape[1] + 1)
                prop_wieght_diag = np.diag(Prop_weight)
                return prop_wieght_diag, sigma0_weight

            # if not 'GCV' in result_quality:
            F = sp.csc_matrix(A, dtype="float32")
            Residux = data_values[:, 0] - F @ result_dx_i  # has a normal distribution
            prop_wieght_diagx, sigma0_weightx = Prop_weight(weight_ix, Residux)
            Residuy = data_values[:, 1] - F @ result_dy_i  # has a normal distribution
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
            Residux = data_values[:, 0] - A.dot(result_dx_i)
            Residuy = data_values[:, 1] - A.dot(result_dy_i)
            dataf = pd.DataFrame(
                {
                    "date1": data_dates[:, 0],
                    "date2": data_dates[:, 1],
                    "vx": vx,
                    "vy": vy,
                    "errorx": data_values[:, 2],
                    "errory": data_values[:, 3],
                    "weightinix": Weightx,
                    "weightiniy": Weighty,
                    "weightlastx": weight_ix,
                    "weightlasty": weight_iy,
                    "residux": Residux,
                    "residuy": Residuy,
                    "sensor": data_str[:, 0],
                    "author": data_str[:, 1],
                }
            )
            if (
                residu_normx is not None
            ):  # save the L2-norm from the last inversion, of the term AXY and the regularization term for the x- and y-component
                NormR = np.zeros(data_values.shape[0])
                NormR[:4] = np.hstack(
                    [residu_normx, residu_normy]
                )  # the order is: AXY and regularization term L2-norm for x-component, and AXY and regularization term L2-norm for y-component
                dataf["NormR"] = NormR
                del NormR
        else:
            dataf, A = None, None

    else:  # If there is no data over this pixel
        if verbose:
            print(f"[Inversion] NO DATA TO INVERSE AT POINT {i, j}")
        return None, None, None

    # pandas dataframe with the saved results
    result = pd.DataFrame(
        {
            "date1": dates_range[:-1],
            "date2": dates_range[1:],
            "result_dx": result_dx_i,
            "result_dy": result_dy_i,
            "xcount_x": xcount_x,
            "xcount_y": xcount_y,
        }
    )
    if residu_normx is not None:  # add the norm of the residual
        normr = np.zeros(result.shape[0])
        if normr.shape[0] > 3:
            normr[:4] = np.hstack([residu_normx, residu_normy])
        else:
            normr[: normr.shape[0]] = np.full(normr.shape[0], np.nan)
        result["NormR"] = normr
        del normr
    if result_quality is not None:  # add the error propagation
        if "Error_propagation" in result_quality:
            result["error_x"] = prop_wieght_diagx
            result["error_y"] = prop_wieght_diagy
            sigma = np.zeros(result.shape[0])
            sigma[:2] = np.hstack([sigma0_weightx, sigma0_weighty])
            result["sigma0"] = sigma

    return A, result, dataf


# %% ======================================================================== #
#                               INTERPOLATION                                 #
# =========================================================================%% #


def interpolation_core(
    result: pd.DataFrame,
    interval_output: int,
    option_interpol: str = "spline",
    first_date_interpol: np.datetime64 | str | None = None,
    last_date_interpol: np.datetime64 | str | None = None,
    unit: int = 365,
    redundancy: int | None = None,
    result_quality: list | None = None,
):

    """
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

    :return dataf_lp: [pd dataframe] --- Result of the temporal interpolation
    """

    ##  Reconstruction of COMMON REF TIME SERIES, e.g. cumulative displacement time series
    dataf = reconstruct_common_ref(result)  # Build cumulative displacement time series
    if first_date_interpol is None:
        start_date = dataf["Ref_date"][0]  # First date at the considered pixel
    else:
        start_date = pd.to_datetime(first_date_interpol)

    x = np.array(
        (dataf["Second_date"] - np.datetime64(start_date)).dt.days
    )  # Number of days according to the start_date
    if len(x) <= 1 or (
        np.isin("spline", option_interpol) and len(x) <= 3
    ):  # It is not possible to interpolate, because too few estimation
        return pd.DataFrame(
            {
                "date1": [],
                "date2": [],
                "vx": [],
                "vy": [],
                "xcount_x": [],
                "xcount_y": [],
                "dz": [],
                "vz": [],
                "xcount_z": [],
                "NormR": [],
            }
        )

    # Compute the functions used to interpolate
    fdx, fdy, fdx_xcount, fdy_xcount, fdx_error, fdy_error = set_function_for_interpolation(
        option_interpol, x, dataf, result_quality
    )

    if redundancy is None:  # No redundancy between two interpolated velocity
        x_regu = np.arange(np.min(x) + (interval_output - np.min(x) % interval_output), np.max(x), interval_output)
    else:  # The overlap between two velocities corresponds to redundancy
        x_regu = np.arange(
            np.min(x) + (redundancy - np.min(x) % redundancy), np.max(x), redundancy
        )  # To make sure that the first element of x_regu is multiple of redundancy

    if len(x_regu) <= 1:  # No interpolation
        return pd.DataFrame(
            {
                "date1": [],
                "date2": [],
                "vx": [],
                "vy": [],
                "xcount_x": [],
                "xcount_y": [],
                "dz": [],
                "vz": [],
                "xcount_z": [],
                "NormR": [],
            }
        )

    ##  Reconstruct a time series with a given temporal sampling, and a given overlap
    step = interval_output if redundancy is None else int(interval_output / redundancy)
    if step >= len(x_regu):
        return pd.DataFrame(
            {
                "date1": [],
                "date2": [],
                "vx": [],
                "vy": [],
                "xcount_x": [],
                "xcount_y": [],
                "dz": [],
                "vz": [],
                "xcount_z": [],
                "NormR": [],
            }
        )

    x_shifted = x_regu[step:]
    dx = fdx(x_shifted) - fdx(
        x_regu[:-step]
    )  # Equivalent to [fdx(x_regu[i + step]) - fdx(x_regu[i]) for i in range(len(x_regu) - step)]
    dy = fdy(x_shifted) - fdy(
        x_regu[:-step]
    )  # Equivalent to [fdy(x_regu[i + step]) - fdy(x_regu[i]) for i in range(len(x_regu) - step)]
    if result_quality is not None:
        if "X_contribution" in result_quality:
            xcount_x = fdx_xcount(x_shifted) - fdx_xcount(x_regu[:-step])
            xcount_y = fdy_xcount(x_shifted) - fdy_xcount(x_regu[:-step])
        if "Error_propagation" in result_quality:
            error_x = fdx_error(x_shifted) - fdx_error(x_regu[:-step])
            error_y = fdy_error(x_shifted) - fdy_error(x_regu[:-step])
    vx = dx * unit / interval_output  # Convert to velocity in m/d or m/y
    vy = dy * unit / interval_output  # Convert to velocity in m/d or m/

    First_date = start_date + pd.to_timedelta(
        x_regu[:-step], unit="D"
    )  # Equivalent to [start_date + pd.Timedelta(x_regu[i], 'D') for i in range(len(x_regu) - step)]
    Second_date = start_date + pd.to_timedelta(x_shifted, unit="D")

    dataf_lp = pd.DataFrame({"date1": First_date, "date2": Second_date, "vx": vx, "vy": vy})
    if result_quality is not None:
        if "X_contribution" in result_quality:
            dataf_lp["xcount_x"] = xcount_x
            dataf_lp["xcount_y"] = xcount_y
        if "Error_propagation" in result_quality:
            dataf_lp["error_x"] = error_x * unit / interval_output
            dataf_lp["error_y"] = error_y * unit / interval_output
            dataf_lp["sigma0"] = np.concatenate([result["sigma0"][:2], np.full(dataf_lp.shape[0] - 2, np.nan)])
    del x_regu, First_date, Second_date, vx, vy

    # Fill with nan values if the first date of the cube which will be interpolated is lower than the first date interpolated for this pixel
    if first_date_interpol is not None and dataf_lp["date1"].iloc[0] > pd.Timestamp(first_date_interpol):
        first_date = np.arange(first_date_interpol, dataf_lp["date1"].iloc[0], np.timedelta64(redundancy, "D"))
        # dataf_lp = full_with_nan(dataf_lp, first_date=first_date,
        #                          second_date=first_date + np.timedelta64(interval_output, 'D'))
        nul_df = pd.DataFrame(
            {
                "date1": first_date,
                "date2": first_date + np.timedelta64(interval_output, "D"),
                "vx": np.full(len(first_date), np.nan),
                "vy": np.full(len(first_date), np.nan),
            }
        )
        if result_quality is not None:
            if "X_contribution" in result_quality:
                nul_df["xcount_x"] = np.full(len(first_date), np.nan)
                nul_df["xcount_y"] = np.full(len(first_date), np.nan)
            if "Error_propagation" in result_quality:
                nul_df["error_x"] = np.full(len(first_date), np.nan)
                nul_df["error_y"] = np.full(len(first_date), np.nan)
        dataf_lp = pd.concat([nul_df, dataf_lp], ignore_index=True)

    # Fill with nan values if the last date of the cube which will be interpolated is higher than the last date interpolated for this pixel
    if last_date_interpol is not None and dataf_lp["date2"].iloc[-1] < pd.Timestamp(last_date_interpol):
        first_date = np.arange(
            dataf_lp["date2"].iloc[-1] + np.timedelta64(redundancy, "D"),
            last_date_interpol + np.timedelta64(redundancy, "D"),
            np.timedelta64(redundancy, "D"),
        )
        nul_df = pd.DataFrame(
            {
                "date1": first_date - np.timedelta64(interval_output, "D"),
                "date2": first_date,
                "vx": np.full(len(first_date), np.nan),
                "vy": np.full(len(first_date), np.nan),
            }
        )
        dataf_lp = pd.concat([dataf_lp, nul_df], ignore_index=True)

    # print(dataf_lp.shape)
    # if dataf_lp.shape[0]!= 567:
    #     print('stop')
    return dataf_lp


def interpolation_to_data(
    result: pd.DataFrame,
    data: pd.DataFrame,
    option_interpol: str = "spline",
    unit: int = 365,
    result_quality: list | None = None,
):

    """
    Interpolate Irregular Leap Frog time series (result of an inversion) to the dates of given data (useful to compare
    TICOI results to a "ground truth").

    :param result: [pd dataframe] --- Leap frog displacement for x-component and y-component
    :param data: [pd dataframe] --- Ground truth data which the interpolation must fit along the temporal axis
    :param option_interpol: [str] [default is 'spline'] --- Type of interpolation, it can be 'spline', 'spline_smooth' or 'nearest'
    :param unit: [int] [default is 365] --- 1 for m/d, 365 for m/y
    :param result_quality: [list | str | None] [default is None] --- List which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    """

    ##  Reconstruction of COMMON REF TIME SERIES, e.g. cumulative displacement time series
    dataf = reconstruct_common_ref(result, result_quality)  # Build cumulative displacement time series
    start_date = dataf["Ref_date"][0]  # First date at the considered pixel
    x = np.array(
        (dataf["Second_date"] - np.datetime64(start_date)).dt.days
    )  # Number of days according to the start_date

    # Interpolation must be caried out in between the min and max date of the original data
    if data["date1"].min() < result["date2"].min() or data["date2"].max() > result["date2"].max():
        data = data[(data["date1"] > result["date2"].min()) & (data["date2"] < result["date2"].max())]

    # Ground truth first and second dates
    x_gt_date1 = np.array((data["date1"] - start_date).dt.days)
    x_gt_date2 = np.array((data["date2"] - start_date).dt.days)

    ##  Interpolate the displacements and convert to velocities
    # Compute the functions used to interpolate
    fdx, fdy, fdx_xcount, fdy_xcount, fdx_error, fdy_error = set_function_for_interpolation(
        option_interpol, x, dataf, result_quality
    )

    # Interpolation
    dx = fdx(x_gt_date2) - fdx(
        x_gt_date1
    )  # Equivalent to [fdx(x_regu[i + step]) - fdx(x_regu[i]) for i in range(len(x_regu) - step)]
    dy = fdy(x_gt_date2) - fdy(
        x_gt_date1
    )  # Equivalent to [fdy(x_regu[i + step]) - fdy(x_regu[i]) for i in range(len(x_regu) - step)]

    # conversion
    vx = dx * unit / data["temporal_baseline"]  # Convert to velocity in m/d or m/y
    vy = dy * unit / data["temporal_baseline"]  # Convert to velocity in m/d or m/y

    # Fill dataframe
    First_date = start_date + pd.to_timedelta(
        x_gt_date1, unit="D"
    )  # Equivalent to [start_date + pd.Timedelta(x_regu[i], 'D') for i in range(len(x_regu) - step)]
    Second_date = start_date + pd.to_timedelta(x_gt_date2, unit="D")
    data_dict = {"date1": First_date, "date2": Second_date, "vx": vx, "vy": vy}
    dataf_lp = pd.DataFrame(data_dict)

    return dataf_lp


# %% ======================================================================== #
#                               GLOBAL PROCESS                                #
# =========================================================================%% #


def process(
    cube: cube_data_class,
    i: float | int,
    j: float | int,
    path_save,
    solver: str = "LSMR",
    regu: int | str = 1,
    coef: int = 100,
    flag: xr.Dataset | None = None,
    apriori_weight: bool = False,
    returned: list | str = "interp",
    obs_filt: xr.Dataset | None = None,
    interpolation_load_pixel: str = "nearest",
    iteration: bool = True,
    interval_output: int = 1,
    first_date_interpol: np.datetime64 | None = None,
    last_date_interpol: np.datetime64 | None = None,
    proj="EPSG:4326",
    threshold_it: float = 0.1,
    conf: bool = True,
    option_interpol: str = "spline",
    redundancy: int | None = None,
    detect_temporal_decorrelation: bool = True,
    unit: int = 365,
    result_quality: list | str | None = None,
    nb_max_iteration: int = 10,
    delete_outliers: int | str | None = None,
    linear_operator: bool = False,
    visual: bool = False,
    verbose: bool = False,
):

    """
    :params i, j: [float | int] --- Coordinates of the point in pixel
    :param solver: [str] [default is 'LSMR'] --- Solver of the inversion: 'LSMR', 'LSMR_ini', 'LS', 'LSQR'
    :param regu: [int | str] [default is 1] --- Type of regularization
    :param coef: [int] [default is 100] --- Coef of Tikhonov regularisation
    :param flag: [xr dataset | None] [default is None] --- If not None, the values of the coefficient used for stable areas, surge glacier and non surge glacier
    :param apriori_weight: [bool] [default is False] --- If True use of aprori weight
    :param returned: [list | str] [default is 'interp'] --- What results must be returned ('raw', 'invert' and/or 'interp')
    :param obs_filt: [xr dataset | None] [default is None] --- Filtered dataset (e.g. rolling mean)
    :param interpolation_load_pixel: [str] [default is 'nearest'] --- Type of interpolation to load the previous pixel in the temporal interpolation ('nearest' or 'linear')
    :param iteration: [bool] [default is True] --- If True, use of iterations
    :param interval_output: [int] [default is 1] --- Temporal sampling of the leap frog time series
    :param first_date_interpol: [np.datetime64 | None] --- First date at which the time series are interpolated
    :param last_date_interpol: [np.datetime64 | None] --- Last date at which the time series are interpolated
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
    :param verbose: [bool] [default is False] --- Print information along the way

    :return dataf_list: [pd dataframe] Result of the temporal inversion + interpolation at point (i, j) if inversion was successful, an empty dataframe if not
    """

    returned_list = []

    # Loading data at pixel location
    data = cube.load_pixel(
        i,
        j,
        proj=proj,
        interp=interpolation_load_pixel,
        solver=solver,
        coef=coef,
        regu=regu,
        rolling_mean=obs_filt,
        flag=flag,
    )

    if "raw" in returned:  # return the raw data
        returned_list.append(data)

    if "invert" in returned or "interp" in returned:
        if flag is not None:  # set regu and coef for every flags
            regu, coef = data[3], data[4]

        # Inversion
        # TODO: to check that!
        if delete_outliers == "median_angle":
            conf = True  # Set conf to True, because the errors have been replaced by confidence indicators based on the cos of the angle between the vector of each observation and the median vector

        result = inversion_core(
            data[0],
            i,
            j,
            dates_range=data[2],
            solver=solver,
            coef=coef,
            apriori_weight=apriori_weight,
            unit=unit,
            conf=conf,
            regu=regu,
            mean=data[1],
            iteration=iteration,
            threshold_it=threshold_it,
            detect_temporal_decorrelation=detect_temporal_decorrelation,
            linear_operator=linear_operator,
            result_quality=result_quality,
            nb_max_iteration=nb_max_iteration,
            visual=visual,
            verbose=verbose,
        )

        if "invert" in returned:
            if result[1] is not None:
                returned_list.append(result[1])
            else:
                if result_quality is not None and "X_contribution" in result_quality:
                    variables = ["result_dx", "result_dy", "xcount_x", "xcount_y"]
                else:
                    variables = ["result_dx", "result_dy"]
                returned_list.append(pd.DataFrame({"date1": [], "date2": [], **{col: [] for col in variables}}))

        if "interp" in returned:
            # Interpolation
            if result[1] is not None:  # If inversion have been performed
                dataf_list = interpolation_core(
                    result[1],
                    interval_output,
                    option_interpol=option_interpol,
                    first_date_interpol=first_date_interpol,
                    last_date_interpol=last_date_interpol,
                    unit=unit,
                    redundancy=redundancy,
                    result_quality=result_quality,
                )

                if result_quality is not None and "Norm_residual" in result_quality:
                    dataf_list["NormR"] = result[1]["NormR"]  # Store norm of the residual from the inversion
                returned_list.append(dataf_list)
            else:
                if result_quality is not None and "Norm_residual" in result_quality:
                    returned_list.append(
                        pd.DataFrame(
                            {"date1": [], "date2": [], "vx": [], "vy": [], "xcount_x": [], "xcount_y": [], "NormR": []}
                        )
                    )
                else:
                    returned_list.append(
                        pd.DataFrame({"date1": [], "date2": [], "vx": [], "vy": [], "xcount_x": [], "xcount_y": []})
                    )

    if len(returned_list) == 1:
        return returned_list[0]
    return returned_list if len(returned_list) > 0 else None


def chunk_to_block(cube: cube_data_class, block_size: float = 1, verbose: bool = False):

    """
    Split a dataset in blocks of a given size (maximum).

    :param cube: [cube_data_class] --- Cube to be splited in blocks
    :param block_size: [float] [default is 1] --- Maximum size (in GB) of the blocks
    :param verbose: [bool] [default is False] --- Print information along the way

    :return blocks: [list] --- List of the boundaries of each blocks (x_start, x_end, y_start, y_end)
    """

    GB = 1073741824
    blocks = []
    if cube.ds.nbytes > block_size * GB:
        num_elements = np.prod([cube.ds.chunks[dim][0] for dim in cube.ds.chunks.keys()])
        chunk_bytes = num_elements * cube.ds["vx"].dtype.itemsize

        nchunks_block = int(block_size * GB // chunk_bytes)

        x_step = int(np.sqrt(nchunks_block))
        y_step = nchunks_block // x_step

        nblocks_x = int(np.ceil(len(cube.ds.chunks["x"]) / x_step))
        nblocks_y = int(np.ceil(len(cube.ds.chunks["y"]) / y_step))

        nblocks = nblocks_x * nblocks_y
        if verbose:
            print(
                f'[Block process] Divide into {nblocks} blocks\n   blocks size: {x_step * cube.ds.chunks["x"][0]} x {y_step * cube.ds.chunks["y"][0]}'
            )

        for i in range(nblocks_y):
            for j in range(nblocks_x):
                x_start = j * x_step * cube.ds.chunks["x"][0]
                y_start = i * y_step * cube.ds.chunks["y"][0]
                x_end = x_start + x_step * cube.ds.chunks["x"][0] if j != nblocks_x - 1 else cube.ds.dims["x"]
                y_end = y_start + y_step * cube.ds.chunks["y"][0] if i != nblocks_y - 1 else cube.ds.dims["y"]
                blocks.append([x_start, x_end, y_start, y_end])
    else:
        blocks.append([0, cube.ds.dims["x"], 0, cube.ds.dims["y"]])
        if verbose:
            print(f"[Block process] Cube size smaller than {block_size}GB, no need to divide")

    return blocks


def load_block(
    cube: cube_data_class, x_start: int, x_end: int, y_start: int, y_end: int, flag: xr.Dataset | None = None
):

    """
    Persist a block in memory, i.e. load it in a distributed way.

    :param cube: [cube_data_class] --- Cube splited in blocks
    :params x_start, x_end, y_start, y_end: [int] --- Boundaries of the block

    :return block: [cube_data_class] --- Sub-cube of cube according to the boundaries (block)
    :return duration: [float] --- Duration of the block loading
    """

    start = time.time()
    block = cube_data_class()
    block.ds = cube.ds.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
    block.ds = block.ds.persist()
    block.update_dimension()
    if flag is not None:
        block_flag = flag.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
        block_flag = block_flag.persist()
    else:
        block_flag = None
    duration = time.time() - start

    return block, block_flag, duration


def process_blocks_refine(
    cube: cube_data_class,
    nb_cpu: int = 8,
    block_size: float = 0.5,
    returned: list | str = "interp",
    preData_kwargs: dict = None,
    inversion_kwargs: dict = None,
    verbose: bool = False,
):

    """
    Separate the cube in several blocks computed synchronously one after the other by loading one block while the other is computed (with
    parallelization) in order to avoid memory overconsumption and kernel crashing, and benefit from smaller computation time.

    :param cube: [cube_data_class] --- Cube of raw data to be processed
    :param nb_cpu: [int] [default is 8] --- Number of processing unit to use for parallel processing
    :param block_size: [float] [default is 0.5] --- Maximum size of the blocks (in GB)
    :param returned: [list | str] [default is 'interp'] --- What results must be returned ('raw', 'invert' and/or 'interp')
    :param preData_kwargs: [dict] [default is None] --- Pre-processing parameters (see cube_data_classxr.filter_cube)
    :param inversion_kwargs: [dict] [default is None] --- Inversion (and interpolation) parameters (see core.process)
    :param verbose: [bool] [default is False] --- Print information along the way

    :return: [pd dataframe] Resulting estimated time series after inversion (and interpolation)
    """

    async def process_block(
        block: cube_data_class, returned: list | str = "interp", nb_cpu: int = 8, verbose: bool = False
    ):
        xy_values = itertools.product(block.ds["x"].values, block.ds["y"].values)
        # Return only raw data => no need to filter the cube
        if "raw" in returned and (type(returned) == str or len(returned) == 1):  # Only load the raw data
            xy_values_tqdm = tqdm(xy_values, total=(block.nx * block.ny))
            result_block = Parallel(n_jobs=nb_cpu, verbose=0)(
                delayed(block.load_pixel)(
                    i,
                    j,
                    proj=inversion_kwargs["proj"],
                    interp=inversion_kwargs["interpolation_load_pixel"],
                    solver=inversion_kwargs["solver"],
                    regu=inversion_kwargs["regu"],
                    rolling_mean=None,
                    visual=inversion_kwargs["visual"],
                )
                for i, j in xy_values_tqdm
            )
            return result_block

        # Filter the cube
        obs_filt, flag_block = block.filter_cube(**preData_kwargs)
        if isinstance(inversion_kwargs, dict):
            inversion_kwargs.update({"flag": flag_block})

        # There is no data on the whole block (masked data)
        if obs_filt is None and "interp" in returned:
            if inversion_kwargs["result_quality"] is not None and "Norm_residual" in inversion_kwargs["result_quality"]:
                return [
                    pd.DataFrame(
                        {"date1": [], "date2": [], "vx": [], "vy": [], "xcount_x": [], "xcount_y": [], "NormR": []}
                    )
                ]
            else:
                return [
                    pd.DataFrame(
                        {"First_date": [], "Second_date": [], "vx": [], "vy": [], "xcount_x": [], "xcount_y": []}
                    )
                ]

        xy_values_tqdm = tqdm(xy_values, total=(obs_filt["x"].shape[0] * obs_filt["y"].shape[0]))
        result_block = Parallel(n_jobs=nb_cpu, verbose=0)(
            delayed(process)(block, i, j, obs_filt=obs_filt, returned=returned, **inversion_kwargs)
            for i, j in xy_values_tqdm
        )

        return result_block

    async def process_blocks_main(cube, nb_cpu=8, block_size=0.5, returned="interp", verbose=False):
        if isinstance(preData_kwargs, dict) and "flag" in preData_kwargs.keys():
            flag = preData_kwargs["flag"]
            if flag is not None:
                flag = cube.create_flag(flag)
        else:
            flag = None

        blocks = chunk_to_block(cube, block_size=block_size, verbose=True)  # Split the cube in smaller blocks

        dataf_list = [None] * (cube.nx * cube.ny)

        loop = asyncio.get_event_loop()
        for n in range(len(blocks)):
            print(f"[Block process] Processing block {n+1}/{len(blocks)}")

            # Load the first block and start the loop
            if n == 0:
                x_start, x_end, y_start, y_end = blocks[0]
                future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end, flag)

            block, block_flag, duration = await future
            print(f"Block {n+1} loaded in {duration:.2f} s")

            if n < len(blocks) - 1:
                # Load the next block while processing the current block
                x_start, x_end, y_start, y_end = blocks[n + 1]
                future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end, flag)

            # need to change the flag back...
            if flag is not None:
                preData_kwargs.update({"flag": block_flag})

            block_result = await process_block(
                block, returned=returned, nb_cpu=nb_cpu, verbose=verbose
            )  # Process TICOI

            # Transform to list
            for i in range(len(block_result)):
                row = i % block.ny + blocks[n][2]
                col = np.floor(i / block.ny) + blocks[n][0]
                idx = int(col * cube.ny + row)

                dataf_list[idx] = block_result[i]

            del block_result, block

        if isinstance(returned, list) and len(returned) > 1:
            dataf_list = {returned[r]: [dataf_list[i][r] for i in range(len(dataf_list))] for r in range(len(returned))}

        return dataf_list

    # /!\ The use of asyncio can cause problems when the code is launched from an IDE if it has its own event loop
    # (leads to RuntimeError), you must launch it in an external terminal (IDEs generally offer this option)
    return asyncio.run(
        process_blocks_main(cube, nb_cpu=nb_cpu, block_size=block_size, returned=returned, verbose=verbose)
    )


# %% ======================================================================== #
#                               VISUALISATION                                 #
# =========================================================================%% #


def visualization_core(
    list_dataf: pd.DataFrame,
    option_visual: List,
    save: bool = False,
    show: bool = True,
    path_save: Optional[str] = None,
    A: Optional[np.array] = None,
    log_scale: bool = False,
    cmap: str = "rainbow",
    colors: List[str] = ["blueviolet", "orange"],
    figsize: tuple[int, int] = (10, 6),
):

    r"""
    Visualization function for the output of pixel_ticoi
    /!\ Many figures can be plotted

    :param list_dataf: [pd.DataFrame] --- cube dataset
    :param option_visual: [list] --- list of options for visualization
    :param save:[bool] [default is False]  --- if True, save the figures
    :param show: [bool] [default is True]  --- if True, show the figures
    :param path_save: [str|None] [default is None] --- path where to save the figures
    :param A: [np.array] [default is None]  --- design matrix
    :param log_scale: [bool] [default is False]  ---  if True, plot the figures into log scale
    :param cmap: [str] [default is 'rainbow''] --- color map used in the plots
    :param colors: [list of str] [default is ['blueviolet', 'orange']] --- List of colors to used for plotting the time series
    :param figsize: tuple[int, int] [default is (10,6)] --- Size of the figures
    """

    pixel_object = pixel_class()
    pixel_object.load(
        list_dataf, save=save, show=show, A=A, path_save=path_save, figsize=figsize, type_data=["obs", "invert"]
    )

    dico_visual = {
        "obs_xy": (lambda pix: pix.plot_vx_vy(color=colors[0], type_data="obs")),
        "obs_magnitude": (lambda pix: pix.plot_vv(color=colors[0], type_data="obs")),
        "obs_vxvy_quality": (lambda pix: pix.plot_vx_vy_quality(cmap=cmap, type_data="obs")),
        "invertxy_overlaid": (lambda pix: pix.plot_vx_vy_overlaid(colors=colors)),
        "invertvv_overlaid": (lambda pix: pix.plot_vv_overlaid(colors=colors)),
        "residuals": (lambda pix: pix.plot_residuals(log_scale=log_scale)),
        "xcount_xy": (lambda pix: pix.plot_xcount_vx_vy(cmap=cmap)),
        "xcount_vv": (lambda pix: pix.plot_xcount_vv(cmap=cmap)),
        "invert_weight": (lambda pix: pix.plot_weights_inversion()),
    }

    for option in option_visual:
        if option in dico_visual.keys():
            dico_visual[option](pixel_object)


def save_cube_parameters(
    cube: "ticoi.cube_data_classxr.cube_data_class",
    load_kwargs: dict,
    preData_kwargs: dict,
    inversion_kwargs: dict,
    returned: list | None = None,
) -> (str, str):
    """

    :param cube: [cube_data_class] --- cube dataset
    :param load_kwargs: [dict] --- parameters used to load the cube
    :param prep_kwargs: [dict] --- parameters used to pre the cube
    :param inversion_kwargs: [dict] --- parameters used to load the cube
    :return:
    """
    sensor_array = np.unique(cube.ds["sensor"])
    sensor_strings = [str(sensor) for sensor in sensor_array]
    sensor = ", ".join(sensor_strings)

    source = f"Temporal inversion on cube {cube.filename} using TICOI"
    source += (
        f' with a selection of dates among {load_kwargs["pick_date"]},'
        if load_kwargs["pick_date"] is not None
        else "" + f' with a selection of the temporal baselines among {load_kwargs["pick_temp_bas"]}'
        if load_kwargs["pick_temp_bas"] is not None
        else ("" + f' with a subset of {load_kwargs["subset"]}')
        if load_kwargs["subset"] is not None
        else ""
    )

    if inversion_kwargs["apriori_weight"]:
        source += " and apriori weight"
    source += f'. The regularisation coefficient is {inversion_kwargs["coef"]}.'
    if "interp" in returned:
        source += f'The interpolation method used is {inversion_kwargs["option_interpol"]}.'
        source += f'The interpolation baseline is {inversion_kwargs["interval_output"]} days.'
        source += f'The temporal spacing (redundancy) is {inversion_kwargs["redundancy"]} days.'

    source += f"The preparation are argument are: {preData_kwargs}"
    return source, sensor
