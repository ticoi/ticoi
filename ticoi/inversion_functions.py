'''
Auxillary functions to process the temporal inversion.

Author : Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import numpy as np
import math as m
import scipy.sparse as sp
from scipy.linalg import pinv, inv
from scipy import optimize
from numba import jit
import scipy.linalg as la
import scipy.optimize as opt


def is_convex(A):
    """
    Check if the dot product function A.dot(X) is convex.

    Parameters:
        A (numpy.ndarray): The matrix A in the dot product function.

    Returns:
        bool: True if the function is convex, False otherwise.
    """
    hessian_matrix = A.T @ A  # Compute the Hessian matrix
    return np.all(np.linalg.eigvals(hessian_matrix) >= 0)


@jit(nopython=True)
def average_absolute_deviation(data):
    ''' Computes the Average Absolute Deviation (AAD). Used when the Median Absolute Deviation (MAD) is equal to 0. '''
    return np.mean(np.absolute(data - np.mean(data)))


@jit(nopython=True)
def find_date_obs(data, dates_range):
    '''
    Finds the index in dates_range corresponding to each first and last date in data
    :param data: An array where each line is (date1, date2, other elements ) for which a velocity is computed (correspond to the original displacements)
    :param dates_range: list of np.datetime64 [D], dates of the estimated displacement in X with an irregular temporal sampling (ILF)
    :return:
    '''
    date1_indices = np.searchsorted(dates_range, data[:, 0])
    date2_indices = np.searchsorted(dates_range, data[:, 1]) - 1
    return np.column_stack((date1_indices, date2_indices))


@jit(nopython=True)
def matvecregu1_numba(X, Y, identification_obs, delta, coef, weight):
    for j in range(len(identification_obs)):
        Y[j] = np.sum(X[identification_obs[j][0]:identification_obs[j][1] + 1]) * weight[j]
    Y[len(identification_obs):len(identification_obs) + len(X) - 1] = np.diff(X / delta) * coef
    return Y


@jit(nopython=True)
def matvec_numba(X, Y, identification_obs):
    for j in range(len(identification_obs)):
        Y[j] = np.sum(X[identification_obs[j][0]:identification_obs[j][1] + 1])
    return Y


@jit(nopython=True)
def rmatvecregu1_numba(X, Y, identification_obs, coef, delta, weight):
    for j in range(len(identification_obs)):
        X[identification_obs[j][0]:identification_obs[j][1] + 1] += Y[j] * weight[j]
    X[0] -= Y[len(identification_obs)] / delta[0] * coef
    for j in range(len(identification_obs) + 1, len(identification_obs) + len(X) - 1):
        X[j - len(identification_obs)] += (Y[j - 1] - Y[j]) / delta[j - len(identification_obs)] * coef
    X[len(X) - 1] += Y[len(identification_obs) + len(X) - 2] / delta[len(X) - 1] * coef
    return X


@jit(nopython=True)
def rmatvecA_numba(X, Y, identification_obs):
    for j in range(len(identification_obs)):
        X[identification_obs[j][0]:identification_obs[j][1] + 1] += Y[j]
    return X


class class_inversion:
    def __init__(self):
        self.X_length = [] #length of the estimated velocity time-series
        self.delta = [] #temporal baseline of the estimated velocity time-series
        self.identification_obs = {}
        self.coef = 50 #coefficient of the regularization

    def load(self, identification_obs, dates_range, coef):
        self.__init__()
        self.X_length = len(dates_range) - 1
        self.delta = (np.diff(dates_range) / np.timedelta64(1, 'D'))
        self.identification_obs = identification_obs
        self.identification_obs_original = identification_obs
        self.coef = coef

    def update_from_weight(self, Y, Weight):
        '''
        Updates the vector Y, the vector Weight according to the Weight
        If some weight are 0, delete the observations
        :param Y:
        :param Weight:
        :return:
        '''
        Y = Y[Weight != 0]
        self.identification_obs = self.identification_obs_original[Weight != 0]
        self.Weight = Weight[Weight != 0]
        return Y

    def matvecregu1(self,X):
        """
        function to go from X to Y, corresponds to A
        Regularisation of first order Tikhonov (minimization of the acceleration), with an apriori or not, the equation is weighted by self.Weight
        :param X: np.array, estimated displacements
        :return: np.array, observed displacements
        """
        Y = np.zeros(len(self.identification_obs) + len(X) - 1)
        Y = matvecregu1_numba(X, Y, self.identification_obs, self.delta, self.coef, self.Weight)#call numba, to make the computation faster
        return Y

    def rmatvecregu1(self,Y):
        """
        function to go from Y to X, corresponds to A.T
         Regularisation of first order Tikhonov (minimization of the acceleration), with an apriori or not, the equation is weighted by self.Weight
        :param Y: np.array, observed displacements
        :return: np.array, estimated displacements
        """
        X = np.zeros((self.X_length))
        X = rmatvecregu1_numba(X, Y, self.identification_obs, self.coef, self.delta, self.Weight)#call numba, to make the computation faster
        return X

    def matvec(self, X):  # No regularization, no weight
        """
        function to go from X to Y, corresponds to A
        No regularization, no weight
        :param X: np.array, estimated displacements
        :return: np.array, observed displacements
          """
        Y = np.zeros(len(self.identification_obs_original))
        Y = matvec_numba(X, Y, self.identification_obs_original)
        return Y

    def rmatvec(self, Y):  # No regularization, no weight
        """
        function to go from X to Y, corresponds to A
        No regularization, no weight
        :param Y: np.array, observed displacements
        :return: np.array, estimated displacements
          """
        X = np.zeros((self.X_length))
        X = rmatvecA_numba(X, Y, self.identification_obs_original)
        return X


def Construction_dates_range_np(data):
    """
    Construction of the dates of the estimated displacement in X with an irregular temporal sampling (ILF)
    :param data: np.ndarray, an array where each line is (date1, date2, other elements) for which a velocity have been mesured
    :return: np.ndarray, the dates of the estimated displacement in X
    """

    dates = np.concatenate([data[:, 0], data[:, 1]])  # concatante date1 and date2
    dates = np.unique(dates)  # remove duplicates
    dates = np.sort(dates)  # Sort the dates
    return dates


@jit(nopython=True)  # use numba
def Construction_A_LP(dates, dates_range):
    """
    Construction of the design matix A in the formulation AX = Y
    
    :param dates: np array, where each line is (date1, date2) for which a velocity is computed (it corresponds to the original displacements)
    :param dates_range: list, dates of estimated displacemements in X
    
    :return: The design matrix A which represent the temporal closure of the displacement measurement network
    """
    # Search at wich indice in dates_range is stored each date in dates
    date1_indices = np.searchsorted(dates_range, dates[:, 0])
    date2_indices = np.searchsorted(dates_range, dates[:, 1]) - 1

    A = np.zeros((dates.shape[0], dates_range[1:].shape[0]), dtype='int32')
    for y in range(dates.shape[0]):
        A[y, date1_indices[y]:date2_indices[y] + 1] = 1

    return A


def Inversion_A_LP(A, dates_range, v_pos, data, solver, Weight, mu, coef=1, ini=None, result_quality=None, regu=1, accel=None,
                   linear_operator=None,
                   verbose=False):
    '''
    Invert the system AX = Y for one component of the velocity, using a given solver
    
    :param A: Matrix of the temporal inversion system AX = Y
    :param dates_range: An array with all the dates included in data, list (dates of X)
    :param v_pos: Position of the v variable within data
    :param data: An array where each line is (date1, date2, other elements) for which a velocity is computed (Y)
    :param solver: 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
    :param Weight: Weight, =1 for Ordinary Least Square
    :param mu: regularization matrix
    :param coef: Coef of Tikhonov regularisation
    :param ini: np array, Initialization of the inversion
    :param: result_quality: None or list of str, which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param regu : str, type of regularization
    :param accel: list or None, apriori on the acceleration
    :param linear_operator: linear operator or None

    :return X: The ILF temporal inversion of AX = Y using the given solver
    :return residu_norm: Norm of the residu (when showing the L curve)
    '''

    # Total process : about 50ms
    if verbose:  # Matrix A properties
        if A.shape[0] < A.shape[1]:  # System is under-determined
            print('under-determined')
        elif A.shape[0] >= A.shape[1]:  # System is over-determined
            print('over-determined')
        if np.linalg.matrix_rank(A) < A.shape[1]:  # System is ill-posed
            print('ill posed')
            print('rank A = {}'.format(np.linalg.matrix_rank(A)))

    v = data[:, v_pos]

    if regu == '1accelnotnull':  # apriori on the acceleration
        D_regu = np.multiply(accel[v_pos - 2], coef)
    else:
        D_regu = np.zeros(mu.shape[0])

    if linear_operator is None:
        F_regu = np.multiply(coef, mu)
    else:
        v = linear_operator.update_from_weight(v, Weight)#update v, Weight,
        A_l = sp.linalg.LinearOperator((v.shape[0] + len(dates_range) - 2, len(dates_range) - 1),
                                       matvec=linear_operator.matvecregu1, rmatvec=linear_operator.rmatvecregu1)

    if solver == 'LSMR':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), F_regu]).astype('float32')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), D_regu]).astype('float32')
        F = sp.csc_matrix(F)  # column-scaling so that each column have the same euclidian norme (i.e. 1)
        X = sp.linalg.lsmr(F, D)[
            0]  # If atol or btol is None, a default value of 1.0e-6 will be used. Ideally, they should be estimates of the relative error in the entries of A and b respectively.

    elif solver == 'LSMR_ini':  # 50ms
        #16.7 ms ± 141 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        if not linear_operator:
            condi = (Weight != 0)
            W = Weight[condi]
            F = sp.csc_matrix(
                np.vstack([np.multiply(W[:, np.newaxis], A[condi]),
                           F_regu]))  # stack ax and regu, and remove rows with only 0
            if verbose: print("Is F convex?", is_convex(F.toarray()))
            D = np.hstack([np.multiply(W, v[condi]), D_regu])  # stack ax and regu, and remove rows with only
        if type(ini) == list:  # if rolling mean
            x0 = ini[v_pos - 2]
        elif ini.shape[0] == 2:  # if only the average of the entire time series
            x0 = np.full(len(dates_range) - 1, ini[v_pos - 2], dtype='float32')
        else:
            x0 = ini

        #24 ms ± 419 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        if not linear_operator:
            X = sp.linalg.lsmr(F, D, x0=x0)[0]
        else:
            X = sp.linalg.lsmr(A_l, np.concatenate([linear_operator.Weight * v, D_regu]), x0=x0)[0]

    elif solver == 'LS':  #136 ms ± 6.48 ms per loop (mean ± std. dev. of 7 runs, 10 loops each) #time consuming
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), F_regu]).astype('float32')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), D_regu]).astype('float32')
        X = np.linalg.lstsq(F, D, rcond=None)[0]

    elif solver == 'L1':  # solving using L1-norm, time consuming !
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), F_regu]).astype('float32')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), D_regu]).astype('float32')
        X = opt.minimize(lambda x: la.norm(D - F @ x, ord=1), np.zeros(F.shape[1]))

    elif solver == 'LSQR':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), F_regu]).astype('float32')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), D_regu]).astype('float32')
        F = sp.csc_matrix(F)
        X, istop, itn, r1norm = sp.linalg.lsqr(F, D)[:4]

    else:
        raise ValueError("Enter 'LSMR', 'LSMR_ini', 'LS', 'LSQR'")

    if result_quality is not None and 'Norm_residual' in result_quality:  # to show the L_curve
        R_lcurve = F.dot(X) - D  #50.7 µs ± 327 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
        residu_norm = [np.linalg.norm(R_lcurve[:np.multiply(Weight[Weight != 0], v[Weight != 0]).shape[0]], ord=2),
                       np.linalg.norm(R_lcurve[np.multiply(Weight[Weight != 0], v[Weight != 0]).shape[0]:] / coef,
                                      ord=2)]
    else:
        residu_norm = None

    return X, residu_norm


def Inversion_A_LPxydir(A, dates_range, v_pos, data, solver, Weight, mu, B=False, coef=1, ini=None,
                        born=None, regu='directionxy', accel=None, show_L_curve=False, verbose=False):
    '''
    Invert the system AX = Y for a single component velocity (amplitude) and for a given solver

    :param A: Matrix of the temporal inversion system AX=Y
    :param dates_range: An array with all the dates included in data, list
    :param v_pos: Position of the v variable within data
    :param data: An array where each line is (date1, date2, other elements) for which a velocity is computed
    :param solver: LS_regu, LS_SVD, LSQR or LSQR_ini
    :param coef: Coef of Tikhonov regularisation
    :param Weight: Weight, =1 for Ordinary Least Square
    
    :return result_dx, result_dy: Computed displacements along x and y axis
    :return residu_normx, residu_norm_y: Norm of the residu along x and y axis (when showing the L curve)
    '''

    if verbose:  # A properties
        if A.shape[0] < A.shape[1]:  # System is under-determined
            print('under-determined')
        elif A.shape[0] >= A.shape[1]:  # System is over-determined
            print('over-determined')
        if np.linalg.matrix_rank(A) < A.shape[1]:  # Systeme is ill-conditioned
            print('ill conditioned')
            print('rank A', np.linalg.matrix_rank(A))

    l_dates_range = len(dates_range)

    c = np.concatenate([A, np.zeros(A.shape)], axis=0)
    A = np.concatenate([c, np.concatenate([np.zeros(A.shape), A], axis=0)], axis=1)
    dates_range = np.concatenate([dates_range, dates_range])
    del c
    F_regu = np.multiply(coef, mu)
    # D_regu = np.zeros(mu.shape[0])
    D_regu = np.ones(mu.shape[0]) * coef

    v = np.concatenate([data[:, 2].T, data[:, 3].T])  # Concatenate vx and vy observations

    # del delta, mean

    if solver == 'LSMR':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), F_regu]).astype('float64')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), D_regu]).astype('float64')
        F = sp.csc_matrix(F)  # column-scaling so that each column have the same euclidian norme (i.e. 1)
        # If atol or btol is None, a default value of 1.0e-6 will be used. Ideally, they should be estimates of the relative error in the entries of A and b respectively.
        X = sp.linalg.lsmr(F, D)[0]

    elif solver == 'LSMR_ini':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), F_regu]).astype(
            'float64')  # stack ax and regu, and remove rows with only 0
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), D_regu]).astype(
            'float64')  # stack ax and regu, and remove rows with only

        if type(ini) == list:
            x0 = np.concatenate(ini)
        elif ini.shape[0] == 2:
            x0 = np.full(F.shape[1], ini[v_pos - 2], dtype='float64')
        else:
            x0 = ini
        # del ini

        F = sp.csc_matrix(F)
        X = sp.linalg.lsmr(F, D, x0=x0)[0]

    elif solver == 'LS':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), coef * mu]).astype('float64')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), np.zeros(mu.shape[0])]).astype('float64')
        X = np.linalg.lstsq(F, D, rcond=None)[0]

        # elif solver == 'L1':
        # import scipy.linalg as la
        # import scipy.optimize as opt
        # # sol = opt.minimize(lambda x: la.norm(D - F @ x,ord=1), np.zeros(F.shape[0]))

    elif solver == 'LS_bounded':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), coef * mu]).astype('float64')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), np.zeros(mu.shape[0])]).astype('float64')
        F = sp.csc_matrix(F)
        X = optimize.lsq_linear(F, D, (born[0][v_pos - 2], born[1][v_pos - 2]), lsq_solver='lsmr')['x']

    elif solver == 'LSQR' or solver == 'LSQR_ini':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), coef * mu]).astype('float64')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), np.zeros(mu.shape[0])]).astype('float64')
        F = sp.csc_matrix(F)  # column-scaling so that each column have the same euclidian norme (i.e. 1)
        X, istop, itn, r1norm = sp.linalg.lsqr(F, D)[:4]

    else:
        raise ValueError('Enter LS, LS_SVD,LSMR, LSQR or LSQR_ini')

    if show_L_curve:
        R_lcurve = F.dot(X) - D
        residu_norm = [np.linalg.norm(R_lcurve[:np.multiply(Weight[Weight != 0], v[Weight != 0]).shape[0]], ord=2),
                       np.linalg.norm(R_lcurve[np.multiply(Weight[Weight != 0], v[Weight != 0]).shape[0]:] / coef,
                                      ord=2)]
    else:
        residu_norm = None

    if residu_norm == None:
        return X[:X.shape[0] // 2], X[X.shape[0] // 2:], None, None
    else:
        return X[:X.shape[0] // 2], X[X.shape[0] // 2:], residu_norm[:X.shape[0] // 2], residu_norm[X.shape[0] // 2:]

def TukeyBiweight(z, c):
    '''
    Tukey's biweight function used at each iteration of the inversion to update the weights.
    
    :param z: Internally studentized residual
    :param c: Constant value
    
    :return weight: Selected weights
    '''
    subset = np.less_equal(abs(z), c)
    weight = np.ma.array(((1 - (z / c) ** 2) ** 2), mask=~subset)

    return weight.filled(0)


def hat_matrix(A, coef, mu, W=None):
    '''

    :param A: matrix of the temporal invserion system AX=Y
    :param coef: coefficient of the regularization
    :param mu: regularization matrix
    :return: hat matrix of the system AX=Y or AX=BY, with a second order Tikonov regularisation
    '''

    if W is None:
        A = sp.csc_matrix(A)
        return A @ inv(A.T @ A + coef * mu.T @ mu) @ A.T
    else:

        A = A[W != 0]
        ATW = np.multiply(A.T, W[W != 0][np.newaxis, :]).astype('float32')
        A = sp.csc_matrix(A)
        return A @ inv(ATW @ A + coef * mu.T @ mu) @ ATW


def GCV_function(H, data_values, Residu, W):
    n = W[W != 0].shape[0]
    d = np.sum(np.diag(H)) / n  # matrix trace divided by n
    gcv = np.sum((Residu / (1 - d)) ** 2) / n  #
    return gcv


def studentized_residual(A, residu, dates_range, W, coef, regu):
    '''

    :param A: matrix of the temporal invserion system AX=Y
    :param residu: residual (difference between AX and Y (or BY))
    :param dates_range: an array with all the dates included in data, list
    :return: internally studentized residual
    '''
    if A.shape[0] == A.shape[1]:
        sigma = m.sqrt(sum(residu ** 2))
    else:
        sigma = m.sqrt(sum(residu ** 2) / (A.shape[0] - A.shape[1]))
    H = np.diag(hat_matrix(A, dates_range, W, coef, regu))
    Hii = np.where(H == 1.0, 0.99, H)  # to avoid a division by 0
    z = np.array(residu / (sigma * np.sqrt(1 - Hii))).astype('float')
    return np.nan_to_num(z)


def externally_studentized_residual(A, residu, dates_range, W, coef, regu):
    '''

    :param A: matrix of the temporal invserion system AX=Y
    :param residu:
    :param dates_range: an array with all the dates included in data, list
    :return: externally studentized residual
    '''
    n = A.shape[0]
    p = A.shape[1]
    r = studentized_residual(A, residu, dates_range, W, coef, regu)
    return r * ((n - p - 1) / (n - p - r ** 2)) ** (1 / 2)


def BYAX_construction_B(A, dates_range, data, all_possibilities=False, verbose=False):
    '''
    Construction of matrix B which combine the displacement observations as explained in the TGRS paper
    :param A: matrix of the temporal invserion system AX=Y
    :param dates_range: an array with all the dates included in data, list
    :param data: an array where each line is (date1, date2, other elements) for which a velocity is computed
    :param all_possibilities:
    :param verbose:
    :return:
    '''

    import copy
    B = np.zeros((A.shape[0], A.shape[0]))
    dates = copy.deepcopy(data[:, :2])
    sensors = copy.deepcopy(data[:, 6])
    authors = copy.deepcopy(data[:, 7])
    for ligne in range(A.shape[0]):
        if A[ligne, :].any() == False:  # il n'y a que des 0 sur la ligne
            if verbose:
                print(f'ligne avec des 0 seulement {ligne}')
                print(f'original dates {data[ligne, 0]} - {data[ligne, 1]}')

            if data[ligne, 1] not in dates_range and data[
                ligne, 0] not in dates_range:  # date1 and date2 are not in Date_range
                if verbose: print('Date1 and Date2  not in dates_range')

                if all_possibilities:
                    Save_d1_add_to_date1 = {}  # dico qui contient l'element de Y ajoute (colone de B) (cle) et la ligne de B sur laquelle cette combinaison est ecrite (valeur)
                    Save_d2_sub_date1 = {}

                Add_to_date2 = np.where(data[:, 0] == data[ligne, 1])[0]
                if Add_to_date2.shape[0] != 0:
                    for i_add_to_date2 in Add_to_date2:
                        if data[i_add_to_date2, 1] in dates_range:
                            # print(f'Add {data[i_add_to_date2, 0]} - {data[i_add_to_date2, 1]}')
                            if B[ligne, :].any() == False:  # s'il n'y a pas encore de combinaison remplie
                                B[ligne, i_add_to_date2] = B[ligne, ligne] = 1
                                if verbose: print(
                                    f'Add {i_add_to_date2} {data[i_add_to_date2, 0]} - {data[i_add_to_date2, 1]}')
                                dates[ligne, 0] = data[ligne, 0]
                                dates[ligne, 1] = data[i_add_to_date2, 1]
                                sensors[ligne] = f'{sensors[ligne]};{sensors[i_add_to_date2]}'
                                authors[ligne] = f'{authors[ligne]};{authors[i_add_to_date2]}'
                                if all_possibilities: Save_d1_add_to_date1[i_add_to_date2] = ligne
                                # print(f'New date {data[ligne, 0]} - {data[i_add_to_date2, 1]}')
                                break
                            elif all_possibilities:  # s'il y a en a mais qu'on veut toute les possibilite
                                B = np.append(B, [[0] * B.shape[1]], axis=0)
                                B[-1, i_add_to_date2] = B[-1, ligne] = 1
                                if verbose: print(
                                    f'Add {i_add_to_date2}  as new line {B.shape[0] - 1} {data[i_add_to_date2, 0]} - {data[i_add_to_date2, 1]}')
                                dates = np.append(dates, [[data[ligne, 0], data[i_add_to_date2, 1]]], axis=0)
                                Save_d1_add_to_date1[i_add_to_date2] = B.shape[0] - 1

                Sub_to_date2 = np.where(data[:, 1] == data[ligne, 1])[0]
                if Sub_to_date2.shape[0] != 0:
                    for i_sub_to_date2 in Sub_to_date2:
                        if data[i_sub_to_date2, 0] > data[ligne, 0]:
                            if data[i_sub_to_date2, 0] in dates_range:
                                if B[ligne, :].any() == False:  # s'il n'y a pas encore de combinaison remplie
                                    B[ligne, ligne] = 1
                                    B[ligne, i_sub_to_date2] = -1
                                    if verbose: print(
                                        f'Sub {i_sub_to_date2}  {data[i_sub_to_date2, 0]} - {data[i_sub_to_date2, 1]}')
                                    dates[ligne, 0] = data[ligne, 0]
                                    dates[ligne, 1] = data[i_sub_to_date2, 0]
                                    sensors[ligne] = f'{sensors[ligne]};{sensors[i_sub_to_date2]}'
                                    authors[ligne] = f'{authors[ligne]};{authors[i_sub_to_date2]}'
                                    if all_possibilities: Save_d2_sub_date1[i_sub_to_date2] = ligne
                                    break  # si une combinaison est trouvee on sort de la boucle for
                                    # print(f'New date {data[ligne, 0]} - {data[i_sub_to_date2, 0]}')
                                elif all_possibilities:
                                    B = np.append(B, [[0] * B.shape[1]], axis=0)
                                    B[-1, ligne] = 1
                                    B[-1, i_sub_to_date2] = -1
                                    if verbose: print(
                                        f'Sub {i_sub_to_date2}  as new ligne {B.shape[0] - 1} {data[i_sub_to_date2, 0]} - {data[i_sub_to_date2, 1]}')
                                    dates = np.append(dates, [[data[ligne, 0], data[i_sub_to_date2, 0]]], axis=0)
                                    Save_d2_sub_date1[i_sub_to_date2] = B.shape[0] - 1

                Add_to_date1 = np.where(data[:, 1] == data[ligne, 0])[0]
                if Add_to_date1.shape[0] != 0:
                    for i_add_to_date1 in Add_to_date1:
                        if data[i_add_to_date1, 0] in dates_range:
                            if np.count_nonzero(B[ligne, :]) < 3:  # only combinations of three displacements
                                B[ligne, i_add_to_date1] = B[ligne, ligne] = 1
                                dates[ligne, 0] = data[i_add_to_date1, 0]
                                sensors[ligne] = f'{sensors[ligne]};{sensors[i_add_to_date1]}'
                                authors[ligne] = f'{authors[ligne]};{authors[i_add_to_date1]}'
                                if verbose:
                                    print(f'Add {i_add_to_date1} {data[i_add_to_date1, 0]} - {data[i_add_to_date1, 1]}')
                                    print(f'New date {data[i_add_to_date1, 0]} - {dates[ligne, 1]}')
                                break
                            elif all_possibilities:
                                if len(Save_d1_add_to_date1) != 0:
                                    for colonneB, ligneB in Save_d1_add_to_date1.items():
                                        if np.count_nonzero(B[ligneB, :]) < 3:
                                            B[ligneB, i_add_to_date1] = 1
                                            dates[ligneB, 0] = data[i_add_to_date1, 0]
                                            if verbose:
                                                print(
                                                    f'Add to the line {ligneB} the index {i_add_to_date1} {data[i_add_to_date1, 0]} - {data[i_add_to_date1, 1]}')
                                                print(f'New date {data[i_add_to_date1, 0]} - {data[colonneB, 1]}')
                                        else:
                                            B = np.append(B, [[0] * B.shape[1]], axis=0)
                                            B[-1, i_add_to_date1] = B[-1, ligne] = B[-1, colonneB] = 1
                                            dates = np.append(dates, [[data[i_add_to_date1, 0], data[colonneB, 1]]],
                                                              axis=0)
                                            if verbose:
                                                print(
                                                    f'Add to the line {B.shape[0]} the index {i_add_to_date1} {data[i_add_to_date1, 0]} - {data[i_add_to_date1, 1]}')
                                                print(f'New date {data[i_add_to_date1, 0]} - {data[colonneB, 1]}')
                                if len(Save_d2_sub_date1) != 0:
                                    for colonneB, ligneB in Save_d2_sub_date1.items():
                                        if np.count_nonzero(B[ligneB, :]) < 3:
                                            B[ligneB, i_add_to_date1] = 1
                                            dates[ligneB, 0] = data[i_add_to_date1, 0]
                                            if verbose:
                                                print(
                                                    f'Add to the line {ligneB} the index {i_add_to_date1} {data[i_add_to_date1, 0]} - {data[i_add_to_date1, 1]}')
                                                print(f'New date {data[i_add_to_date1, 0]} - {data[colonneB, 0]}')
                                        else:
                                            B = np.append(B, [[0] * B.shape[1]], axis=0)
                                            B[-1, i_add_to_date1] = B[-1, ligne] = 1
                                            B[-1, colonneB] = -1
                                            dates = np.append(dates, [[data[i_add_to_date1, 0], data[colonneB, 0]]],
                                                              axis=0)
                                            if verbose:
                                                print(
                                                    f'Add to the line {B.shape[0] - 1} the index {i_add_to_date1} {data[i_add_to_date1, 0]} - {data[i_add_to_date1, 1]}')
                                                print(f'New date {data[i_add_to_date1, 0]} - {data[colonneB, 0]}')

                Sub_to_date1 = np.where(data[:, 0] == data[ligne, 0])[0]
                if Sub_to_date1.shape[0] != 0:
                    for i_sub_to_date1 in Sub_to_date1:
                        if data[i_sub_to_date1, 1] < data[ligne, 1]:
                            if data[i_sub_to_date1, 1] in dates_range:
                                if np.count_nonzero(B[ligne, :]) < 3 and np.count_nonzero(B[ligne,
                                                                                          :ligne + 1]) < 2:  # si une combinaison de 3 n'a pas deja ete trouve et que date1 n'a pas deja ete modifie
                                    B[ligne, ligne] = 1
                                    B[ligne, i_sub_to_date1] = -1
                                    dates[ligne, 0] = data[i_sub_to_date1, 1]
                                    sensors[ligne] = f'{sensors[ligne]};{sensors[i_sub_to_date1]}'
                                    authors[ligne] = f'{authors[ligne]};{authors[i_sub_to_date1]}'
                                    if verbose:
                                        print(
                                            f'Sub {i_sub_to_date1} {data[i_sub_to_date1, 0]} - {data[i_sub_to_date1, 1]}')
                                        print(f'New date {data[i_sub_to_date1, 0]} - {data[ligne, 1]}')
                                    break
                                elif all_possibilities:
                                    if len(Save_d1_add_to_date1) != 0:
                                        for colonneB, ligneB in Save_d1_add_to_date1.items():
                                            if np.count_nonzero(B[ligneB, :]) < 3:
                                                B[ligneB, i_sub_to_date1] = -1
                                                dates[ligneB, 0] = data[i_sub_to_date1, 1]
                                                if verbose:
                                                    print(
                                                        f'Add to the line {ligneB} the index {i_sub_to_date1}  {data[i_sub_to_date1, 0]} - {data[i_sub_to_date1, 1]}')
                                                    print(f'New date {data[i_sub_to_date1, 1]} - {data[colonneB, 1]}')
                                            else:
                                                B = np.append(B, [[0] * B.shape[1]], axis=0)
                                                B[-1, i_sub_to_date1] = -1
                                                B[-1, ligne] = B[-1, colonneB] = 1
                                                dates = np.append(dates, [[data[i_sub_to_date1, 1], data[colonneB, 1]]],
                                                                  axis=0)
                                                if verbose:
                                                    print(
                                                        f'Add to the line {B.shape[0] - 1} the index {i_sub_to_date1}  {data[i_sub_to_date1, 0]} - {data[i_sub_to_date1, 1]}')
                                                    print(f'New date {data[i_sub_to_date1, 1]} - {data[colonneB, 1]}')
                                    if len(Save_d2_sub_date1) != 0:
                                        for colonneB, ligneB in Save_d2_sub_date1.items():
                                            if np.count_nonzero(B[ligneB, :]) < 3:
                                                B[ligneB, i_sub_to_date1] = -1
                                                dates[ligneB, 0] = data[i_sub_to_date1, 1]
                                                if verbose:
                                                    print(
                                                        f'Add to the line {ligneB} the index {i_sub_to_date1} {data[i_sub_to_date1, 0]} - {data[i_sub_to_date1, 1]}')
                                                    print(f'New date {data[i_sub_to_date1, 0]} - {data[colonneB, 0]}')
                                            else:
                                                B = np.append(B, [[0] * B.shape[1]], axis=0)
                                                B[-1, i_sub_to_date1] = B[-1, colonneB] = -1
                                                B[-1, ligne] = 1
                                                dates = np.append(dates, [[data[i_sub_to_date1, 1], data[colonneB, 1]]],
                                                                  axis=0)
                                                if verbose:
                                                    print(
                                                        f'Add to the line {B.shape[0] - 1} the index {i_sub_to_date1} {data[i_sub_to_date1, 0]} - {data[i_sub_to_date1, 1]}')
                                                    print(f'New date {data[i_sub_to_date1, 0]} - {data[colonneB, 0]}')

                if np.count_nonzero(B[ligne,
                                    :]) == 2:  # if a comination of 3 displacements have not been found or the combination found does not work since date1 is higher than date2
                    if verbose:
                        print('Try of combination of three failed')
                        print(dates[ligne])
                    B[ligne, :].fill(0)
                    B[ligne, ligne] = 1
                    dates[ligne, 0] = data[ligne, 0]
                    dates[ligne, 1] = data[ligne, 1]
                    sensors[ligne] = sensors[ligne].split(';')[0]
                    authors[ligne] = authors[ligne].split(';')[0]

                if dates[ligne, 0] >= dates[ligne, 1]:
                    if verbose:
                        print('ligne qui ne satisfait pas date1 inf a date2')
                        print(dates[ligne])
                    B[ligne, :].fill(0)
                    B[ligne, ligne] = 1
                    dates[ligne, 0] = data[ligne, 0]
                    dates[ligne, 1] = data[ligne, 1]
                    sensors[ligne] = sensors[ligne].split(';')[0]
                    authors[ligne] = authors[ligne].split(';')[0]

            elif data[
                ligne, 0] not in dates_range:  # si la date1 seulement n'existe pas dans X, alors on essaye d'ajouter un deplacement a la date1, ou de retirer a la date1
                if verbose: print(f'Date1 {data[ligne, 0]} pas dans dates_range')
                Add_to_date1 = np.where(data[:, 1] == data[ligne, 0])[
                    0]  # Les Y ayant une date2 égale à la date1 de l'element Y[ligne]
                if Add_to_date1.shape[0] != 0:
                    for i_add_to_date1 in Add_to_date1:
                        if data[i_add_to_date1, 0] in dates_range:
                            if B[ligne, :].any() == False:
                                if B[i_add_to_date1, i_add_to_date1] != 1 or B[
                                    i_add_to_date1, ligne] != 1:  # si la combinaison n'a pas deja etait utilisee
                                    B[ligne, i_add_to_date1] = B[ligne, ligne] = 1
                                    dates[ligne, 0] = data[i_add_to_date1, 0]
                                    dates[ligne, 1] = data[ligne, 1]
                                    sensors[ligne] = f'{sensors[ligne]};{sensors[i_add_to_date1]}'
                                    authors[ligne] = f'{authors[ligne]};{authors[i_add_to_date1]}'
                                    if verbose:
                                        print(
                                            f'Add {i_add_to_date1} {data[i_add_to_date1, 0]} - {data[i_add_to_date1, 1]}')
                                        print(f'New date {data[i_add_to_date1, 0]} - {data[ligne, 1]}')
                                elif verbose:
                                    print(
                                        f'Combi deja existante Add {i_add_to_date1} {data[i_add_to_date1, 0]} - {data[i_add_to_date1, 1]}')
                            elif all_possibilities:
                                B = np.append(B, [[0] * B.shape[1]], axis=0)
                                B[-1, i_add_to_date1] = B[-1, ligne] = 1
                                dates = np.append(dates, [[data[i_add_to_date1, 0], data[ligne, 1]]], axis=0)
                                if verbose:
                                    print(
                                        f'Add {i_add_to_date1} as a new line {B.shape[0]} {data[i_add_to_date1, 0]} - {data[i_add_to_date1, 1]}')
                                    print(f'New date {data[i_add_to_date1, 0]} - {data[ligne, 1]}')

                Sub_to_date1 = np.where(data[:, 0] == data[ligne, 0])[
                    0]  # Les Y ayant une date2 égale à la date1 de l'element Y[ligne]
                if Sub_to_date1.shape[0] != 0:
                    for i_sub_to_date1 in Sub_to_date1:
                        if data[i_sub_to_date1, 1] < data[ligne, 1]:
                            if data[i_sub_to_date1, 1] in dates_range:
                                if B[ligne, :].any() == False:
                                    if B[i_sub_to_date1, i_sub_to_date1] != 1 or B[i_sub_to_date1, ligne] != 1:
                                        B[ligne, ligne] = 1
                                        B[ligne, i_sub_to_date1] = -1
                                        dates[ligne, 0] = data[i_sub_to_date1, 1]
                                        dates[ligne, 1] = data[ligne, 1]
                                        sensors[ligne] = f'{sensors[ligne]};{sensors[i_sub_to_date1]}'
                                        authors[ligne] = f'{authors[ligne]};{authors[i_sub_to_date1]}'
                                        if verbose:
                                            print(f'New date {data[i_sub_to_date1, 0]} - {data[ligne, 1]}')
                                            print(
                                                f'Sub {i_sub_to_date1} {data[i_sub_to_date1, 0]} - {data[i_sub_to_date1, 1]}')
                                    elif verbose:
                                        print(
                                            f'Combi deja existante Sub {i_sub_to_date1} {data[i_sub_to_date1, 0]} - {data[i_sub_to_date1, 1]}')
                                elif all_possibilities:
                                    B = np.append(B, [[0] * B.shape[1]], axis=0)
                                    B[-1, ligne] = 1
                                    B[-1, i_sub_to_date1] = -1
                                    dates = np.append(dates, [[data[i_sub_to_date1, 1], data[ligne, 1]]], axis=0)
                                    if verbose:
                                        print(f'New date {data[i_sub_to_date1, 0]} - {data[ligne, 1]}')
                                        print(
                                            f'Sub {i_sub_to_date1}  as new ligne {B.shape[0]} {data[i_sub_to_date1, 0]} - {data[i_sub_to_date1, 1]}')

            else:
                if verbose: print(f'Date2 {data[ligne, 1]} pas dans dates_range')
                Add_to_date2 = np.where(data[:, 0] == data[ligne, 1])[
                    0]  # Les Y ayant une date1 égale à la date2 de l'element Y[ligne]
                # Les Y ayant une date1 égale à la date2 de l'element Y[ligne]
                if Add_to_date2.shape[0] != 0:
                    for i_add_to_date2 in Add_to_date2:
                        if data[i_add_to_date2, 1] in dates_range:
                            # print(f'Add {data[i_add_to_date2, 0]} - {data[i_add_to_date2, 1]}')
                            if B[ligne, :].any() == False:
                                if B[i_add_to_date2, i_add_to_date2] != 1 or B[i_add_to_date2, ligne - 1] != 1:
                                    B[ligne, i_add_to_date2] = B[ligne, ligne] = 1
                                    dates[ligne, 0] = data[ligne, 0]
                                    dates[ligne, 1] = data[i_add_to_date2, 1]
                                    sensors[ligne] = f'{sensors[ligne]};{sensors[i_add_to_date2]}'
                                    authors[ligne] = f'{authors[ligne]};{authors[i_add_to_date2]}'
                                    if verbose:
                                        print(f'New date {data[ligne, 0]} - {data[i_add_to_date2, 1]}')
                                        print(
                                            f'Add {i_add_to_date2} {data[i_add_to_date2, 0]} - {data[i_add_to_date2, 1]}')
                                elif verbose:
                                    print(
                                        f'Combi deja existante Add {i_add_to_date2} {data[i_add_to_date2, 0]} - {data[i_add_to_date2, 1]}')
                            elif all_possibilities:
                                B = np.append(B, [[0] * B.shape[1]], axis=0)
                                B[-1, i_add_to_date2] = B[-1, ligne] = 1
                                dates = np.append(dates, [[data[ligne, 0], data[i_add_to_date2, 1]]], axis=0)
                                if verbose:
                                    print(
                                        f'Add {i_add_to_date2}  as new line {B.shape[0]} {data[i_add_to_date2, 0]} - {data[i_add_to_date2, 1]}')
                                    print(f'New date {data[ligne, 0]} - {data[i_add_to_date2, 1]}')

                Sub_to_date2 = np.where(data[:, 1] == data[ligne, 1])[0]
                if Sub_to_date2.shape[0] != 0:
                    for i_sub_to_date2 in Sub_to_date2:
                        if data[i_sub_to_date2, 0] > data[ligne, 0]:
                            if data[i_sub_to_date2, 0] in dates_range:
                                if B[ligne, :].any() == False:
                                    if B[i_sub_to_date2, ligne] != 1 or B[i_sub_to_date2, i_sub_to_date2] != -1:
                                        B[ligne, ligne] = 1
                                        B[ligne, i_sub_to_date2] = -1
                                        dates[ligne, 0] = data[ligne, 0]
                                        dates[ligne, 1] = data[i_sub_to_date2, 0]
                                        sensors[ligne] = f'{sensors[ligne]};{sensors[i_sub_to_date2]}'
                                        authors[ligne] = f'{authors[ligne]};{authors[i_sub_to_date2]}'
                                        if verbose:
                                            print(f'New date {data[ligne, 0]} - {data[i_sub_to_date2, 0]}')
                                            print(
                                                f'Sub {i_sub_to_date2}  {data[i_sub_to_date2, 0]} - {data[i_sub_to_date2, 1]}')
                                    elif verbose:
                                        print(
                                            f'Combi deja existante Sub {i_sub_to_date2}  {data[i_sub_to_date2, 0]} - {data[i_sub_to_date2, 1]}')
                                elif all_possibilities:
                                    B = np.append(B, [[0] * B.shape[1]], axis=0)
                                    B[-1, ligne] = 1
                                    B[-1, i_sub_to_date2] = -1
                                    dates = np.append(dates, [[data[ligne, 0], data[i_sub_to_date2, 0]]], axis=0)
                                    if verbose:   print(
                                        f'Sub {i_sub_to_date2} as new ligne {B.shape[0]} {data[i_sub_to_date2, 0]} - {data[i_sub_to_date2, 1]}')
        else:
            B[ligne, ligne] = 1

    return B, dates, sensors, authors


