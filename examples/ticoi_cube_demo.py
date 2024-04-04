'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
The implementation can be divided in three parts:
    - Data Download : find and download the cube corresponding for the area pixel we want to study,
    - Inversion & Interpolation: for each pixel of the cube: solving a system AX = Y to produce Irregular Leap Frog time series using the IRLS method, then interpolate the obtained ILF time series to Regular LF time series using interpolation
    - Results saving: save the result in a new netdf file

Author : Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''
from ticoi.core import *
from joblib import Parallel, delayed
import time
from ticoi.cube_data_classxr import cube_data_class
import os
from pyproj import CRS

# %%
# Selection of data
cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/ITS_LIVE_Lowell_Lower_test.nc'  # Path where the Sentinel-2 IGE cubes are stored
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "results"))}'  # Path where to stored the results

dates_input = ['2013-01-01',
               '2023-01-01']  # to select certain temporal baselines in the dataset, if you want to select all the temporal put None, else put ['yy1-mm1-dd1','yy2-mm2-dd2']
temp_baseline = None  # to select certain temporal baselines in the dataset
sensor = None
conf = False  # if you want confidence indicators ranging between 0 and 1, with 1 the lowest errors
unit = 'm/y'
delete_outliers = None  # if None, all the data are included; if an integer, the data with a error higher than this interger are removed; if median_average, the data with a direction 45° away compared to the averaged direction are removed

# Where to save the results
name_result = 'test'  # name of the cube where to save the results
path_save = f'../../results/ticoi_cube_demo'  # folder where to save the results

####  Inversion
# Variables to play with
coef = 150  # lambda : coef of the regularisation
regu = 1  # Type of regularisation : 1, 2,'1accelnotnull' : 1 is Tikhonov first order, 2 is Tikhonov second order and '1accelnotnull is Tikhonov first order with an apriori on the acceleration
apriori_weight = True  # Add a weight in the first step of the inversion, True ou False
# Varibales which can stay stable
solver = 'LSMR_ini'  # Solver for the inversion : 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
detect_temporal_decorrelation = True  # Detect temporal decorrelation by setting a weight of 0 at the beginning at the first inversion to all observation with a temporal baseline larger than 200
# result_quality = None  # None or list of str, which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
result_quality = ['X_contribution']

####  Interpolation
option_interpol = 'spline'  # Type of interpolation : 'spline', 'nearest' or 'spline_smooth' for smoothing spline
interpolation_bas = 30  # Temporal sampling of the velocity time series
redundancy = 5

# Process
nb_cpu = 8
verbose = False
save = True
interpolation = True
linear_operator = None
option_visual = ['orginal_velocity_xy', 'original_magnitude', 'X_magnitude_zoom',
                 'X',
                 'X_vxvy',
                 'vv_quality']  # ['orginal_velocity_xy','original_magnitude','error','vv_good_quality','vv_quality','vxvy_quality','X','X_vxvy','X_magnitude','X_magnitude_Zoom','X_filter','X_filterZoom','X_magnitude_filter','Y_contribution','Residu','Residu_magnitude']

unit = 365 if unit == 'm/y' else 1
# %% Data download
start_process = time.time()
cube = cube_data_class()
cube.load(cube_name, pick_date=dates_input, proj='EPSG:4326', pick_temp_bas=temp_baseline,
          pick_sensor=sensor, chunks={})
print(f'Time download cube {round((time.time() - start_process), 4)} sec')

start = time.time()
obs_filt = cube.preData_np(s_win=3, t_win=90, proj='EPSG:3413', regu=regu,
                           delete_outliers=None, verbose=False,
                           velocity_or_displacement='disp')
print(f'Time rolling_mean {round((time.time() - start), 4)} sec')

cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
merged = None
start_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())

print(f'Cube of dimesion (nz,nx,ny) ITS_LIVE: ({cube.nz},{cube.nx},{cube.ny}) ')

start = time.time()

# %% Initialisation of the cube to store the data
start = time.time()
if save:
    sensor_array = np.unique(cube.ds['sensor'])

    sensor_strings = [str(sensor) for sensor in sensor_array]
    sensor = ', '.join(sensor_strings)
    if merged is None:
        source = f'Temporal inversion on cube {cube.filename} using TICOI with a selection of the dates between: {dates_input}, with a selection of the temporal baselines {temp_baseline}'
    else:
        source = f'Temporal inversion on cube {cube.filename} & {cube2.filename} using TICOI with a selection of the dates: {dates_input}, with a selection of the baseline {temp_baseline}, and a temporal spacing every {redundancy} days '
    if apriori_weight:
        source += ' and apriori weight'
    source += f'. The Tikhonov coef is: {coef}.'
    if interpolation:
        source += f'The interpolation option is: {option_interpol}.'
        if interpolation_bas:
            source += f'The interpolation baseline is: {interpolation_bas} days.'
        source += f'The temporal spacing every {redundancy} days.'

EPSG = f'EPSG:{CRS(cube.ds.proj4).to_epsg()}'
# %% Inversion
result = Parallel(n_jobs=nb_cpu, verbose=0)(
    delayed(process)(cube,
        i, j, solver, coef, apriori_weight, path_save,obs_filt=obs_filt,interpolation_load_pixel='nearest', first_date_interpol=start_date_interpol,
        last_date_interpol=last_date_interpol, conf=conf, regu=regu, interpolation_bas=interpolation_bas,
        option_interpol=option_interpol, redundancy=redundancy,
        detect_temporal_decorrelation=detect_temporal_decorrelation, unit=unit, result_quality=result_quality,
        delete_outliers=delete_outliers, interpolation=interpolation,linear_operator=linear_operator,
        verbose=verbose)
    for i in cube.ds['x'].values for j in cube.ds['y'].values)


# For debuging, the version without parallelization
# result = []
# for i in cube.ds['x'].values:
#     for j in cube.ds['y'].values:
#         result.append(process(
#         cube.Load_Data(i, j, proj=EPSG, interp='linear', delete_outliers=delete_outliers, solver=solver,
#                        regu=regu),
#         i, j, solver, coef, apriori_weight, path_save, first_date_interpol=start_date_interpol,
#         last_date_interpol=last_date_interpol, conf=conf, regu=regu, interpolation_bas=interpolation_bas,
#         option_interpol=option_interpol, redundancy=redundancy,
#             detect_temporal_decorrelation=detect_temporal_decorrelation, unit=unit,result_quality=result_quality, verbose=verbose))
print(f'Time inversion {round((time.time() - start), 4)} sec')

# %% save the res
cube.write_result_TICOI(result, source, sensor, filename=name_result,
                        savepath=path_save, result_quality=result_quality, verbose=verbose)
print(f'Total process {(time.time() - start_process) / 60} min')
print('stop')
