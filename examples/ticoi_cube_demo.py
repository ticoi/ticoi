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
Area = '121pix'
temp_baseline = None  # to select certain temporal baselines in the dataset
dates_input = ['2013-01-01', '2023-12-31']  # to select certain temporal baselines in the dataset
# sensor = ['Sentinel-1', 'Sentinel-2', 'Landsat-8', 'Landsat-9']
sensor = None
conf = False  # convert errors to confidence value between 0 and 1
IGE = None
delete_outliers = None  # if None, all the data are included; if an integer, the data with a error higher than this interger are removed; if median_average, the data with a direction 45° away compared to the averaged direction are removed
unit = 'm/y'

# Inversion
# variable to play with
coef = 150  # coef de la regularisation
regu = '1accelnotnull'  # 1, 2,'1accelnotnull','regu01', 'regu01accelnotnull', 'regu01variablecoef', 'direction' or 'directionxy

# varibale that can stay stable for the moment
apriori_weight = False  # add a weight in the first step of the inversion, True ou False
solver = 'LSMR_ini'  # can be 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
detect_temporal_decorrelation = False  # Detect temporal decorrelation by setting a weight of 0 at the beginning at the first inversion to all observation with a temporal baseline larger than 200

# Interpolation
option_interpol = 'spline'  # type of interpolation 'spline', 'nearest' or 'spline_smooth' for smoothing spline
interpolation_bas = 30  # temporal sampling of the velocity time series
redundancy = 5

# Visualization
name_result = 'test'
path_save = '/home/charriel/Documents/Bettik/Yukon/'

# Process
nb_cpu = 8
verbose = False
save = True
interpolation = True
linear_operator = True
# result_quality = ['X_contribution']
result_quality = None

if Area == 'Kask1':
    # cube_name_its = '/home/charriel/Documents/Bettik/Yukon/Kask1.nc'
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr'
    subset = [-139.13562486278613051, -138.91473683122822536, 60.78278231023399059, 60.70082738686445367]

elif Area == 'Kask2':
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr'
    subset = [-138.92020049278619354, -138.78282843075726305, 60.78122126407456705, 60.70004686378474901]

elif Area == 'Kask3':
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr'
    subset = [-138.79375575387319941, -138.66809153803993127, 60.80385643338615864, 60.70863261766155716]

if Area == 'reg1+reg2':
    cube_name_IGE = '/home/charriel/Documents/Bettik/Yukon/STACK/c_x17640_y07105_STACKL8S2_v2017-2018_aligned_withoutproj.nc'
    cube_name_its = '/home/charriel/Documents'
    subset = [-138.91256549918699648, -139.14221896048982785, 60.76261506545485958, 60.70280279139559099]  # reg1+reg2
    name_result = f'TICOI_20d_spline_cube_reg1reg2_all_bas_correction_shift_{coef}'

elif Area == 'middle':
    cube_name_IGE = '/home/charriel/Documents/Bettik/Yukon/STACK/c_x17885_y07105_STACKL8S2_v2017-2018_aligned_withoutproj.nc'
    subset = [-138.91513162202585363, -138.76944066698951019, 60.75629845538990992, 60.704952803013540]  # middle
    cube_name_its = '/home/charriel/Documents/Bettik/ITS_reg1+reg2.nc'
    name_result = f'TICOI_20d_spline_cube_middle_all_bas_correction_shift_big_cube_{coef}'
    print('TC cube middle')

elif Area == 'lower':
    cube_name_IGE = '/home/charriel/Documents/Bettik/Yukon/STACK//c_x17885_y06860_STACKL8S2_v2017-2018_aligned_withoutproj.nc'
    subset = [-138.78586801228252057, -138.68659767706421349, 60.75548142105910188, 60.78725508665475985]  # lower
    name_result = f'TICOI_20d_spline_cube_lower_all_bas_correction_shift_{coef}_corrected'

elif Area == '952pix':
    subset = [-138.91529983476164034, -138.84537074675375834, 60.70473221046721335, 60.73145775970764504]
    cube_name_IGE = '/home/charriel/Documents/Bettik/Yukon/STACK/c_x17885_y07105_STACKL8S2_v2017-2018_aligned_withoutproj.nc'
    cube_name_its = '/home/charriel/Documents/Bettik/ITS_reg1+reg2.nc'
    name_result = f'TICOI_20d_spline_cube_test_{Area}_all_bas_correction_shift_{coef}_LSMR_ini_cpu1'

if Area == '121pix':
    # cube_name_IGE = '/home/charriel/Documents/Bettik/Yukon/STACK/c_x17885_y06860_STACKL8S2_v2017-2018_aligned_withoutproj.nc'
    cube_name_IGE = None
    subset = [-138.97693019608831833, -138.95625602833800372, 60.74712197600209862, 60.7572939175453115]  # 121pix
    cube_name_its = '/home/charriel/Documents/Bettik/ITS_reg1+reg2.nc'
    name_result = f'TICOI_20d_spline_cube_test_{Area}_all_bas_correction_shift_{coef}_LSMR_cpu8_test'

elif Area == 'Lowell3':
    cube_name_IGE = None
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y150000.zarr'
    subset = [-138.43247970484043208, -138.28565361442687731, 60.27660958460058538, 60.34746353993884327]

elif Area == 'Lowell4_1':
    cube_name_IGE = None
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y150000.zarr'
    subset = [-138.57208229005672706, -138.4250997652048909, 60.27275492222882036, 60.34442309314081143]

elif Area == 'Lowell4_2':
    cube_name_IGE = None
    cube_name_its = '/home/charriel/Documents/Bettik/ITS_Lowell4_2.nc'
    # cube_name_its = 's3://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr'
    subset = [-138.57208229005672706, -138.4250997652048909, 60.27275492222882036, 60.34442309314081143]

elif Area == 'Lowell2':
    cube_name_IGE = None
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y150000.zarr'
    subset = [-138.28962881999922274, -138.16078000676117199, 60.25934205396930565, 60.32362315079532777]

elif Area == 'Lowell1':
    cube_name_IGE = None
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y150000.zarr'
    subset = [-138.16647995824769168, -138.02419119378481582, 60.26248522281820641, 60.32696431340952614]

elif Area == 'Lowell_Lower_for_GPS':
    cube_name_IGE = None
    cube_name_its = '/home/charriel/Documents/Bettik/ITS_Lowell_Lower_for_GPS.nc'
    # cube_name_its = 's3://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y150000.zarr'
    subset = [-138.18358952707811227, -138.13575467499342153, 60.28709907547193581, 60.29787339394205503]
    name_result = f'TICOI_30d_spline_lowell_{Area}'

elif Area == 'Lowellh1':
    cube_name_IGE = None
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr'
    subset = [-138.74834900302442975, -138.57935704293058166, 60.31045067013290861, 60.394657163254791]

elif Area == 'Lowell4':
    cube_name_IGE = None
    cube_name_its = 'http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr'
    subset = [-138.5913335969987088, -138.42080144776221573, 60.27152474331228404, 60.35713284272632251]



if not os.path.exists(path_save):  # create a subfolder if it doesnt exist
    os.mkdir(path_save)

unit = 365 if unit == 'm/y' else 1
# %% Data download
start_process = time.time()
cube = cube_data_class()
cube.load(cube_name_its, pick_date=dates_input, subset=subset, proj='EPSG:4326', pick_temp_bas=temp_baseline,
          pick_sensor=sensor, chunks={})
print(f'Time download cube {round((time.time() - start_process), 4)} sec')

start = time.time()
obs_filt = cube.preData_np(s_win=3, t_win=90, proj='EPSG:3413', regu=regu,
                           delete_outliers=None, verbose=False,
                           velocity_or_displacement='disp')
print(f'Time rolling_mean {round((time.time() - start), 4)} sec')

cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
if IGE is not None:
    cube2 = cube_data_class()
    cube2.load(cube_name_IGE, pick_date=dates_input, pick_temp_bas=temp_baseline, proj='EPSG:4326',
               verbose=True, conf=conf, subset=subset)
    merged = [cube2]
    cube2_date1 = cube2.date1_().tolist()
    cube2_date1.remove(np.min(cube2_date1))
    start_date_interpol = np.min([np.min(cube2_date1), np.min(cube_date1)])
    last_date_interpol = np.max([np.max(cube.date2_()), np.max(cube2.date2_())])
else:
    merged = None
    start_date_interpol = np.min(cube_date1)
    last_date_interpol = np.max(cube.date2_())

print(f'Cube of dimesion (nz,nx,ny) ITS_LIVE: ({cube.nz},{cube.nx},{cube.ny}) ')

if verbose:
    if merged is not None: print(f'Cube of dimesion (nz,nx,ny) IGE: ({cube2.nz},{cube2.nx},{cube2.ny}) ')

start = time.time()

# %% Initialisation of the cube to store the data
start = time.time()
if save:
    if merged is None:
        sensor_array = np.unique(cube.ds['sensor'])
    else:
        sensor_array = f"{np.unique(cube.ds['sensor'])},{np.unique(cube2.ds['sensor'])}"
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
