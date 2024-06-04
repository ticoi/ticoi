'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
For one cube of data
Author: Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''
# %%
from ticoi.core import *
from Visualisation.Download_cube_from_shp import search_catalog_ige_TC
from ticoi.cube_data_classxr import *
from joblib import Parallel, delayed

#### INPUTS

# Selection of data
# SENTINEL-2 data
cube_path = '/media/charriel/Elements/Donnees_IGE_Alpes/cubes_S2/'  # path are the Sentinel-2 IGE cubes are stored
catalog_cubeige = '/media/charriel/Elements/Donnees_IGE_Alpes/cubes_S2/cube_grid.shp'  # path of the catalog, which represent the location of the different cubes
# PLEAIDES
cube_path_pleaide = None  # the inversion is computed using Sentinel-2 data only
# cube_path_pleaide = '/media/charriel/Elements/Donnees_IGE_Alpes/cubes_Pleiades/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc' #Pleaide data are added

# point where to carry on the computation
i, j = 6.8406, 45.8459  # Taconnaz, upper part

dates_input = ['2016-01-01',
               '2023-01-01']  # to select certain temporal baselines in the dataset, if you want to select all the temporal put None, else put ['yy1-mm1-dd1','yy2-mm2-dd2']
temp_baseline = None  # to select certain temporal baselines in the dataset
conf = False  # if you want confidence indicators ranging between 0 and 1, with 1 the lowest errors
unit = 'm/y'
delete_outliers = None  # if None, all the data are included; if an integer, the data with a error higher than this interger are removed; if median_average, the data with a direction 45° away compared to the averaged direction are removed
subset = [6.81221216303653065, 6.81533512845225499, 45.86674397287811189, 45.86887069444943108]
proj = 'EPSG:32632'
flags = None
#Preparation
smooth_method = "gaussian"
t_win = 90
s_win = 3
select_temporal_baseline = 120


####  Inversion
# Variables to play with
coef = 100  # lambda : coef of the regularisation
regu = 1  # Type of regularisation : 1, 2,'1accelnotnull','regu01'
apriori_weight = True  # Add a weight in the first step of the inversion, True ou False
# Varibales which can stay stable for the moment
solver = 'LSMR_ini'  # Solver for the inversion : 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
detect_temporal_decorrelation = False  # Detect temporal decorrelation by setting a weight of 0 at the beginning at the first inversion to all observation with a temporal baseline larger than 200
# result_quality = None  # None or list of str, which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
result_quality = ['X_contribution', 'Norm_residual']

####  Interpolation
option_interpol = 'spline'  # Type of interpolation : 'spline', 'nearest' or 'spline_smooth' for smoothing spline
interpolation_bas = 30  # Temporal sampling of the velocity time series
redundancy = 5

# Visualization
name_result = 'test'
path_save = f'/media/charriel/Elements/Donnees_IGE_Alpes/Test_calcul/Taconnaz/'  # path where to stored the results

# Process
nb_cpu = 8
verbose = False
save = True
interpolation = False

option_visual = ['orginal_velocity_xy', 'original_magnitude', 'X_magnitude_zoom',
                 'X',
                 'X_vxvy',
                 'vv_quality']  # ['orginal_velocity_xy','original_magnitude','error','vv_good_quality','vv_quality','vxvy_quality','X','X_vxvy','X_magnitude','X_magnitude_Zoom','X_filter','X_filterZoom','X_magnitude_filter','Y_contribution','Residu','Residu_magnitude']
#

coef = 150 # lambda : coef of the regularisation
regu = '1accelnotnull' # Type of regularisation : 1, 2'1accelnotnull'  # Type of regularisation : 1, 2,'1accelnotnull' : 1 is Tikhonov first order, 2 is Tikhonov second order and '1accelnotnull is Tikhonov first order with an apriori on the acceleration

load_kwargs = {'filepath': None,
               'chunks': {},
               'conf': False,
               'subset': subset,
               'buffer': None,
               'pick_date': ['2013-01-01', '2023-03-01'],
               'pick_sensor': None,
               'pick_temp_bas': None,
               'proj': 'EPSG:4326',
               'verbose': False}

preData_kwargs = {'smooth_method': 'gaussian',
                  's_win': 3,
                  't_win': 90,
                  'sigma': 3,
                  'order': 3,
                  'unit': 365,
                  'delete_outliers': None,
                  'flags': flags,
                  'regu': regu,
                  'solver': 'LSMR_ini',
                  'proj': proj,
                  'velo_or_disp': 'disp',
                  'verbose': True}

inversion_kwargs = {'solver': 'LSMR_ini',
                    'coef': coef,
                    'apriori_weight': True,
                    'path_save': path_save,
                    'interpolation_load_pixel': 'nearest',
                    'iteration': True,
                    'interval_output': 1,
                    'treshold_it': 0.1,
                    'conf': False,
                    'flags': flags,
                    'regu': regu,
                    'interpolation_bas': 90,
                    'option_interpol': 'spline',
                    'redundancy': 30,
                    'proj': proj,
                    'detect_temporal_decorrelation': True,
                    'unit': 365,
                    'result_quality': ['X_contribution'],
                    'nb_max_iteration': 10,
                    'delete_outliers': None,
                    'interpolation': False,
                    'linear_operator': None,
                    'visual': False,
                    'verbose': False
}


if not os.path.exists(path_save):
    os.mkdir(path_save)

# lazy load the original data
start_process = time.time()
cube = cube_data_class()
cube_name = search_catalog_ige_TC(catalog_cubeige, i, j, cube_path, EPSG='EPSG:4326')
load_kwargs.update({'cube_name':cube_name})


cube.load(load_kwargs['cube_name'], pick_date=load_kwargs['pick_date'], chunks=load_kwargs['chunks'], conf=load_kwargs['conf'],
          pick_sensor=load_kwargs['pick_sensor'], pick_temp_bas=load_kwargs['pick_temp_bas'], proj=load_kwargs['proj'], verbose=load_kwargs['verbose'],subset = load_kwargs['subset'])
print(f'Time download cube {round((time.time() - start_process), 4)} sec')


cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
first_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())

inversion_kwargs.update({'first_date_interpol': first_date_interpol, 'last_date_interpol': last_date_interpol})

start = time.time()
result = process_blocks_refine(cube, nb_cpu=nb_cpu, block_size=0.5, preData_kwargs=preData_kwargs, inversion_kwargs=inversion_kwargs,verbose=True)


print(f'Time inversion {round((time.time() - start), 4)} sec')

#
# %% Initialisation of the cube to store the data
merged = None
if save:
    if merged is None:
        sensor_array = np.unique(cube.ds['sensor'])
    else:
        sensor_array = f"{np.unique(cube.ds['sensor'])},{np.unique(cube2.ds['sensor'])}"
    sensor_strings = [str(sensor) for sensor in sensor_array]
    sensor = ', '.join(sensor_strings)
    if merged is None:
        source = f'Temporal inversion on cube {cube.filename} using TICOI with a selection of the dates: {dates_input}, with a selection of the baseline {temp_baseline}, and a temporal spacing every {redundancy} days '
    else:
        source = f'Temporal inversion on cube {cube.filename} & {cube2.filename} using TICOI with a selection of the dates: {dates_input}, with a selection of the baseline {temp_baseline}, and a temporal spacing every {redundancy} days '
    if apriori_weight:
        source += ' and apriori weight'
    source += f'. The Tikhonov coef is: {coef}.'
    if interpolation:
        source += f'The interpolation option is: {option_interpol}.'
        if interpolation_bas:
            source += f'The interpolation baseline is: {interpolation_bas} days.'

# %% save the result
if interpolation:
    cube.write_result_TICOI(result, source, sensor, filename=name_result, result_quality=result_quality,
                            savepath=path_save)
else:
    cube.write_result_tico(result, source, sensor, filename=name_result, result_quality=result_quality,
                           savepath=path_save)  # save the cumulative displacement time series
end = time.time()
print(f'Total process {(end - start) / 60} min')
print('stop')



