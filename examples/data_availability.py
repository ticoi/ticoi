'''
A few tools to evaluate the data availability on a set of data, its repartition throughout the period of measurement and its quality 
in some cases, in order to see whether it is relevent or not to study glacier velocities at a point (seaonality, surge...). It uses 
TICOI's optimized loading to load the data.

Several possibilities are available :
    - Compute monthly data availability : Each data count for one and is ponderated depending on the month.s it covers, a bar plot
    with the total amount of data available in the set for each month of the period of measurements can be plot (linear or log)
    
    - Generate seasonal data availability maps : 4 maps giving the amount of available data at each pixel for the 4 seasons of
    the year (winter, spring, summer, autumn), added up for each year of the considered period of measurements, and based on the 
    previously computed monthly availability (using the ponderation strategy)
    
    - Compute various indices, using monthly data availability or not, using seasonal data availability maps or not (each of those
    indices are defined as functions in the next two sections):
        - Indices using raw data only :
            . all_data : Just summing up all the available data at the point over the whole considered period
            . median_baseline : Median temporal baseline (gap in days between the two acquisition dates) of the data at the point
            . max_leap_frog (can be extended by '_ny' to apply a periodic computation) : All the dates (start and end dates for each 
            data) are concatenated in a single list, which is then sorted. This index returns the maximum gap between two dates 
            (leap frog) as an approximation of the lowest redundancy in the data.
        
        - Indices using monthly data availability :
            . mini_nmonth (n must be replaced by a number, or nothing) : Minimum amount of data available on a sliding window of 
            n months over the whole period of the measurements
            . mean_nmonth (n must be replaced by a number, or nothing) : Month (or selection of n months among the 12 months) 
            with the lowest average of available data on the whole dataset
            . median_nmonth (n must be replaced by a number, or nothing) : Month (or selection of n months among the 12 months) 
            with the lowest median of available data on the whole dataset
            
        - Indices using seasonal data availability maps :
            . mini_season : Each pixel takes the value of the season with the lowest amount of available data
            . min_all_season : Season with the least data / all data : index to evaluate the repartition of the data over the year
            
To specify which indices you want to generate and, if it is not automatically done to compute the indices, whether you want to
comput the monthly availability bar plot and the seasonal data availability maps or not, you must indicate the name of the indices
in the index list between '' (some specifications are given between () above, please follow them). For monthly data availability,
write 'monthly' ('monthly_graph' if you want the monthly data availability bar plot), and 'availability_maps' to generate seasonal 
data availability maps.

Periodic computation : Some indices can be evaluated periodically (using the '_ny' extension in its name), this means they can be
individually computed on a rolling window of n consecutive years (minimum period to evaluate what we'd like to see on the data
(ex: we have to see a seasonality during two consecutive years to tell there's one at this pixel')), we then obtain several values
for the indices (every n years window possible on the whole period of measuremenst), only the best value is returned. This allows
to greatly limit the influence of some 'bad' years in terms of data availability on the overall index value.

Results are saved in a .tiff format to the specified path.

Authors : Nathan Lioret, Laurane Charrier, Lei Guo
'''

import time
import os
import warnings
import itertools
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from osgeo import gdal, osr
from joblib import Parallel, delayed
from tqdm import tqdm

from ticoi.core import process_blocks_refine
from ticoi.cube_data_classxr import cube_data_class


# %%========================================================================= #
#                                   PARAMETERS                                #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ----------------------- Choose pixel loading method --------------------- ##
# Choose the TICOI cube processing method you want to use ('block_process' or 'direct_process')
#    - 'block_process' (recommended) : This implementation divides the data in smaller data cubes loaded one after the other in a synchronous manner,
# in order to avoid memory overconsumption and kernel crashing. Computations within the blocks are parallelized so this method goes way faster
# than every other loading methods.
#      /!\ This implementation uses asyncio (way faster) which requires its own event loop to run : if you launch this code from a raw terminal, 
# there should be no problem, but if you try to launch it from an IDE (PyCharm, VSCode, Spyder...), think of specifying to your IDE to launch it 
# in a raw terminal instead of the default console (which leads to a RuntimeError)
#    - 'direct_process' : No subdivisition of the data is made beforehand which generally leads to memory overconsumption and kernel crashes
# if the amount of pixel to load is too high (depending on your available memory). If you want to load big amount of data, you should use
# 'block_process', which is also faster. This method is essentially used for debug purposes.

load_pixel_process = 'block_process'

save = True # If True, save TICOI results to a netCDF file
save_mean_velocity = True # Save a .tiff file with the mean reulting velocities, as an example

## --------- ------------ Data availability parameters --------------------- ##
index = ['monthly_graph',
         'all_data', 'median_baseline', 'max_leap_frog',
         'mini_month', 'mini_3month', 'mean_month', 'median_month',
         'mini_season', 'min_all_season']

# Adjust index list looking at the dependencies of each index
if ('mini_season' in index or 'min_all_season' in index) and 'availability_maps' not in index:
    index.append('availability_maps')
if ('availability_maps' in index or 'monthly_graph' in index) and 'monthly' not in index: 
    index.append('monthly')
for ind in index:
    if len(ind.split('_')) >= 2 and ind.split('_')[0] in ['mini', 'mean', 'median'] and ind[-5:] == 'month':
        if 'monthly' not in index:
            index.append('monthly')
        break

## ------------------------------ Data selection --------------------------- ##
# List of the paths where the data cubes are stored
cube_names = ['nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01225_y03675_all_filt-multi.nc',]
               # 'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc']
flag_file = None  # Path where the flag file is stored
mask_file = 'nathan/Tests_MB/Areas/Full_MB/mask/Full_MB.shp' # Path where the mask file is stored
path_save = 'nathan/Tests_MB/' # Path where to store the results
result_fn = 'test'# Name of the netCDF file to be created

proj = 'EPSG:32632'  # EPSG system of the given coordinates

assign_flag = False
flags = None
if assign_flag:
    flags = xr.open_dataset(flag_file)
    flags.load()

regu = '1accelnotnull'
coef = 100
solver = 'LSMR_ini'

## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {'chunks': {}, 
               'conf': False, # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
               'subset': [330200, 331000, 5075500, 5076800] , # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
               'buffer': None, # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
               'pick_date': ['2015-01-01', '2023-01-01'], # Select dates ([min, max] or None to select all)
               'pick_sensor': None, # Select sensors (None to select all)
               'pick_temp_bas': None, # Select temporal baselines ([min, max] in days or None to select all)
               'proj': proj, # EPSG system of the given coordinates
               'verbose': False # Print information throughout the loading process 
               }

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {'smooth_method': 'gaussian', # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
                  's_win': 3, # Size of the spatial window
                  't_win': 90, # Time window size for 'ewma' smoothing
                  'sigma': 3, # Standard deviation for 'gaussian' filter
                  'order': 3, # Order of the smoothing function
                  'unit': 365, # 365 if the unit is m/y, 1 if the unit is m/d
                  'delete_outliers': 'vvc_angle', # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle') 
                  'flags': flags, # Divide the data in several areas where different methods should be used
                  'regu': regu, # Regularization method.s to be used (for each flag if flags is not None)
                  'solver': solver, # Solver for the inversion
                  'proj': proj, # EPSG system of the given coordinates
                  'velo_or_disp': 'velo', # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
                  'verbose': True # Print information throughout the filtering process 
                  }

## ---------------- Parameters for the pixel loading part ------------------ ##
load_pixel_kwargs = {'regu': regu, # Regularization method.s to be used (for each flag if flags is not None)
                     'solver': solver, # Solver for the inversion
                     'proj': proj, # EPSG system of the given coordinates
                     'interpolation_load_pixel': 'nearest', # Interpolation method used to load the pixel when it is not in the dataset
                     'visual': False, # Plot results along the way
                     'verbose':False # Print information throughout TICOI processing
                     }

## ----------------------- Parallelization parameters ---------------------- ##
nb_cpu = 12 # Number of CPU to be used for parallelization
block_size = 0.5 # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method

if not os.path.exists(path_save):
    os.mkdir(path_save)
    

# %%========================================================================= #
#                                 CUBE LOADING                                #
# =========================================================================%% #

start = [time.time()]

# Load the first cube
cube = cube_data_class()
cube.load(cube_names[0], pick_date=load_kwargs['pick_date'], chunks=load_kwargs['chunks'], conf=load_kwargs['conf'], 
          pick_sensor=load_kwargs['pick_sensor'], pick_temp_bas=load_kwargs['pick_temp_bas'], proj=load_kwargs['proj'], 
          subset=load_kwargs['subset'], verbose=load_kwargs['verbose'])

# Several cubes have to be merged together
filenames = [cube.filename]
if len(cube_names) > 1:
    for n in range(1, len(cube_names)):
        cube2 = cube_data_class()
        subset = load_kwargs['subset']
        res = cube.ds['x'].values[1] - cube.ds['x'].values[0] # Resolution of the main data
        cube2.load(cube_names[n], pick_date=load_kwargs['pick_date'], chunks=load_kwargs['chunks'], 
                   conf=load_kwargs['conf'], pick_sensor=load_kwargs['pick_sensor'], pick_temp_bas=load_kwargs['pick_temp_bas'], 
                   proj=load_kwargs['proj'], verbose=load_kwargs['verbose'],
                   subset=[subset[0]-res, subset[1]+res, subset[2]-res, subset[3]+res] if subset is not None else None)
        filenames.append(cube2.filename)
        # Align the new cube to the main one (interpolate the coordinate and/or reproject it)
        cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
        cube.merge_cube(cube2) # Merge the new cube to the main one
    del cube2

# Mask some of the data
if mask_file is not None:
    cube.mask_cube(mask_file)

stop = [time.time()]
print(f'[data_availability] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ')
print(f'[data_availability] Data loading took {round(stop[0] - start[0], 2)} s')


# %%========================================================================= #
#                                 PIXEL LOADING                               #
# =========================================================================%% #

start.append(time.time())

# The data cube is subdivided in smaller cubes computed one after the other in a synchronous manner (uses async)
if load_pixel_process == 'block_process': 
    result = process_blocks_refine(cube, nb_cpu=nb_cpu, block_size=block_size, load_only=True, preData_kwargs=preData_kwargs, inversion_kwargs=load_pixel_kwargs)

# Direct loading of the whole cube
elif load_pixel_process == 'direct_process':    
    # Progression bar
    xy_values = itertools.product(cube.ds['x'].values, cube.ds['y'].values)
    xy_values_tqdm = tqdm(xy_values, total=len(cube.ds['x'].values)*len(cube.ds['y'].values), mininterval=0.5)
    
    # Parallelized loading of the pixels (list of arrays converted to dataframes)
    print('[data_availability] Loading pixels...')
    result = Parallel(n_jobs=nb_cpu, verbose=0)(
                delayed(cube.load_pixel)(i, j, proj=load_pixel_kwargs['proj'], interp=load_pixel_kwargs['interpolation_load_pixel'],
                             solver=load_pixel_kwargs['solver'], regu=load_pixel_kwargs['regu'], rolling_mean=None, 
                             visual=load_pixel_kwargs['visual'], verbose=load_pixel_kwargs['verbose'])
                for i, j in xy_values_tqdm)

result = [pd.DataFrame(data={'date1': r[0][0][:, 0], 'date2': r[0][0][:, 1],
                             'vx': r[0][1][:, 0], 'vy': r[0][1][:, 1],
                             'errorx': r[0][1][:, 2], 'errory': r[0][1][:, 3],
                             'temporal_baseline': r[0][1][:, 4]}) for r in result]

# Pixel filtering
if len(index) > 0:
    # Remove pixels with no data
    empty = list(filter(bool, [d if not result[d].empty and result[d][result[d]['vx'] == 0].shape[0] == 0 else False for d in range(len(result))]))
    prev_length = len(result)
    positions = np.unique(np.array([(i, j) for i in cube.ds['x'].values for j in cube.ds['y'].values]), axis=0)[empty, :]
    data = [result[i] for i in empty]
    
    # Coordinates
    resolution = int(cube.ds['x'].values[1] - cube.ds['x'].values[0])
    longitude = np.array([positions[i][0] for i in range(len(data))])
    latitude = np.array([positions[i][1] for i in range(len(data))])
    coord_data = {'resolution': resolution,
                  'longitude': longitude,
                  'latitude': latitude,
                  'long_data': (longitude - np.min(longitude)).astype(int) // resolution,
                  'lat_data': (latitude - np.min(latitude)).astype(int) // resolution,
                  'nb_long_data': int(np.max(longitude) - np.min(longitude)) // resolution + 1,
                  'nb_lat_data': int(np.max(latitude) - np.min(latitude)) // resolution + 1}
else:
    print('[data_availability] index list is empty, no indices or availability map will be computed...')

stop.append(time.time())
print(f'[data_availability] Pixel loading and filtering took {round(stop[1] - start[1], 1)} s')


# %%========================================================================= #
#                          INDICES USING ONLY RAW DATA                        #
# =========================================================================%% #

# Generate a .tiff file summarizing an index value for all pixels of the data
# The indices using this function only use the raw data (not the monthly availability computed afterwards
# nor the data availability maps)
def generate_tiff_index_map(index, data, period, coord_data, driver, srs, path, parallel=True, nb_cpu=12, dtype='float32'):
    # Generate index map according to the selected method
    index_map = np.zeros([coord_data['nb_long_data'], coord_data['nb_lat_data']], dtype=dtype)
    data_tqdm = tqdm(data, total=len(data), mininterval=0.5)
    if parallel: # Use parallelization (for ressource-consuming indices)
        index_map[coord_data['long_data'], coord_data['lat_data']] = Parallel(n_jobs=nb_cpu, verbose=0)(
            delayed(index)(d, period=period) for d in data_tqdm)
    else: # Direct process of the indices
        index_map[coord_data['long_data'], coord_data['lat_data']] = [index(d) for d in data_tqdm]

    # Generate tiff file
    if dtype == 'float32':
        tiff = driver.Create(path, index_map.shape[0], index_map.shape[1], 1, gdal.GDT_Float32)
    elif dtype == 'int16':
        tiff = driver.Create(path, index_map.shape[0], index_map.shape[1], 1, gdal.GDT_Int16)
    tiff.SetGeoTransform([np.min(coord_data['longitude']), coord_data['resolution'], 0, 
                          np.max(coord_data['latitude']), 0, -coord_data['resolution']])
    tiff.GetRasterBand(1).WriteArray(index_map.T)
    tiff.SetProjection(srs.ExportToWkt())
    tiff = None

# Just summing up all the available data at the point over the whole considered period
def all_data(data, period=None):
    return data.shape[0]

# Median temporal baseline (gap in days between the two acquisition dates) of the data at the point
def median_baseline(data, period=None):
    return data['temporal_baseline'].median()

# All the dates (start and end dates for each data) are concatenated in a single list, which is then sorted. This index returns the
# maximum gap between two dates (leap frog) as an approximation of the lowest redundancy in the data.
# This index can be computed periodically (see the definition in introduction) with the period parameter.
def max_leap_frog(data, period=None):
    if data.empty:
        return 0

    if period is not None:
        data = data.copy()
        data = data.set_index(data['date1'] + (data['date2'] - data['date1']) // 2).sort_index()
        years = pd.date_range(start=data['date1'].min() - pd.DateOffset(years=1) if data['date1'].min().month < 6 else data['date1'].min(),
                              end=data['date2'].max() + pd.DateOffset(years=1) if data['date2'].max().month > 6 else data['date2'].max(), 
                              freq='YS', inclusive='neither')
        season_indices = [max_leap_frog(data[(data.index >= years[i]) & (data.index <= years[i+period])]) for i in range(len(years)-period)]
        return np.min(season_indices) if len(season_indices) > 0 else 0
    
    return np.max(np.diff(np.sort(np.concatenate([data['date1'], data['date2']])))
                  .astype('timedelta64[D]').astype('int'))


# To generate GeoTiff files
driver = gdal.GetDriverByName('GTiff')
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS(proj)

tic = time.time()

is_raw_data_indices = False
for ind in index:
    if 'all_data' in ind or 'median_baseline' in ind or 'max_leap_frog' in ind:   
        period = None
        if ind[-1] == 'y' and ind[-2] in '0123456789':
            period = int(ind.split('_')[-1][:-1])
        
        if 'all_data' in ind:
            print('[data_availability] Computing all_data index...')
            generate_tiff_index_map(all_data, data, period, coord_data, driver, srs, f'{path_save}index_all_data.tiff', 
                                    dtype='float32', parallel=False)
        if 'median_baseline' in ind:
            print('[data_availability] Computing median_baseline index...')
            generate_tiff_index_map(median_baseline, data, period, coord_data, driver, srs, f'{path_save}index_median_baseline.tiff', 
                                    dtype='int16', parallel=False)
        if 'max_leap_frog' in ind:
            print('[data_availability] Computing max_leap_frog index...')
            generate_tiff_index_map(max_leap_frog, data, period, coord_data, driver, srs, f'{path_save}index_max_leap_frog.tiff', 
                                    dtype='int16', parallel=False)
            
        is_raw_data_indices = True

if is_raw_data_indices:
    tac = time.time()
    print(f'[data_availability] Computing indices based on raw data only took {round(tac-tic, 1)} s')

driver = None


# %%========================================================================= #
#                          MONTHLY DATA AVAILABILITY                          #
# =========================================================================%% #
# How many data are covering each month of the period of measurements ? We use a ponderation strategy
# where each data count for one and is ponderated depending on the month.s it covers: a data with a big temporal
# baseline will only count a litle for each of the months it covers but on many months, where a data with a small
# temporal baseline will count much for the few months it covers.

# Each data has a value of one which is ponderated by the monthly coverage (how many days the data has in the considered month)
# divided by the temporal baseline of the data
if 'monthly' in index:
    tic = time.time()
    
    # Apply a monthly ponderation to each of the data and sum it up for each month 
    def monthly_ponderation_pixel(dt, month):
        pond = []
        for m in month:
            d = dt.copy()
            # Select the period of the data which covers part or full month m
            d.loc[d['date1'] < m, 'date1'] = m
            d.loc[d['date2'] > m + pd.DateOffset(months=1), 'date2'] = m + pd.DateOffset(months=1)            
            diff = (d['date2'] - d['date1']).dt.days # Of how many days is month m cover by the data ?
            diff = diff.where(diff >= 0, 0)
            pond.append((diff / d['temporal_baseline']).sum(axis=0)) # Compute ponderation for each data and sum it
        return pond
    
    # Parallelized computation
    print('[data_availability] Computing monthly data availability...')
    monthly = pd.DataFrame(index=[(i, j) for (i, j) in positions],
                           columns=pd.date_range(start=np.min([d['date1'].min() for d in data]), 
                                                 end=np.max([d['date2'].max() for d in data]), freq='MS'))    
    data_tqdm = tqdm(data, total=len(data), mininterval=0.5)
    monthly_pond = np.array(Parallel(n_jobs=nb_cpu, verbose=0)(
                                delayed(monthly_ponderation_pixel)(dt, monthly.columns) for dt in data_tqdm))
    
    monthly.loc[:, :] = monthly_pond
    monthly = monthly.astype('float32')
    monthly.to_csv(f'{path_save}monthly_data.csv')
    
    # Availability DataFrame
    availability = monthly.sum(axis=0)
    availability = availability.reindex(pd.date_range(start=availability.index[0], end=availability.index[-1], freq='MS'), fill_value=0)
    availability.index = [f'{availability.index[i].year}-{"0" if availability.index[i].month < 10 else ""}{availability.index[i].month}' for i in range(len(availability.index))]

    tac = time.time()
    print(f'[data_availability] Computing monthly data availability with the ponderation approach took {round(tac-tic, 1)} s')

    # Plot a bar plot of the average monthly data availability of the data on the loaded cube (whole cube or subset)
    if 'monthly_graph' in index:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(availability.index, availability.values / len(result))
        ax.set_xticks(ticks=range(0, len(availability.index), 3), labels=availability.index[::3], rotation='vertical')
        ax.set_xlabel('Date', fontsize=16)
        ax.set_ylabel('Amount of available velocity data\nfor one pixel', fontsize=16)
        if load_kwargs['subset'] is not None:
            fig.suptitle(f'Monthly availability of data for subset {load_kwargs["subset"]}\n' + 
                         f'of cube {cube_names[0].split("/")[-1][:-18]}', fontsize=18)
        else:
            fig.suptitle(f'Monthly availability of data for cube {cube_names[0].split("/")[-1][:-18]}', fontsize=18)
        plt.subplots_adjust(bottom=0.2)
        fig.savefig(f'{path_save}monthly_availability.png')
        
        ax.set_yscale('log')
        fig.savefig(f'{path_save}monthly_availability_log.png')
        print(f'[data_availability] Monthly availability graphs were saved to {path_save}monthly_availability(_log).png')
        
    del availability
    
    
# %%========================================================================= #
#                            DATA AVAILABILITY MAP                            #
# =========================================================================%% #
# Generate maps for each season (winter, spring, summer, autumn) in order to see (or not) seasonal variability
# among data availability
# Seasons are defined as follow (which does not correspond to the exact definition of the seasons):
#    - winter : december, january and february
#    - spring : march, april and may
#    - summer : june, july and august
#    - autumn : september, october and november

if 'availability_maps' in index:
    tic = time.time()
    
    # Initialisation
    seasons = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1])
    winter = np.zeros([coord_data['nb_long_data'], coord_data['nb_lat_data']], dtype=np.float32) # Available data during winter (Decembre 1st -> February 28th or 29th)
    spring = np.zeros([coord_data['nb_long_data'], coord_data['nb_lat_data']], dtype=np.float32) # Available data during spring (Marche 1st -> May 31th)
    summer = np.zeros([coord_data['nb_long_data'], coord_data['nb_lat_data']], dtype=np.float32) # Available data during summer (June 1st -> August 31ths)
    autumn = np.zeros([coord_data['nb_long_data'], coord_data['nb_lat_data']], dtype=np.float32) # Available data during autumn (September 1st -> November 30th)
    long_data = (longitude - np.min(longitude)).astype(int) // resolution
    lat_data = (latitude - np.min(latitude)).astype(int) // resolution
    
    # Fill monthly map
    monthly_map = monthly.groupby(monthly.columns.month, axis=1).sum()
    winter[coord_data['long_data'], coord_data['lat_data']] = monthly_map[monthly_map.columns.values[seasons == 1]].sum(axis=1)
    spring[coord_data['long_data'], coord_data['lat_data']] = monthly_map[monthly_map.columns.values[seasons == 2]].sum(axis=1)
    summer[coord_data['long_data'], coord_data['lat_data']] = monthly_map[monthly_map.columns.values[seasons == 3]].sum(axis=1)
    autumn[coord_data['long_data'], coord_data['lat_data']] = monthly_map[monthly_map.columns.values[seasons == 4]].sum(axis=1)
    
    # Generate GeoTiff files (each band = a season)
    driver = gdal.GetDriverByName('GTiff')
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS(proj)
    
    tiff = driver.Create(f'{path_save}seasonal_data_availability.tiff', winter.shape[0], winter.shape[1], 4, gdal.GDT_Float32)
    tiff.SetGeoTransform([np.min(longitude), resolution, 0, np.max(latitude), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(winter.T)
    tiff.GetRasterBand(2).WriteArray(spring.T)
    tiff.GetRasterBand(3).WriteArray(summer.T)
    tiff.GetRasterBand(4).WriteArray(autumn.T)
    tiff.SetProjection(srs.ExportToWkt())
    
    driver = None
    tiff = None
    
    tac = time.time()
    print(f'[data_availability] Generating the availability maps took {round(tac-tic, 1)} s')
    print(f'[data_availability] Availability map was saved to {path_save}seasonal_data_availability.tiff')
 
    
# %%========================================================================= #
#         INDICES USING MONTHLY AVAILABILITY AND/OR AVAILABILITY MAPS         #
# =========================================================================%% #

## --------------- ------- Using monthly availability ---------------------- ## 
# Generate a .tiff file summarizing an index value for all pixels of the data
# The indices using this function require the monthly availability to be computed (but not any data availability map)
def generate_tiff_index_map_from_monthly_availability(index, monthly, positions, n_month, coord_data, path):
    index_map = index(monthly, positions, coord_data, n_month=n_month)
    tiff = driver.Create(path, index_map.shape[0], index_map.shape[1], 1, gdal.GDT_Float32)
    tiff.SetGeoTransform([np.min(coord_data['longitude']), coord_data['resolution'], 0, 
                          np.max(coord_data['latitude']), 0, -coord_data['resolution']])
    tiff.GetRasterBand(1).WriteArray(index_map.T)
    tiff.SetProjection(srs.ExportToWkt())
    tiff = None

# Minimum amount of data available on a sliding window of n months over the whole period of the measurements
def mini_nmonth(monthly, positions, coord_data, n_month=1):
    index_map = np.zeros([coord_data['nb_long_data'], coord_data['nb_lat_data']], dtype='float32')
    if n_month > 1:
        monthly = monthly.rolling(window=n_month, axis=1).sum()[monthly.columns[n_month-1:]]
    index_map[coord_data['long_data'], coord_data['lat_data']] = [monthly.loc[[(i, j)]].squeeze().min() for (i, j) in positions]
    return index_map

# Month (or selection of n months among the 12 months) with the lowest average of available data on the whole dataset
def mean_nmonth(monthly, positions, coord_data, n_month=1):
    index_map = np.zeros([coord_data['nb_long_data'], coord_data['nb_lat_data']], dtype='float32')
    monthly = monthly.groupby(monthly.columns.month, axis=1).mean()
    if n_month > 1:
        monthly = monthly.rolling(window=n_month, axis=1).sum()[monthly.columns[n_month-1:]]
    index_map[coord_data['long_data'], coord_data['lat_data']] = monthly_map.min(axis=1).to_list()
    return index_map

# Month (or selection of n months among the 12 months) with the lowest median of available data on the whole dataset
def median_nmonth(monthly, positions, coord_data, n_month=1):
    index_map = np.zeros([coord_data['nb_long_data'], coord_data['nb_lat_data']], dtype='float32')
    monthly = monthly.groupby(monthly.columns.month, axis=1).median()
    if n_month > 1:
        monthly = monthly.rolling(window=n_month, axis=1).sum()[monthly.columns[n_month-1:]]
    index_map[coord_data['long_data'], coord_data['lat_data']] = monthly_map.min(axis=1).to_list()
    return index_map

## ---------------------- Using data availability maps --------------------- ##
# Generate a .tiff file summarizing an index value for all pixels of the data
# The indices using this function require the data availability maps to be computed
def generate_tiff_index_map_from_availability_map(index, maps, coord_data, driver, srs, path):
    index_map = index(maps)
    tiff = driver.Create(path, index_map.shape[0], index_map.shape[1], 1, gdal.GDT_Float32)
    tiff.SetGeoTransform([np.min(coord_data['longitude']), coord_data['resolution'], 0, 
                          np.max(coord_data['latitude']), 0, -coord_data['resolution']])
    tiff.GetRasterBand(1).WriteArray(index_map.T)
    tiff.SetProjection(srs.ExportToWkt())
    tiff = None

# Each pixel take the value of the season with the lowest amount of available data
def mini_season(maps):
    winter, spring, summer, autumn = maps
    return np.min(np.stack([winter, spring, summer, autumn]), axis=0)

# Season with the least data / all data : index to evaluate the repartition of the data over the year
def min_all_season(maps):
    winter, spring, summer, autumn = maps
    return np.nan_to_num(4 * np.min(np.stack([winter, spring, summer, autumn], axis=0), axis=0)
                / (winter + spring + summer + autumn))


# To generate GeoTiff files
driver = gdal.GetDriverByName('GTiff')
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS(proj)

# Monthly indices ('mini_nmonth', 'mean_nmonth' or 'median_nmonth' where n is a number or empty)
tic = time.time()

methods = {'mini': mini_nmonth,
           'mean': mean_nmonth,
           'median': median_nmonth}
is_monthly_indices = False
for ind in index:
    if len(ind.split('_')) >= 2 and ind.split('_')[0] in ['mini', 'mean', 'median'] and ind[-5:] == 'month':
        n_month = 1
        if len(ind.split('_')[1]) > 5:
            n_month = int(ind.split('_')[1][:-5])
        method = ind.split('_')[0]
        
        print(f'[data_availability] Computing {method}_{n_month}month index...')
        generate_tiff_index_map_from_monthly_availability(methods[method], monthly, positions, n_month, coord_data, 
                                                          f'{path_save}index_{method}_{n_month}month.tiff')
        
        is_monthly_indices = True

if is_monthly_indices:
    tac = time.time()
    print(f'[data_availability] Computing indices based on monthly data availability took {round(tac-tic, 1)} s')

    
# Seasonal indices
if 'mini_season' in index or 'min_all_season' in index:
    tic = time.time()
    
    maps = (winter, spring, summer, autumn)   
    if 'mini_season' in index:
        print('[data_availability] Computing mini_season index...')
        generate_tiff_index_map_from_availability_map(mini_season, maps, coord_data, driver, srs, 
                                                      f'{path_save}index_mini_season.tiff')
    if 'min_all_season' in index:
        print('[data_availability] Computing min_all_season index...')
        generate_tiff_index_map_from_availability_map(min_all_season, maps, coord_data, driver, srs,
                                                      f'{path_save}index_min_all_season.tiff')
        
    tac = time.time()
    print(f'[data_availability] Computing indices based on seasonal data availability maps took {round(tac-tic, 1)} s')

driver = None

stop.append(time.time())
print(f'[data_availability] Overall processing took {round(stop[2] - start[0], 0)} s')