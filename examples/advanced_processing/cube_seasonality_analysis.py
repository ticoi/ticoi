'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method to compute entire data cubes.
An aditional seasonality analysis is implemented, by matching a sinus to TICOI results for each pixel of the considered cube/subset,
thus generating maps with the amplitude of the best matching sinus, the position of its first maximum and an index comparing its amplitude
to the local variations of the raw data.s

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import time
import os
import warnings
import itertools
import xarray as xr
import numpy as np
import pandas as pd
import scipy.fft as fft
import scipy.signal as signal

from osgeo import gdal, osr
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import curve_fit

from ticoi.core import process_blocks_refine, process
from ticoi.cube_data_classxr import cube_data_class

# %%========================================================================= #
#                                   PARAMETERS                                #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------- Choose TICOI cube processing method ----------------- ##
# Choose the TICOI cube processing method you want to use :
#    - 'block_process' (recommended) : This implementation divides the data in smaller data cubes processed one after the other in a synchronous manner,
# in order to avoid memory overconsumption and kernel crashing. Computations within the blocks are parallelized so this method goes way faster
# than every other TICOI processing methods.
#      /!\ This implementation uses asyncio (way faster) which requires its own event loop to run : if you launch this code from a raw terminal, 
# there should be no problem, but if you try to launch it from some IDE (like Spyder), think of specifying to your IDE to launch it 
# in a raw terminal instead of the default console (which leads to a RuntimeError)
#    - 'direct_process' : No subdivisition of the data is made beforehand which generally leads to memory overconsumption and kernel crashes
# if the amount of pixel to compute is too high (depending on your available memory). If you want to process big amount of data, you should use
# 'block_process', which is also faster. This method is essentially used for debug purposes.
#   - 'load' : The  TICOI cube was already calculated before, load it by giving the cubes to be loaded in a dictionary like {name: path} (at least 
# 'raw' and 'interp' must be given)

TICOI_process = 'load'

save = True # If True, save TICOI results to a netCDF file
save_mean_velocity = True # Save a .tiff file with the mean reulting velocities, as an example

## ------------------------------ Data selection --------------------------- ##
# Path.s to the data cube.s (can be a list of str to merge several cubes, or a single str, 
# cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_example.nc'
# If TICOI_process is 'load', must be a dictionary like {name: path} to load existing cubes and name them (path can be a list of str or a single str)
cube_name = {'raw': f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_example.nc',
             'interp': f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))}/Argentiere_example_interp.nc'}
flag_file = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_displacement_S2_flags.nc' # Path to flags file
mask_file = None # Path to mask file (.shp file) to mask some of the data on cube
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))}/' # Path where to store the results
result_fn = 'Argentiere_example' # Name of the netCDF file to be created (if save is True)

proj = 'EPSG:32632'  # EPSG system of the given coordinates

# Divide the data in several areas where different methods should be used
assign_flag = True
flags = None
if assign_flag:
    flags = xr.open_dataset(flag_file)
    flags.load()

# Regularization method.s to be used (for each flag if flags is not None)
regu = {0: 1, 1: '1accelnotnull'} # With flags (0: stable ground, 1: glaciers)
# regu = '1accelnotnull' # Without flags
# Regularization coefficient.s to be used (for each flag if flags is not None)
coef = {0: 500, 1: 200} # With flags (0: stable ground, 1: glaciers)
# coef = 200 # Without flags
solver = 'LSMR_ini' # Solver for the inversion

# What results must be returned from TICOI processing (not used for TICOI_process='load')
#   - 'raw' for loading raw data at pixels too
#   - 'invert' for the results of the inversion
#   - 'interp' for the results of the interpolation
returned = ['invert', 'interp']

## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {'chunks': {}, 
               'conf': False, # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
               'subset': None, # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
               'buffer': None, # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
               'pick_date': ['2015-01-01', '2023-01-01'], # Select dates ([min, max] or None to select all)
               'pick_sensor': None, # Select sensors (None to select all)
               'pick_temp_bas': None, # Select temporal baselines ([min, max] in days or None to select all)
               'proj': proj, # EPSG system of the given coordinates
               'mask_file': mask_file, # Path to mask file (.shp file) to mask some of the data on cube
               'verbose': False} # Print information throughout the loading process
               
## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {'smooth_method': 'gaussian', # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
                  's_win': 3, # Size of the spatial window
                  't_win': 90, # Time window size for 'ewma' smoothing
                  'sigma': 3, # Standard deviation for 'gaussian' filter
                  'order': 3, # Order of the smoothing function
                  'unit': 365, # 365 if the unit is m/y, 1 if the unit is m/d
                  'delete_outliers': 'median_angle', # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle') 
                  'flags': flags, # Divide the data in several areas where different methods should be used
                  'regu': regu, # Regularization method.s to be used (for each flag if flags is not None)
                  'solver': solver, # Solver for the inversion
                  'proj': proj, # EPSG system of the given coordinates
                  'velo_or_disp': 'velo', # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
                  'verbose': True} # Print information throughout the filtering process 

## ---------------- Inversion and interpolation parameters ----------------- ##
inversion_kwargs = {'regu': regu, # Regularization method.s to be used (for each flag if flags is not None)
                    'coef': coef, # Regularization coefficient.s to be used (for each flag if flags is not None)
                    'solver': solver, # Solver for the inversion
                    'flags': flags, # Divide the data in several areas where different methods should be used
                    'conf': False, # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
                    'unit': 365, # 365 if the unit is m/y, 1 if the unit is m/d
                    'delete_outliers': 'vvc_angle', # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle') 
                    'proj': proj, # EPSG system of the given coordinates
                    'interpolation_load_pixel': 'nearest', # Interpolation method used to load the pixel when it is not in the dataset
                    
                    'iteration': True, # Allow the inversion process to make several iterations
                    'nb_max_iteration': 10, # Maximum number of iteration during the inversion process
                    'threshold_it': 0.1, # Threshold to test the stability of the results between each iteration, used to stop the process
                    'apriori_weight': True, # If True, use apriori weights
                    'detect_temporal_decorrelation': True, # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
                    'linear_operator': None, # Perform the inversion using this specific linear operator
                    
                    'interval_output': 30, 
                    'option_interpol': 'spline', # Type of interpolation ('spline', 'spline_smooth', 'nearest')
                    'redundancy': 30, # Redundancy in the interpolated time series in number of days, no redundancy if None
                    
                    'result_quality': 'X_contribution', # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                    'visual': False, # Plot results along the way
                    'path_save': path_save, # Path where to store the results
                    'verbose': False} # Print information throughout TICOI processing
                    
## ----------------------- Parallelization parameters ---------------------- ##
nb_cpu = 12 # Number of CPU to be used for parallelization
block_size = 0.5 # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method

## ------------------- Parameters for seasonality analysis ----------------- ##
# Is the periodicity frequency imposed to 1/365.25 (one year seasonality) ?
impose_frequency = True
# Filter to use in the first place
# 'highpass' : apply a bandpass filter between low frequencies (reject variations over several years (> 1.5 y))
# and the Nyquist frequency to ensure Shanon theorem
# 'lowpass' : or apply a lowpass filter only (to Nyquist frequency) : risk of tackling an interannual trend (long period)
filt = 'highpass'
# Method used to compute local variations
# 'rolling_7d' : median of the std of the data centered in +- 3 days around each central date
# 'uniform_7d' : median of the std of the data centered in +- 3 days around dates constantly distributed every redundnacy 
# days -- BEST
# 'uniform_all' : median of the std of each data covering the dates, which are constantly distributed every redundancy days
# 'residu' : standard deviation of the data previously substracted by TICOI results (ground truth) = standard deviation of the "noise"
local_var_method = 'uniform_7d'

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA LOADING                                #
# =========================================================================%% #

start, stop = [], []
start.append(time.time())

# Load the cube.s
cube = cube_data_class()

if TICOI_process == 'load': 
    assert 'raw' in cube_name.keys(), "'raw' data must be loaded by giving the associated path in cube_name"
    cube.load(cube_name['raw'], **load_kwargs)
else:
    cube.load(cube_name, **load_kwargs)

# Load raw data at pixels if required
if TICOI_process == 'load':
    print('[Data loading] Loading raw data...')
    data_raw = process_blocks_refine(cube, nb_cpu=nb_cpu, block_size=block_size, returned=['raw'], inversion_kwargs=inversion_kwargs)
    data_raw = [pd.DataFrame(data={'date1': raw[0][0][:, 0], 'date2': raw[0][0][:, 1],
                                    'vx': raw[0][1][:, 0], 'vy': raw[0][1][:, 1],
                                    'errorx': raw[0][1][:, 2], 'errory': raw[0][1][:, 3],
                                    'temporal_baseline': raw[0][1][:, 4]}) for raw in data_raw]

# Prepare interpolation dates
cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
first_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())

inversion_kwargs.update({'first_date_interpol': first_date_interpol, 'last_date_interpol': last_date_interpol})

stop.append(time.time())
print(f'[Data loading] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ')
print(f'[Data loading] Data loading took {round(stop[-1] - start[-1], 3)} s')


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

start.append(time.time())

cube_interp, cube_invert = None, None

# The data cube is subdivided in smaller cubes computed one after the other in a synchronous manner (uses async)
# TICOI computation is then parallelized among those cubes
if TICOI_process == 'block_process':
    result = process_blocks_refine(cube, nb_cpu=nb_cpu, block_size=block_size, returned=returned, 
                                   preData_kwargs=preData_kwargs, inversion_kwargs=inversion_kwargs)

# Direct computation of the whole TICOI cube
elif TICOI_process == 'direct_process':
    # Preprocessing of the data (compute rolling mean for regu='1accelnotnull', delete outliers...)
    obs_filt = cube.filter_cube(**preData_kwargs)
    
    # Progression bar
    xy_values = itertools.product(cube.ds['x'].values, cube.ds['y'].values)
    xy_values_tqdm = tqdm(xy_values, total=len(cube.ds['x'].values)*len(cube.ds['y'].values), mininterval=0.5)
    
    # Main processing of the data with TICOI algorithm, individually for each pixel
    result = Parallel(n_jobs=nb_cpu, verbose=0)(delayed(process)(cube, i, j, returned=returned, obs_filt=obs_filt,
                                                                 **inversion_kwargs) for i, j in xy_values_tqdm)

elif TICOI_process == 'load':    
    # Load inversion results
    if 'invert' in cube_name.keys():
        cube_invert = cube_data_class()
        cube_invert.load(cube_name['invert'], **load_kwargs)
        
    # Load interpolation results
    if 'interp' in cube_name.keys():
        cube_interp = cube_data_class()
        cube_interp.load(cube_name['interp'], **load_kwargs)
        
        print('[TICOI processing] Loading TICOI data...')
        result = process_blocks_refine(cube_interp, nb_cpu=nb_cpu, block_size=block_size, returned=['raw'], inversion_kwargs=inversion_kwargs)
        result = [pd.DataFrame(data={'First_date': r[0][0][:, 0], 'Second_date': r[0][0][:, 1],
                                     'vx': r[0][1][:, 0], 'vy': r[0][1][:, 1],
                                     'errorx': r[0][1][:, 2], 'errory': r[0][1][:, 3],
                                     'temporal_baseline': r[0][1][:, 4]}) for r in result]
            
stop.append(time.time())
print(f'[TICOI processing] TICOI {"processing" if TICOI_process != "load" else "loading"} took {round(stop[-1] - start[-1], 0)} s')


# %%========================================================================= #
#                                INITIALISATION                               #
# =========================================================================%% #

if TICOI_process != 'load':    
    # Write down some informations about the data and the TICOI processing performed
    if save:
        start.append(time.time())
        sensor_array = np.unique(cube.ds['sensor'])
        sensor_strings = [str(sensor) for sensor in sensor_array]
        sensor = ', '.join(sensor_strings)
        
        if len(cube_name) > 1:
            source = f'Temporal inversion on cubes {", ".join(cube.filename)} using TICOI'
        else:
            source = f'Temporal inversion on cube {cube.filename} using TICOI'
        source += f' with a selection of dates among {load_kwargs["pick_date"]},' if load_kwargs['pick_date'] is not None else '' + \
                  f' with a selection of the temporal baselines among {load_kwargs["pick_temp_bas"]}' if load_kwargs['pick_temp_bas'] is not None else ''
        
        if inversion_kwargs['apriori_weight']:
            source += ' and apriori weight'
        source += f'. The regularisation coefficient is {inversion_kwargs["coef"]}.'
        if 'interp' in returned:
            source_interp = source + f'The interpolation method used is {inversion_kwargs["option_interpol"]}.'
            source_interp += f'The interpolation baseline is {inversion_kwargs["interval_output"]} days.'
            source_interp += f'The temporal spacing (redundancy) is {inversion_kwargs["redundancy"]} days.'
    
        stop.append(time.time())    
        print(f'[Writing results] Initialisation took {round(stop[-1] - start[-1], 3)} s')


# %%========================================================================= #
#                                WRITING RESULTS                              #
# =========================================================================%% #

start.append(time.time())
if TICOI_process != 'load':
    # Save TICO.I results to a netCDF file, thus obtaining a new data cube    
    several = (type(returned) == list and len(returned) >= 2)
    j = 1 if 'raw' in returned else 0
    if 'invert' in returned:
        cube_invert = cube.write_result_tico([result[i][j] for i in range(len(result))] if several else result, source, sensor, 
                                             filename=f'{result_fn}_invert' if several else result_fn, 
                                             savepath=path_save if save else None, 
                                             result_quality=inversion_kwargs['result_quality'], verbose=inversion_kwargs['verbose'])
    if 'interp' in returned:
        cube_interp = cube.write_result_ticoi([result[i][j+1] for i in range(len(result))] if several else result, source_interp, sensor, 
                                              filename=f'{result_fn}_interp' if several else result_fn, 
                                              savepath=path_save if save else None, 
                                              result_quality=inversion_kwargs['result_quality'], verbose=inversion_kwargs['verbose'])

# Plot the mean velocity as an example
if save_mean_velocity and cube_interp is not None:    
    mean_vv = np.sqrt(cube_interp.ds['vx'].mean(dim='mid_date') ** 2 + cube_interp.ds['vy'].mean(dim='mid_date') ** 2).to_numpy().astype(np.float32)
    mean_vv = np.flip(mean_vv.T, axis=0)
    
    driver = gdal.GetDriverByName('GTiff')
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('EPSG:32632')
    
    dst_ds_temp = driver.Create(f'{path_save}{result_fn}_mean_velocity.tiff', mean_vv.shape[1], mean_vv.shape[0], 1, gdal.GDT_Float32)
    if TICOI_process != 'load' or (TICOI_process == 'load' and 'raw' in cube_name.keys()):
        resolution = int(cube.ds['x'].values[1] - cube.ds['x'].values[0])
        dst_ds_temp.SetGeoTransform([np.min(cube.ds['x'].values), resolution, 0, np.min(cube.ds['y'].values), 0, resolution])
    else:
        resolution = int(cube_interp.ds['x'].values[1] - cube_interp.ds['x'].values[0])
        dst_ds_temp.SetGeoTransform([np.min(cube_interp.ds['x'].values), resolution, 0, np.min(cube_interp.ds['y'].values), 0, resolution])
    dst_ds_temp.GetRasterBand(1).WriteArray(mean_vv)
    dst_ds_temp.SetProjection(srs.ExportToWkt())
    
    dst_ds_temp = None
    driver = None
        
if save or (save_mean_velocity and cube_interp is not None):
    print(f'[Writing results] Results saved at {path_save}')

stop.append(time.time())
if TICOI_process != 'load':
    print(f'[Writing results] Writing cube to netCDF file took {round(stop[-1] - start[-1], 3)} s')


# %%========================================================================= #
#                               PERIODICITY MAPS                              #
# =========================================================================%% #
# Match a sinus to the data (frequency which can be fixed to 1/365.25, amplitude, phase which gives the date of the f)

start.append(time.time())

def match_sine(d, impose_frequency=True, filt=None):
    
    '''
       Match a sine curve to TICOI results to look for a periodicity among the velocities. The period can either 
    be set to 365.25 days, or estimated along with the other parameters (amplitude, phase, offset).
    
       :param d: pandas dataframe of the data at the considered pixel
       :param impose_frequency: bool, whether we should impose the frequency to 1/365.25 or not (default True)
       :param filt: which filter to use before processing the sinus ('highpass', 'lowpass' or None, default None)
    '''
    
    d = d.dropna()
    dates = (d['First_date'] + (d['Second_date'] - d['First_date']) // 2 - d['First_date'].min()).dt.days.to_numpy()
    N = len(dates)
    if N <= 4: return np.nan, np.nan
    vv = np.sqrt(d['vx']**2 + d['vy']**2).to_numpy()
    Ts = dates[1] - dates[0]
    
    # Filtering
    if filt == 'highpass':
        b, a = signal.butter(4, [1/(1.5*365), 1/(2.001*Ts)], 'bandpass', fs=1/Ts, output='ba')
        vv_filt = signal.filtfilt(b, a, vv - np.mean(vv))
    elif filt == 'lowpass':
        sos = signal.butter(4, 1/(2.001*Ts), 'lowpass', fs=1/Ts, output='sos')
        vv_filt = signal.sosfilt(sos, vv - np.mean(vv))
    else:
        vv_filt = vv
    
    # Frequency is set to 1/365.25 (one year)
    if impose_frequency:
        def sine_fconst(t, A, phi, off):
            f = 1/365.25 # One year
            return A * np.sin(2*np.pi*f*t + phi) + off
        
        guess = [np.max(vv_filt) - np.min(vv_filt), 0, 0]
        try:
            popt, pcov = curve_fit(sine_fconst, dates, vv, p0=guess)
            A, phi, off = popt
            f = 1/365.25
        except RuntimeError:
            A, f, phi = np.nan, np.nan, np.nan
    # Frequency is to be found too     
    else:
        n = 64*N
        window = signal.windows.hann(N)
        vv_win_tf = fft.rfft(vv_filt * window, n=n)
        freq = fft.rfftfreq(n, d=Ts)
        
        # Match a sinus to the data
        def sine(t, A, f, phi, off):
            return A * np.sin(2*np.pi*f*t + phi) + off
        
        # Initial guess of the best matching sinus parameters
        guess = np.array([np.max(2/N*np.abs(vv_win_tf)), freq[np.argmax(np.abs(vv_win_tf))], 
                np.angle(vv_win_tf)[np.argmax(np.abs(vv_win_tf))], np.mean(vv)], dtype='float')
    
        try:
            popt, pcov = curve_fit(sine, dates, vv, p0=guess)
            A, f, phi, off = popt
        except RuntimeError:
            A, f, phi = np.nan, np.nan, np.nan
    
    # To find the day associated to the maximum of the sine curve
    def right_phi(phi):
        if phi >= 0 and phi < np.pi/2: return np.pi/2 - phi
        elif phi >= np.pi/2 and phi < 3*np.pi/2: return 5*np.pi/2 - phi
        return 5*np.pi/2 - phi
    
    if phi < 0: phi += 2*np.pi
    first_max_day  = pd.Timedelta(int((right_phi(phi) + (np.pi if A < 0 else 0)) / (2*np.pi*f)), 'D') +  d['First_date'].min()
    max_day = (first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1)).days
    
    return 1/f, A, max_day # Period, amplitude and phase of the periodicity

def AtoVar(A, raw, dataf_lp, local_var_method='uniform_7d'):
    
    '''
       Compute Amplitude to local VARiations index, which compares the amplitude of the best matching sinus to the standard
    deviation of the noise using one of the four given methods.
    
       :param A: float, amplitude of the best matchning sinus
       :param raw: list, raw data
       :param dataf_lp: list of pandas dataframes, TICOI results
       :param local_var_method: str, method to be used to process the local variations
    '''
    
    if A == np.nan: return np.nan
    
    # Format raw data to velocities
    raw['vx'] = raw['vx'] * preData_kwargs['unit'] / raw['temporal_baseline']
    raw['vy'] = raw['vy'] * preData_kwargs['unit'] / raw['temporal_baseline']
    raw['vv'] = np.sqrt(raw['vx'] ** 2 + raw['vy'] ** 2)
    raw.index = raw['date1'] + (raw['date2'] - raw['date1']) // 2
    
    # Compute local variations
    if local_var_method == 'rolling_7d':
        var = raw['vv'].rolling(window='7D', center=True).std(ddof=0).drop_duplicates().dropna().median().item()
        
    elif local_var_method.split('_')[0] == 'uniform':
        period_between_dates = np.diff(np.sort(np.concatenate([raw['date1'], raw['date2']]))).astype('timedelta64[D]').astype('int')
        min_period = np.min(period_between_dates[period_between_dates > 0])
        var_dates = pd.date_range(start=raw['date1'].min(), end=raw['date2'].max(), freq=f'{min_period}D')
        local_var = pd.Series(index=var_dates)
        
        if local_var_method == 'uniform_7d':
            for date in var_dates:
                local_var[date] = raw.loc[(raw.index > date - pd.Timedelta('3D')) & (raw.index < date + pd.Timedelta('3D')), 'vv'].std(ddof=0)
        elif local_var_method == 'uniform_all':
            for date in var_dates:
                local_var[date] = raw.loc[(raw['date1'] < date) & (raw['date2'] > date), 'vv'].std(ddof=0)
                
        var = local_var[local_var > 0].dropna().median()  
        
    elif local_var_method == 'residu':
        dataf_lp.index = dataf_lp['First_date'] + (dataf_lp['Second_date'] - dataf_lp['First_date']) // 2
        dataf_lp['vv'] = np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2)
        dataf_lp = dataf_lp.reindex(index=np.unique(dataf.index)).interpolate().dropna()
        dataf = dataf[dataf.index >= dataf_lp.index[0]]
        dataff_vv_c = dataf['vv'] - dataf_lp['vv']
        var = dataff_vv_c.std(ddof=0)
      
    return max(0, 1 - var / abs(A))
    
driver = gdal.GetDriverByName('GTiff')
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS(proj)

# Remove pixels with no data
empty = list(filter(bool, [d if not (result[d].empty and result[d][result[d]['vx'] == 0].shape[0] == 0) else False for d in range(len(result))]))
positions = np.array(list(itertools.product(cube.ds['x'].values, cube.ds['y'].values)))[empty, :]
usefull_result = [result[i] for i in empty]
usefull_data_raw = [data_raw[i] for i in empty]

# Coordinates informations
resolution = int(cube.ds['x'].values[1] - cube.ds['x'].values[0])
longitude = np.array([positions[i][0] for i in range(len(positions))])
latitude = np.array([positions[i][1] for i in range(len(positions))])
long_data = (longitude - np.min(cube.ds['x'].values)).astype(int) // resolution
lat_data = (latitude - np.min(cube.ds['y'].values)).astype(int) // resolution

####  Best matching sinus map (amplitude and phase, and period if not fixed)
print('[Fourier analysis] Computing periodicity map...')
if not impose_frequency:
    period_map = np.empty([cube.nx, cube.ny])
    period_map[:,:] = np.nan
amplitude_map = np.empty([cube.nx, cube.ny])
amplitude_map[:,:] = np.nan
AtoVar_map = np.empty([cube.nx, cube.ny])
AtoVar_map[:,:] = np.nan
peak_map = np.empty([cube.nx, cube.ny])
peak_map[:, :] = np.nan

result_tqdm = tqdm(usefull_result, total=len(usefull_result), mininterval=0.5)
match_res = np.array(Parallel(n_jobs=nb_cpu, verbose=0)(delayed(match_sine)(d) for d in result_tqdm))
if not impose_frequency:
    period = np.abs(match_res[:, 0])
    period_map[long_data, lat_data] = np.sign(period - 365) * (1 - np.minimum(period, 365) / np.maximum(period, 365))
amplitude_map[long_data, lat_data] = np.abs(match_res[:, 1])
peak_map[long_data, lat_data] = match_res[:, 2]
raw_tqdm = tqdm(zip(match_res[:, 1], usefull_data_raw, usefull_result), total=len(usefull_data_raw), mininterval=0.5)
AtoVar_map[long_data, lat_data] = Parallel(n_jobs=nb_cpu, verbose=0)(delayed(AtoVar)(A, raw, dataf_lp, local_var_method) 
                                                    for A, raw, dataf_lp in raw_tqdm)

# Save the maps to a .tiff file with two bands (one for period, and one for amplitude)
if impose_frequency:
    tiff = driver.Create(f'{path_save}matching_sine_map_fconst_{filt}_{local_var_method}.tiff', 
                         amplitude_map.shape[0], amplitude_map.shape[1], 3, gdal.GDT_Float32)
    tiff.SetGeoTransform([np.min(cube.ds['x'].values), resolution, 0, np.max(cube.ds['y'].values), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(np.flip(amplitude_map.T, axis=0))
    tiff.GetRasterBand(2).WriteArray(np.flip(peak_map.T, axis=0))
    tiff.GetRasterBand(3).WriteArray(np.flip(AtoVar_map.T, axis=0))
else:
    tiff = driver.Create(f'{path_save}matching_sine_map_{filt}_{local_var_method}.tiff', 
                         period_map.shape[0], period_map.shape[1], 4, gdal.GDT_Float32)
    tiff.SetGeoTransform([np.min(cube.ds['x'].values), resolution, 0, np.max(cube.ds['y'].values), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(np.flip(period_map.T, axis=0))
    tiff.GetRasterBand(2).WriteArray(np.flip(amplitude_map.T, axis=0))
    tiff.GetRasterBand(3).WriteArray(np.flip(peak_map.T, axis=0))
    tiff.GetRasterBand(4).WriteArray(np.flip(AtoVar_map.T, axis=0))
tiff.SetProjection(srs.ExportToWkt())

# Needed to effectively save the .tiff file
tiff = None
driver = None 

stop.append(time.time())
print(f'[Fourier analysis] Computing periodicity maps took {round(stop[-1] - start[-1], 0)} s')
print(f'[Overall] Overall processing took {round(stop[-1] - start[0], 0)} s')