#!/usr/bin/env python3

"""
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
for one pixel. An additional seasonality analysis is implemented, the idea is to match a sine to TICOI results
(fixed frequency or not).

Author: Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft as fft
import scipy.signal as signal
from scipy.optimize import curve_fit
from sklearn.metrics import root_mean_squared_error

from ticoi.core import interpolation_core, inversion_core, visualisation
from ticoi.cube_data_classxr import cube_data_class

# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

## ------------------------------ Data selection --------------------------- ##
# Path.s to the data cube.s (can be a list of str to merge several cubes, or a single str)
cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc'
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "pixel"))}/'  # Path where to store the results
proj = "EPSG:32632"  # EPSG system of the given coordinates

i, j = 342537.1, 5092253.3  # Point (pixel) where to carry on the computation

## --------------------------- Main parameters ----------------------------- ##
regu = 1  # Regularization method to be used
coef = 300  # Regularization coefficient to be used
solver = "LSMR_ini"  # Solver for the inversion
unit = 365  # 1 for m/d, 365 for m/y
result_quality = (
    "X_contribution"  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
)

## ----------------------- Visualization parameters ------------------------ ##
verbose = True  # Print information throughout TICOI processing
visual = False  # Plot information along the way
save = False  # Save the results or not
# Visualisation options
option_visual = [
    "original_velocity_xy",
    "original_magnitude",
    "X_magnitude_zoom",
    "X_magnitude",
    "X_zoom",
    "X",
    "vv_quality",
    "vxvy_quality",
    "Residu_magnitude",
    "Residu",
    "X_z",
    "Y_contribution",
    "direction",
]
vmax = [False, False]  # Vertical limits for the plots

## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "buffer": [i, j, 500],  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": ["2015-01-01", "2023-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": proj,  # EPSG system of the given coordinates
    "verbose": verbose,
}  # Print information throughout the loading process

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "smooth_method": "gaussian",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for 'ewma' smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": "vvc_angle",  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "regu": regu,  # Regularization method to be used
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "velo",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "verbose": verbose,
}  # Print information throughout the filtering process

## ---------------- Parameters for the pixel loading part ------------------ ##
load_pixel_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": coef,
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "interp": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "visual": visual,  # Plot results along the way
    "verbose": verbose,
}  # Print information throughout TICOI processing

## --------------------------- Inversion parameters ------------------------ ##
inversion_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": coef,  # Regularization coefficient to be used
    "solver": solver,  # Solver for the inversion
    "conf": False,  # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "iteration": True,  # Allow the inversion process to make several iterations
    "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
    "threshold_it": 0.1,  # Threshold to test the stability of the results between each iteration, used to stop the process
    "weight": True,  # If True, use apriori weights
    "detect_temporal_decorrelation": True,  # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
    "linear_operator": None,  # Perform the inversion using this specific linear operator
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "visual": visual,  # Plot results along the way
    "verbose": verbose,
}  # Print information throughout TICOI processing

## ----------------------- Interpolation parameters ------------------------ ##
interpolation_kwargs = {
    "interval_output": 30,  # Temporal baseline of the time series resulting from TICOI (after interpolation)
    "redundancy": 5,  # Redundancy in the interpolated time series in number of days, no redundancy if None
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "visual": visual,  # Plot results along the way
    "vmax": vmax,  # vmin and vmax of the legend
    "verbose": verbose,
}  # Print information throughout TICOI processing

## ------------------- Parameters for seasonality analysis ----------------- ##
# Is the periodicity frequency imposed to 1/365.25 (one year seasonality) ?
impose_frequency = True
# Add several sinus at different freqs (1/365.25 and harmonics (2/365.25, 3/365.25...) if impose_frequency is True)
#   (only available for impose_frequency = True for now)
several_freq = 1
# Compute also the best matching sinus to raw data, for comparison
raw_seasonality = True
# Filter to use in the first place
# 'highpass' : apply a bandpass filter between low frequencies (reject variations over several years (> 1.5 y))
# and the Nyquist frequency to ensure Shanon theorem
# 'lowpass' : or apply a lowpass filter only (to Nyquist frequency) : risk of tackling an interannual trend (long period)
filt = "highpass"
# Method used to compute local variations
# 'rolling_7d' : median of the std of the data centered in +- 3 days around each central date
# 'uniform_7d' : median of the std of the data centered in +- 3 days around dates constantly distributed every redundnacy
# days -- BEST
# 'uniform_all' : median of the std of each data covering the dates, which are constantly distributed every redundancy days
# 'residu' : standard deviation of the data previously subtracted by TICOI results (ground truth) = standard deviation of the "noise"
local_var_method = "uniform_7d"

# Create a subfolder if it does not exist
if not os.path.exists(path_save):
    os.mkdir(path_save)


# %% ======================================================================== #
#                                DATA LOADING                                 #
# =========================================================================%% #

start = [time.time()]

# Load data cube.s
cube = cube_data_class()
cube.load(cube_name, **load_kwargs)

stop = [time.time()]
print(f"[Data loading] Loading the data cube.s took {round((stop[-1] - start[-1]), 4)} s")
print(f"[Data loading] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ")

start.append(time.time())

# Filter the cube (compute rolling_mean for regu=1accelnotnull)
obs_filt, _ = cube.filter_cube(**preData_kwargs)
# Load pixel data
data, mean, dates_range = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)

cube2_date1 = cube.date1_().tolist()
cube2_date1.remove(np.min(cube2_date1))
start_date_interpol = np.min(cube2_date1)
last_date_interpol = np.max(cube.date2_())

stop.append(time.time())
print(f"[Data loading] Loading the pixel took {round((stop[-1] - start[-1]), 4)} s")


# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #

start.append(time.time())

# Proceed to inversion
A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, mean=mean, **inversion_kwargs)

stop.append(time.time())
print(f"[Inversion] Inversion took {round((stop[-1] - start[-1]), 4)} s")

# Plot the results of the inversion
if visual:
    visualisation(
        dataf,
        result,
        option_visual,
        path_save,
        A=A,
        dataf=dataf,
        unit=inversion_kwargs["unit"],
        show=True,
        figsize=(12, 6),
    )
if save:
    result.to_csv(f"{path_save}/ILF_result.csv")


# %% ======================================================================== #
#                              INTERPOLATION                                  #
# =========================================================================%% #

start.append(time.time())

if interpolation_kwargs["interval_output"] == False:
    interpolation_kwargs["interval_output"] = 1
start_date_interpol = np.min(np.min(cube.date2_()))
last_date_interpol = np.max(np.max(cube.date2_()))

# Proceed to interpolation
dataf_lp = interpolation_core(
    result,
    path_save=path_save,
    data=dataf,
    first_date_interpol=start_date_interpol,
    last_date_interpol=last_date_interpol,
    **interpolation_kwargs,
)

stop.append(time.time())
print(f"[Interpolation] Interpolation took {round((stop[-1] - start[-1]), 4)} s")

if save:
    dataf_lp.to_csv(f"{path_save}/RLF_result.csv")


# %% ======================================================================== #
#                           BEST MATCHING SINUS                               #
# =========================================================================%% #
# Compute the best periodicity of the signal (frequency with the highest TF)
# If 1/best_freq is around 365 days, there might be an annual periodicity (the
# significance is evaluated in the next section MATCH SINE CURVE)

start.append(time.time())

if not os.path.exists(f"{path_save}Fourier/"):
    os.mkdir(f"{path_save}Fourier/")

## ------------------------- Preparation of the data ----------------------- ##
dataf_lp = dataf_lp.dropna()

# Get dates and velocities from TICOI results
dates_c = dataf_lp["First_date"] + (dataf_lp["Second_date"] - dataf_lp["First_date"]) // 2  # Central dates
dates = (
    dates_c - dataf_lp["First_date"].min()
).dt.days.to_numpy()  # Number of days to the reference day (first day of acquisition at the point)
vv = np.sqrt(dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2).to_numpy()  # Velocity magnitude
vv_c = vv - np.mean(vv)  # Centered velocities

# Format raw data in a pandas dataframe
dataff = pd.DataFrame(
    data={
        "date1": data[0][:, 0],
        "date2": data[0][:, 1],
        "vx": data[1][:, 0],
        "vy": data[1][:, 1],
        "errorx": data[1][:, 2],
        "errory": data[1][:, 3],
        "temporal_baseline": data[1][:, 4],
    }
)
dataff["vx"] = dataff["vx"] * unit / dataff["temporal_baseline"]
dataff["vy"] = dataff["vy"] * unit / dataff["temporal_baseline"]
dataff["vv"] = np.sqrt(dataff["vx"] ** 2 + dataff["vy"] ** 2)
dataff.index = dataff["date1"] + (dataff["date2"] - dataff["date1"]) // 2

N = len(dates)
Ts = dates[1] - dates[0]
print(f"[ticoi_pixel_demo] Sampling period after interpolation : {Ts} days")

# Filter the results...
if filt == "highpass":  # ...to remove low frequencies (general trend over several years)
    b, a = signal.butter(4, [1 / (1.5 * 365), 1 / (2.001 * Ts)], "bandpass", fs=1 / Ts, output="ba")
    vv_filt = signal.filtfilt(b, a, vv_c)
elif filt == "lowpass":  # ...to ensure Shanon critrion
    sos = signal.butter(4, 1 / (2.001 * Ts), "lowpass", fs=1 / Ts, output="sos")
    vv_filt = signal.sosfilt(sos, vv_c)
else:  # Don't filter
    vv_filt = vv_c

if impose_frequency:
    fig, axe = plt.subplots(figsize=(12, 6))
else:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
    axe = ax[0]
axe.plot(dates_c, vv_c, "blue", label="Before filtering")
axe.plot(dates_c, vv_filt, "red", label="After filtering")
axe.set_xlabel("Centered velocity [m/y]", fontsize=16)
axe.set_ylabel("Central date", fontsize=16)
axe.set_title("Effect of filtering", fontsize=16)
axe.legend(loc="lower left")


## --------------------------- Best matching sinus ------------------------- ##
# To find the first maximum of the sinus
def right_phi(phi):
    if phi >= 0 and phi < np.pi / 2:
        return np.pi / 2 - phi
    elif phi >= np.pi / 2 and phi < 3 * np.pi / 2:
        return 5 * np.pi / 2 - phi
    return 5 * np.pi / 2 - phi


# Frequency is imposed to 1/365.25 day-1 (1 year-1)
if impose_frequency:

    def sine_fconst(t, *args, freqs=1, f=1 / 365.25):
        sine = args[0] * np.sin(2 * np.pi * f * t + args[1])
        for freq in range(1, freqs):
            sine += args[2 * freq] * np.sin(2 * np.pi * (freq + 1) * f * t + args[2 * freq + 1])
        return sine + args[-1]

    f = 1 / 365.25

    ##  Find the best matching sinus to TICOI results
    if several_freq is None:
        several_freq = 1
    guess = np.concatenate([np.concatenate([[np.max(vv_filt) - np.min(vv_filt), 0] for _ in range(several_freq)]), [0]])
    popt, pcov = curve_fit(lambda t, *args: sine_fconst(t, *args, freqs=several_freq), dates, vv_filt, p0=guess)

    # Parameters
    A = [popt[2 * freq] for freq in range(several_freq)]
    phi = [popt[2 * freq + 1] for freq in range(several_freq)]
    phi = [phi[i] if phi[i] > 0 else phi[i] + 2 * np.pi for i in range(several_freq)]
    off = popt[-1]
    sine = sine_fconst(dates, *popt, freqs=several_freq)

    # information about the periodicity of TICOI results
    print(
        "[ticoi_pixel_demo] Amplitude of the best matching sinus to TICOI results: "
        f"{[round(abs(A[i]), 2) for i in range(several_freq)] if len(A) > 1 else round(abs(A[0]), 2)} m/y"
    )
    first_max_day = [
        pd.Timedelta(int((right_phi(phi[i]) + (np.pi if A[i] < 0 else 0)) / (2 * np.pi * f)), "D")
        + dataf_lp["First_date"].min()
        for i in range(several_freq)
    ]
    max_day = [
        (first_max_day[i] - pd.Timestamp(year=first_max_day[i].year, month=1, day=1)) for i in range(several_freq)
    ]
    print(
        f"                   Maximum at day {[max_day[i].days for i in range(several_freq)] if len(max_day) > 1 else max_day[0].days}"
    )
    print(f"                   Mean value of {round(np.mean(sine + np.mean(vv)), 1)} m/y")
    print(f"                   RMSE : {round(root_mean_squared_error(sine, vv_filt), 2)} m/y")

    if raw_seasonality:
        ##  Find the best matching sinus to raw data
        dates_raw = (dataff.index - dataf_lp["First_date"].min()).days.to_numpy()
        raw_c = dataff["vv"] - dataff["vv"].mean()
        guess_raw = np.concatenate(
            [np.concatenate([[np.max(raw_c) - np.min(raw_c), 0] for _ in range(several_freq)]), [0]]
        )
        popt_raw, pcov_raw = curve_fit(
            lambda t, *args: sine_fconst(t, *args, freqs=several_freq), dates_raw, raw_c, p0=guess_raw
        )

        # Parameters
        A_raw = [popt_raw[2 * freq] for freq in range(several_freq)]
        phi_raw = [popt_raw[2 * freq + 1] for freq in range(several_freq)]
        phi_raw = [phi_raw[i] if phi_raw[i] > 0 else phi_raw[i] + 2 * np.pi for i in range(several_freq)]
        off_raw = popt_raw[-1]
        sine_raw = sine_fconst(dates_raw, *popt_raw, freqs=several_freq)

        # information about the periodicity of raw data
        print(
            "[ticoi_pixel_demo] Amplitude of the best matching sinus to raw data: "
            f"{[round(abs(A_raw[i]), 2) for i in range(several_freq)] if len(A_raw) > 1 else round(abs(A_raw[0]), 2)} m/y"
        )
        first_max_day_raw = [
            pd.Timedelta(int((right_phi(phi_raw[i]) + (np.pi if A_raw[i] < 0 else 0)) / (2 * np.pi * f)), "D")
            + dataf_lp["First_date"].min()
            for i in range(several_freq)
        ]
        max_day_raw = [
            (first_max_day_raw[i] - pd.Timestamp(year=first_max_day_raw[i].year, month=1, day=1))
            for i in range(several_freq)
        ]
        print(
            f"                   Maximum at day {[max_day_raw[i].days for i in range(several_freq)] if len(max_day_raw) > 1 else max_day_raw[0].days}"
        )
        print(f'                   Mean value of {round(np.mean(sine_raw + dataff["vv"].mean()), 1)} m/y')
        print(f"                   RMSE : {round(root_mean_squared_error(sine_raw, raw_c), 2)} m/y")

# Frequency is to be found via a TF transform
else:
    # Apply a Hanning window
    window = signal.windows.hann(N)
    ax[1].plot(dates_c, vv_filt * window, "blue", label="With Hanning windowing")
    ax[1].plot(dates_c, vv_filt, "black", label="Without windowing")
    ax[1].set_xlabel("Centered velocity [m/y]", fontsize=16)
    ax[1].set_ylabel("Central date", fontsize=16)
    ax[1].set_title("Effect of Hanning windowing", fontsize=16)
    ax[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(f"{path_save}Fourier/Windowing_Filtering.png")

    # TFD
    n = 64 * N
    vv_tf = fft.rfft(vv_filt, n=n)
    vv_win_tf = fft.rfft(vv_filt * window, n=n)
    freq = fft.rfftfreq(n, d=Ts)

    # Plot the TF
    plt.figure(figsize=(12, 6))
    plt.plot(freq, 2 / N * np.abs(vv_tf), "blue", label="TF without windowing")
    plt.plot(freq, 2 / N * np.abs(vv_win_tf), "red", label="TF after Hanning windowing")
    plt.vlines(
        [i / 365 for i in range(1, 4)],
        0,
        1.1 * 2 / N * max(np.max(np.abs(vv_tf)), np.max(np.abs(vv_win_tf))),
        color="black",
        label="365d periodicity",
    )
    plt.xlim([0, 0.01])
    plt.ylim([0, 1.1 * 2 / N * max(np.max(np.abs(vv_tf)), np.max(np.abs(vv_win_tf)))])
    plt.xlabel("Frequency [day-1]", fontsize=16)
    plt.ylabel("Amplitude [m/y]", fontsize=16)
    plt.legend(loc="best")
    plt.title(f"Fourier Transform of the TICOI-resulting velocities at point ({i}, {j})", fontsize=16)
    plt.savefig(f"{path_save}Fourier/TF.png")

    # Best matching sinus
    def sine(t, A, f, phi, off):
        return A * np.sin(2 * np.pi * f * t + phi) + off

    # Initial guess from the TF
    guess = np.array(
        [
            np.max(2 / N * np.abs(vv_win_tf)),
            freq[np.argmax(np.abs(vv_win_tf))],
            np.angle(vv_win_tf)[np.argmax(np.abs(vv_win_tf))],
            np.mean(vv_win_tf),
        ],
        dtype="float",
    )

    popt, pcov = curve_fit(sine, dates, vv_filt, p0=guess)
    A, f, phi, off = popt
    sine = sine(dates, A, f, phi, off)

    if phi < 0:
        phi += 2 * np.pi

    print(f"[ticoi_pixel_demo] Period of the best matching sinus : {round(1/f, 0)} days")
    print(f"                   Amplitude : {round(abs(A), 1)} m/y")
    first_max_day = (
        pd.Timedelta(int((right_phi(phi) + (np.pi if A < 0 else 0)) / (2 * np.pi * f)), "D") + dataf_lp["date1"].min()
    )
    max_day = first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1)
    print(f"                   Maximum at day {max_day.days}")
    print(f"                   RMSE : {round(root_mean_squared_error(sine, vv_filt))} m/y")

## ------------------------------ Plot the data  --------------------------- ##
# Plot raw data, TICOI results and the best matching sinus
plt.figure(figsize=(12, 6))
plt.plot(dataff.index, dataff["vv"], linestyle="", marker="x", markersize=2, color="orange", label="Raw data")
plt.plot(dates_c, vv, "black", alpha=0.5, label="TICOI velocities")
plt.plot(dates_c, vv_filt + np.mean(vv), "red", alpha=0.7, label="Filtered TICOI velocities")
if impose_frequency and raw_seasonality:
    plt.plot(dataff.index, sine_raw + dataff["vv"].mean(), linewidth=3, label="Best matching sinus to raw data")
plt.plot(dates_c, sine + np.mean(vv), "cyan", linewidth=3, label="Best matching sinus to TICOI results")
plt.vlines(
    pd.date_range(start=first_max_day[0], end=dataf_lp["Second_date"].max(), freq=f"{int(1/f)}D"),
    np.min(vv),
    np.max(vv),
    "black",
    label="Maximum",
)
plt.xlabel("Central dates", fontsize=16)
plt.ylabel("Velocity", fontsize=16)
plt.legend(loc="best")
plt.title("Best matching sinus around an annual seasonality")
plt.savefig(f"{path_save}Fourier/matching_sine.png")

## ------------------------------- AtoVar index ---------------------------- ##
# Compute local variations
if local_var_method == "rolling_7d":
    var = dataff["vv"].rolling(window="7D", center=True).std(ddof=0).drop_duplicates().dropna().median().item()

elif local_var_method.split("_")[0] == "uniform":
    period_between_dates = (
        np.diff(np.sort(np.concatenate([dataff["date1"], dataff["date2"]]))).astype("timedelta64[D]").astype("int")
    )
    min_period = np.min(period_between_dates[period_between_dates > 0])
    var_dates = pd.date_range(start=dataff["date1"].min(), end=dataff["date2"].max(), freq=f"{min_period}D")
    local_var = pd.Series(index=var_dates)

    if local_var_method == "uniform_7d":
        for date in var_dates:
            local_var[date] = dataff.loc[
                (dataff.index > date - pd.Timedelta("3D")) & (dataff.index < date + pd.Timedelta("3D")), "vv"
            ].std(ddof=0)
    elif local_var_method == "uniform_all":
        for date in var_dates:
            local_var[date] = dataff.loc[(dataff["date1"] < date) & (dataff["date2"] > date), "vv"].std(ddof=0)

    var = local_var[local_var > 0].dropna().median()

elif local_var_method == "residu":
    dataf_lp.index = dataf_lp["date1"] + (dataf_lp["date2"] - dataf_lp["date1"]) // 2
    dataf_lp["vv"] = np.sqrt(dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2)
    dataf_lp = dataf_lp.reindex(index=np.unique(dataff.index)).interpolate().dropna()
    dataff = dataff[dataff.index >= dataf_lp.index[0]]
    dataff_vv_c = dataff["vv"] - dataf_lp["vv"]
    var = dataff_vv_c.std(ddof=0)

    plt.figure(figsize=(12, 6))
    plt.plot(dataff.index, dataff["vv"], linestyle="", marker="x", markersize=2, color="orange")
    plt.plot(dataff.index, dataff_vv_c + dataf_lp["vv"].mean(), linestyle="", marker="x", markersize=2, color="red")
    plt.plot(dataf_lp.index, dataf_lp["vv"], linestyle="", marker="x", markersize=2, color="blue")
    plt.hlines(
        [np.mean(vv) + var, np.mean(vv) - var, np.mean(vv)], np.min(dataff.index), np.max(dataff.index), color="black"
    )
    plt.savefig(f"{path_save}Fourier/residu.png")

# Amplitude to median local variations factor (AtoVar index)
AtoVar = max(0, 1 - var / abs(popt[0]))

print(f"[ticoi_pixel_demo] Local variations : {round(var, 2)} m/y")
print(f"[ticoi_pixel_demo] Amplitude to local variations factor : {round(AtoVar, 2)}")

stop.append(time.time())
print(f"[ticoi_pixel_demo] Fourier analysis took {round((stop[-1] - start[-1]), 4)} s")

plt.show()


# %% ======================================================================== #
#                               ANNUAL CURVES                                 #
# =========================================================================%% #
# Superpose the curves for each year

dates_c = dataf_lp["First_date"] + (dataf_lp["Second_date"] - dataf_lp["First_date"]) // 2  # Central dates
vv = np.sqrt(dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2).to_numpy()  # Velocity magnitude

years = np.unique(np.array([dates_c.iloc[i].year for i in range(dates_c.size)]))
months_start = {
    "January": 1,
    "February": 32,
    "March": 60,
    "April": 91,
    "May": 121,
    "June": 152,
    "July": 182,
    "August": 213,
    "September": 244,
    "October": 274,
    "November": 305,
    "December": 335,
}

fig, ax = plt.subplots(figsize=(12, 4))
for y in years:
    dates = dates_c[[dates_c.iloc[i].year == y for i in range(dates_c.size)]] - pd.Timestamp(year=y, month=1, day=1)
    dates = np.array([dates.iloc[i].days for i in range(dates.size)])
    vv_y = vv[[dates_c.iloc[i].year == y for i in range(dates_c.size)]]
    ax.plot(dates, vv_y, linestyle=":", linewidth=3, label=str(y))

ax.set_xticks(list(months_start.values()), list(months_start.keys()))
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
ax.set_xlabel("Day of the year", fontsize=14)
ax.set_ylabel("Velocity magnitude [m/y]", fontsize=14)
ax.legend(loc="best")
ax.set_title("Superposed annual TICOI resulting velocities", fontsize=16)

plt.show()

stop.append(time.time())
print(f"[ticoi_pixel_demo] Overall processing took {round((stop[-1] - start[0]), 4)} s")
