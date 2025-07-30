import copy
from typing import List

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.lines as malines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft as fft
import scipy.signal as signal
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# %%========================================================================= #
#                           DATAFRAME_DATA OBJECT                             #
# =========================================================================%% #


class DataframeData:
    """Object to define a pd.Dataframe storing velocity observations"""

    def __init__(self, dataf: pd.DataFrame = pd.DataFrame()):
        self.dataf = dataf

    def set_temporal_baseline_central_date_offset_bar(self):
        """Set temporal baselines ('temporal_baseline'), centrale date (date_cori), and offset bar ('offset_bar'), used for plotting"""

        delta = self.dataf["date2"] - self.dataf["date1"]  # temporal baseline of the observations
        self.dataf["date_cori"] = np.asarray(self.dataf["date1"] + delta // 2).astype("datetime64[D]")  # central date
        try:
            self.dataf["temporal_baseline"] = np.asarray((delta).dt.days).astype(
                "int"
            )  # temporal baseline as an integer
        except TypeError:
            self.dataf["temporal_baseline"] = np.array([delta[i].days for i in range(delta.shape[0])])
        self.dataf["offset_bar"] = delta // 2  # to plot the temporal baseline of the plots

    def set_vx_vy_invert(self, type_data: str = "invert", conversion: int = 365):
        """
        Convert displacements into velocity

        :param type_data: [str] [default is "invert"] --- Type of the data to be converted to velocities (generally "invert" or "obs_filt")
        :param conversion: [int] [default is 365] --- Conversion factor: 365 is the unit of the velocity is m/y and 1 if it is m/d
        """

        if "result_dx" in self.dataf.columns:
            self.dataf = self.dataf.rename(columns={"result_dx": "vx", "result_dy": "vy"})
        self.dataf["vx"] = self.dataf["vx"] / self.dataf["temporal_baseline"] * conversion
        self.dataf["vy"] = self.dataf["vy"] / self.dataf["temporal_baseline"] * conversion

    def set_vx_vy_my(self, type_data: str = "obs_filt", conversion: int = 365):
        if "result_dx" in self.dataf.columns:
            self.dataf = self.dataf.rename(columns={"result_dx": "vx", "result_dy": "vy"})
        self.dataf["vx"] = self.dataf["vx"] * conversion
        self.dataf["vy"] = self.dataf["vy"] * conversion

    def set_vv(self):
        """Set velocity magnitude variable (here vv) in the dataframe"""

        self.dataf["vv"] = np.round(
            np.sqrt((self.dataf["vx"] ** 2 + self.dataf["vy"] ** 2).astype("float")), 2
        )  # Compute the magnitude of the velocity

    def set_minmax(self):
        """Set the attribute minimum and maximum fir vx, vy, and possibly vv"""

        self.vxymin = int(self.dataf["vx"].min())
        self.vxymax = int(self.dataf["vx"].max())
        self.vyymin = int(self.dataf["vy"].min())
        self.vyymax = int(self.dataf["vy"].max())
        if "vv" in self.dataf.columns:
            self.vvymin = int(self.dataf["vv"].min())
            self.vvymax = int(self.dataf["vv"].max())


# %%========================================================================= #
#                             PIXEL_CLASS OBJECT                              #
# =========================================================================%% #


class PixelClass:
    """Object class to store the data on a given pixel"""

    def __init__(
        self,
        save: bool = False,
        path_save: str = "",
        show: bool = True,
        figsize: tuple[int, int] = (10, 6),
        unit: str = "m/y",
        A: np.ndarray | None = None,
        dataobs: pd.DataFrame | None = None,
    ):
        """
        Initialize the pixel_class object with general plotting parameters, or set them to default values if no parameters are given.

        :param save: [bool] [default is False] --- Save the figures to path_save
        :param path_save: [str] [default is ""] --- Path where to save the figures if save is True
        :param show: [bool] [default is True] --- Plot the figures
        :param figsize: [tuple<int, int>] [default is (10, 6)] --- Size of the figures
        :param unit: [str] [default is "m/y"] --- Unit of the velocities ("m/y" or "m/d")
        :param A: [np.array | None] [default is None] --- Design matrix
        :param dataobs: [pd.DataFrame | None] [default is None] --- Observation data
        """

        self.dataobs = dataobs
        self.datainvert = None
        self.datainterp = None
        self.dataobsfilt = None
        self.save = save
        self.show = show
        self.path_save = path_save
        self.figsize = figsize
        self.unit = unit
        self.A = A

    def set_data_from_pandas_df(
        self, dataf_ilf: pd.DataFrame, type_data: str = "invert", conversion: int = 365, variables: List[str] = ["vv"]
    ):
        """
        Set the data as a pandas DataFrame (using methods from the dataframe_data object).

        :param dataf_ilf: [pd.DataFrame] --- Data
        :param type_data: [str] [default is "invert"] --- Type of the data (raw data, results of TICO, TICOI...)
        :param conversion: [int] [default is 365] --- Conversion factor: 365 is the unit of the velocity is m/y and 1 if it is m/d
        :param variables: [List<str>] [default is ['vv']] --- List of variable to plot
        """

        if type_data == "invert":
            self.datainvert = DataframeData(dataf_ilf)
            self.datainvert.dataf = self.datainvert.dataf.rename(columns={"error_x": "errorx", "error_y": "errory"})
            datatemp = self.datainvert
        elif type_data == "interp":
            self.datainterp = DataframeData(dataf_ilf)
            datatemp = self.datainterp
        elif type_data == "obs":
            self.dataobs = DataframeData(dataf_ilf)
            datatemp = self.dataobs
        elif type_data == "obs_filt":
            self.dataobsfilt = DataframeData(dataf_ilf)
            datatemp = self.dataobsfilt
        else:
            raise ValueError(
                "Please enter 'invert' for inverted results, 'interp' for ineterpolated results, 'obs' for observation or 'obs_filt' for filtered observations"
            )

        datatemp.set_temporal_baseline_central_date_offset_bar()  # Set the temporal baseline,
        if type_data == "invert":
            datatemp.set_vx_vy_invert(type_data=type_data, conversion=conversion)  # Convert displacement in vx and vy
        elif type_data == "obs_filt":
            datatemp.set_vx_vy_my(type_data=type_data, conversion=conversion)
        if "vv" in variables:
            datatemp.set_vv()  # set velocity magnitude
        datatemp.set_minmax()  # set min and max, for figures plots

    def load(
        self,
        dataf: pd.DataFrame | List[pd.DataFrame],
        type_data: str = "obs",
        dataformat: str = "df",
        save: bool = False,
        show: bool = False,
        figsize: tuple[int, int] = (10, 6),
        unit: str = "m/y",
        path_save: str = "",
        variables: List[str] | None = ["vv", "vx", "vy"],
        A: np.ndarray | None = None,
    ):
        """
        Load the data from dataf and format it in a dataframe_data object using the set_data_from_pandas_df method, depending on the type of data (type_data).
        Initialize the object with general plotting parameters.

        :param dataf: [pd.DataFrame | List[pd.DataFrame]] --- observations orresults from the inversion
        :param type_data: [str] [default is 'obs'] --- of 'obs' dataf corresponds to obsevations, if 'invert', it corresponds to inverted velocity
        :param dataformat: [str] [default is 'df'] --- id 'df' dataf is a pd.DataFrame
        :param save: [bool] [default is False]  --- if True, save the figures
        :param show: [bool] [default is True]  --- if True, show the figures
        :param figsize: tuple[int, int]  --- size of the figure
        :param unit: [str]   --- unit wanted for plotting
        :param filt: [List[bool] | None] [default is None] --- Are dataf data filtered ? Put True if dataf data are displacemenst, None if all data are not filtered
        :param path_save:[str] --- path where to store the data
        :param variables: [List[str]] [default is ['vv']] --- list of variable to plot
        :param A: [np.array] --- design matrix
        """

        self.__init__(save=save, show=show, figsize=figsize, unit=unit, path_save=path_save, A=A)

        conversion = self.get_conversion()  # Conversion factor
        if isinstance(dataf, list) and len(dataf) > 1:
            assert isinstance(type_data, list) and (len(dataf) == len(type_data)), (
                "If 'dataf' is a list, 'type_data' must be a list of the same length"
            )

            for i in range(len(dataf)):
                if dataformat == "df":
                    self.set_data_from_pandas_df(
                        dataf[i], type_data=type_data[i], conversion=conversion, variables=variables
                    )
        elif (isinstance(dataf, list) and len(dataf) == 1) or isinstance(dataf, pd.DataFrame):
            assert (isinstance(type_data, list) and len(type_data) == 1) or isinstance(type_data, str), (
                "If 'dataf' is a dataframe or list of a single dataframe, 'type_data' must either be a list of a single string element, or a string"
            )

            if dataformat == "df":
                self.set_data_from_pandas_df(
                    dataf[0] if isinstance(dataf, list) else dataf,
                    type_data=type_data[0] if isinstance(type_data, list) else type_data,
                    conversion=conversion,
                    variables=variables,
                )
        else:
            raise ValueError(f"'dataf' must be a list or a pandas dataframe, not {type(dataf)}")

    def get_dataf_invert_or_obs_or_interp(self, type_data: str = "obs") -> (pd.DataFrame, str):  # type: ignore
        """
        Get dataframe either obs or invert

        :param type_data: [str] [default is 'obs'] --- If 'obs', dataf corresponds to obsevations. If 'invert', it corresponds to the inverted velocities

        :return [pd.DataFrame] --- Dataframe from obs, invert or interp
        :return [str] --- Label used in the legend of the figures
        """

        # Get data when there is only dataframe loaded
        if self.dataobs is None and self.datainterp is None and self.dataobsfilt is None:
            return self.datainvert, "Results from the inversion"
        elif self.datainvert is None and self.datainterp is None and self.dataobsfilt is None:
            return self.dataobs, "Observations"
        elif self.datainvert is None and self.dataobs is None and self.dataobsfilt is None:
            return self.datainterp, "Results from TICOI"
        elif self.datainvert is None and self.dataobs is None and self.datainter is None:
            return self.dataobsfilt, "Observations filtered"
        elif self.datainvert is None and self.dataobs is None and self.datainterp is None and self.dataobsfilt is None:
            raise ValueError("Please load at least one dataframe")
        else:  # else
            if type_data == "invert":
                return self.datainvert, "Results from the inversion"
            elif type_data == "obs":
                return self.dataobs, "Observations"
            elif type_data == "obs_filt":
                return self.dataobsfilt, "Observations filtered"
            else:
                return self.datainterp, "Results from TICOI"

    def get_conversion(self):
        """
        Get conversion factor

        :return: [int] --- conversion factor
        """

        conversion = 365 if self.unit == "m/y" else 1
        return conversion

    def get_direction(self, data: "PixelClass.DataframeData") -> (np.array, np.array):  # type: ignore
        """
        Get the direction of the provided data

        :param data: [ticoi.pixel_class.dataframe_data] --- Dataframe from obs, invert or interp

        :return directionm: [np.array] --- Directions of the data
        :return directionm_mean: [np.array] --- Averaged direction of the data
        """

        directionm = np.arctan2(data.dataf["vy"].astype("float32"), data.dataf["vx"].astype("float32"))
        directionm[directionm < 0] += 2 * np.pi
        directionm_mean = np.arctan2(np.mean(data.dataf["vy"]), np.mean(data.dataf["vx"]))
        if directionm_mean < 0:
            directionm_mean += 2 * np.pi

        # Convert to degrees
        directionm *= 360 / (2 * np.pi)
        directionm_mean *= 360 / (2 * np.pi)
        return directionm, directionm_mean

    def get_filtered_results(self, filt: str | None = None):
        """
        Filter TICOI results using a given filter.

        :param filt: [str | None] [default is None] --- Filter to be used ('highpass' for a highpass filtering removing the trend over several years, 'lowpass' to just respect Shannon criterium, or None to don't apply any filter)

        :return vv_filt: [np array] --- Filtered velocities (magnitude)
        :return vv_c: [np array] --- Centered velocities (magnitude)
        :return dates_c: [np array] --- Central dates of the data
        :return dates: [np array] --- For each data, the number of days between its central date and a reference (first date of the data)
        """

        # Get dates and velocities from TICOI results
        dates_c = (
            self.datainterp.dataf["date1"] + (self.datainterp.dataf["date2"] - self.datainterp.dataf["date1"]) // 2
        )  # Central dates
        dates = (
            dates_c - self.datainterp.dataf["date1"].min()
        ).dt.days.to_numpy()  # Number of days to the reference day (first day of acquisition at the point)

        vv = self.datainterp.dataf["vv"]  # Velocity magnitude
        vv_c = vv - np.mean(vv)  # Centered velocities

        Ts = dates[1] - dates[0]

        # Filter the results...
        if filt == "highpass":  # ...to remove low frequencies (general trend over several years)
            b, a = signal.butter(4, [1 / (1.5 * 365), 1 / (2.001 * Ts)], "bandpass", fs=1 / Ts, output="ba")
            vv_filt = signal.filtfilt(b, a, vv_c)
        elif filt == "lowpass":  # ...to ensure Shanon critrion
            sos = signal.butter(4, 1 / (2.001 * Ts), "lowpass", fs=1 / Ts, output="sos")
            vv_filt = signal.sosfilt(sos, vv_c)
        else:  # Don't filter
            vv_filt = vv_c

        return vv_filt, vv_c, dates_c, dates

    def get_TF(
        self,
        filtered_results: list = None,
        filt: str | None = None,
        verbose: bool = False,
    ):
        """
        Compute the Fourier Transform (TF) of the interpolated results after applying a Hanning window.

        :param filtered_results: [list | None] [default is None] --- Results of the filtering (get_filtered_results method) if previously processed. If None, it is processed here
        :param filt: [str | None] [default is None] --- Filter to be used ('highpass' for a highpass filtering removing the trend over several years, 'lowpass' to just respect Shannon criterium, or None to don't apply any filter)
        :param verbose: [bool] [default is False] --- If True, print the maximum and the amplitude of the TF

        :return vv_tf: [np array] --- TF of the interpolated velocities without windowing
        :return vv_win_tf: [np array] --- TF of the interpolated velocities after windowing
        :return freq: [np array] --- Frequencies of the TF
        :return N: [np array] --- Number of dates
        """

        if filtered_results is not None:
            vv_filt, vv_c, dates_c, dates = filtered_results
        else:
            vv_filt, vv_c, dates_c, dates = self.get_filtered_results(filt)
        vv_filt = np.array(vv_filt)

        N = len(dates)
        Ts = dates[1] - dates[0]

        # Hanning window
        window = signal.windows.hann(N)

        # TFD
        n = 64 * N
        vv_tf = fft.rfft(vv_filt, n=n)
        vv_win_tf = fft.rfft(vv_filt * window, n=n)
        freq = fft.rfftfreq(n, d=Ts)

        if verbose:
            f = freq[np.argmax(np.abs(vv_win_tf))]
            print(f"TF maximum for f = {round(f, 5)} day-1 (period of {round(1 / f, 2)} days)")
            print(
                f"Amplitude of the TF at this frequency : {round(2 / N * np.abs(vv_tf[np.argmax(np.abs(vv_win_tf))]), 2)} m/y"
            )

        return vv_tf, vv_win_tf, freq, N

    def get_best_matching_sinus(
        self,
        filt: str | None = None,
        impose_frequency: bool = True,
        raw_seasonality: bool = False,
        several_freq: int = 1,
        verbose: bool = False,
    ):
        """
        Match a sinus (with fixed frequency or not) or a composition of several sinus (fundamental and harmonics) to the resulting TICOI data (and raw data)
        to measure its amplitude, the position of its maximum, the RMSE with the original data...

        :param filt: [str | None] [default is None] --- Filter to be used ('highpass' for a highpass filtering removing the trend over several years, 'lowpass' to just respect Shannon criterium, or None to don't apply any filter)
        :param impose_frequency: [bool] [default is True] --- If True, impose the frequency to 1/365.25 days-1 (one year seasonality). If False, look for the best matching frequency too, using the Fourier Transform in the first place
        :param raw_seasonality: [bool] [default is False] --- Also look for the best matching sinus directly on the raw data
        :param several_freq: [int] [default is 1] --- Number of harmonics to be computed (combination of sinus at frequencies 1/365.25, 2/365.25, etc...). If 1, only compute the fundamental.
        :param verbose: [bool] [default is False] --- If True, print the amplitude, the position of the maximum and the RMSE between the best matching sinus and the original data (TICOI results and raw data), and the best matching frequency if impose_frequency is False

        :return sine_f: [function] --- The function used for the optimization (can be used like sine = sine_f(dates[0], *popt, freqs=several_freq))
        :return popt: [list] --- Parameters of the best matching sinus to TICOI results
        :return popt_raw: [list] --- Parameters of the best matching sinus to raw data
        :return [dates, dates_c, dates_raw]: [list] --- dates, dates_raw: number of days between the central dates and the first central dates, dates_c: central dates of the data
        :return vv_filt: [np array] --- Filtered velocities (magnitude)
        :return stats: [list] --- Statistics about the best matching sinus to TICOI results [first maximum (date), day of the year of the maximum, amplitude, RMSE]
        :return stats_raw: [list] --- Statistics about the best matching sinus to raw data
        """

        # sine_fconst if impose_frequency else sine_fvar, popt, popt_raw, [dates, dates_c, dates_raw], vv_filt, stats, stats_raw

        vv_filt, vv_c, dates_c, dates = self.get_filtered_results(filt=filt)

        N = len(dates)

        if impose_frequency:
            # Sinus function (can add harmonics)
            def sine_fconst(t, *args, freqs=1, f=1 / 365.25):
                sine = args[0] * np.sin(2 * np.pi * f * t + args[1])
                for freq in range(1, freqs):
                    sine += args[2 * freq] * np.sin(2 * np.pi * (freq + 1) * f * t + args[2 * freq + 1])
                return sine + args[-1]

            f = 1 / 365.25

            # Find the best matching sinus to TICOI results
            guess = np.concatenate(
                [np.concatenate([[np.max(vv_filt) - np.min(vv_filt), 0] for _ in range(several_freq)]), [0]]
            )
            popt, pcov = curve_fit(lambda t, *args: sine_fconst(t, *args, freqs=several_freq), dates, vv_filt, p0=guess)

            # Parameters
            sine = sine_fconst(dates, *popt, freqs=several_freq)
            sine_year = sine_fconst(np.linspace(1, 365, 365), *popt, freqs=several_freq)

            first_max_day = pd.Timedelta(np.argmax(sine_year), "D") + self.datainterp.dataf["date1"].min()
            max_day = first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1)
            max_value = np.max(sine_year) - popt[-1]
            RMSE = np.sqrt(mean_squared_error(sine, vv_filt))

            del sine_year

            if verbose:
                print(
                    f"Amplitude of the best matching sinus (with period 365.25 days) to TICOI results: {round(max_value, 2)} m/y"
                )
                print(f"Maximum at day {max_day.days}")
                print(f"RMSE : {round(RMSE, 2)} m/y")

            if raw_seasonality:
                # Find the best matching sinus to raw data
                dates_raw = (self.dataobs.dataf.index - self.datainterp.dataf["date1"].min()).days.to_numpy()
                raw_c = self.dataobs.dataf["vv"] - self.dataobs.dataf["vv"].mean()
                guess_raw = np.concatenate(
                    [np.concatenate([[np.max(raw_c) - np.min(raw_c), 0] for _ in range(several_freq)]), [0]]
                )
                popt_raw, pcov_raw = curve_fit(
                    lambda t, *args: sine_fconst(t, *args, freqs=several_freq), dates_raw, raw_c, p0=guess_raw
                )

                # Parameters
                sine_raw = sine_fconst(dates_raw, *popt_raw, freqs=several_freq)
                sine_year_raw = sine_fconst(np.linspace(1, 365, 365), *popt_raw, freqs=several_freq)

                first_max_day_raw = pd.Timedelta(np.argmax(sine_year_raw), "D") + self.datainterp.dataf["date1"].min()
                max_day_raw = first_max_day_raw - pd.Timestamp(year=first_max_day_raw.year, month=1, day=1)
                max_value_raw = np.max(sine_year_raw) - popt_raw[-1]
                RMSE_raw = np.sqrt(mean_squared_error(sine_raw, raw_c))

                stats_raw = [first_max_day_raw, max_day_raw, max_value_raw, RMSE_raw]
                del sine_year_raw

                if verbose:
                    print(
                        f"Amplitude of the best matching sinus (with period 365.25 days) to raw data: {round(max_value_raw, 2)} m/y"
                    )
                    print(f"Maximum at day {max_day_raw.days}")
                    print(f"RMSE : {round(RMSE_raw, 2)} m/y")

        else:
            vv_tf, vv_win_tf, freq, _ = self.get_TF(vv_filt, vv_c, dates_c, dates, filt=filt, verbose=False)

            # Sinus function
            def sine_fvar(t, A, f, phi, off, freqs=None):
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

            popt, pcov = curve_fit(sine_fvar, dates, vv_filt, p0=guess)
            A, f, phi, off = popt
            sine = sine_fvar(dates, A, f, phi, off)
            sine_year = sine_fvar(np.linspace(1, 365, 365), A, f, phi, off)

            first_max_day = pd.Timedelta(np.argmax(sine_year), "D") + self.datainterp.dataf["date1"].min()
            max_day = first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1)
            max_value = np.max(sine_year) - off
            RMSE = np.sqrt(mean_squared_error(mean_squared_error(sine, vv_filt)))

            del sine_year

            if verbose:
                print(f"Period of the best matching sinus : {round(1 / f, 2)} days")
                print(f"Amplitude : {round(max_value, 2)} m/y")
                print(f"Maximum at day {max_day.days}")
                print(f"RMSE : {round(RMSE, 2)} m/y")

        stats = [first_max_day, max_day, max_value, RMSE]
        if not (impose_frequency and raw_seasonality):
            popt_raw, dates_raw, stats_raw = None, None, None

        return (
            sine_fconst if impose_frequency else sine_fvar,
            popt,
            popt_raw,
            [dates, dates_c, dates_raw],
            vv_filt,
            stats,
            stats_raw,
        )

    # %%========================================================================= #
    #              PLOTS ABOUT RAW DATA / INTERPOLATION RESULTS                   #
    # =========================================================================%% #

    def plot_vx_vy(self, color: str = "orange", type_data: str = "invert", block_plot: bool = True):
        """
        Plot vx and vy in two plots of the same figure.

        :param color: [str] [default is 'orange'] --- Color used for the plot
        :param type_data: [str] [default is 'obs'] --- If 'obs' dataf corresponds to observations, if 'invert', it corresponds to inverted velocity
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.

        :return fig, ax: Axis and Figures of the plot
        """

        data, label = self.get_dataf_invert_or_obs_or_interp(type_data)

        # Display the vx components
        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        ax[0].set_ylim(data.vxymin, data.vxymax)
        ax[0].plot(data.dataf["date_cori"], data.dataf["vx"], linestyle="", marker="o", markersize=2, color=color)
        ax[0].errorbar(
            data.dataf["date_cori"],
            data.dataf["vx"],
            xerr=data.dataf["offset_bar"],
            color=color,
            alpha=0.2,
            fmt=",",
            zorder=1,
        )
        ax[0].set_ylabel(f"Vx [{self.unit}]", fontsize=14)

        # Display the vy components
        ax[1].set_ylim(data.vyymin, data.vyymax)
        ax[1].plot(
            data.dataf["date_cori"], data.dataf["vy"], linestyle="", marker="o", markersize=2, color=color, label=label
        )
        ax[1].errorbar(
            data.dataf["date_cori"],
            data.dataf["vy"],
            xerr=data.dataf["offset_bar"],
            color=color,
            alpha=0.2,
            fmt=",",
            zorder=1,
        )
        ax[1].set_ylabel(f"Vy [{self.unit}]", fontsize=14)
        ax[1].set_xlabel("Central dates", fontsize=14)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc="lower left", bbox_to_anchor=(0.02, -0.4), fontsize=14)

        fig.suptitle("X and Y components of raw data velocities", y=0.95, fontsize=16)

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig.savefig(f"{self.path_save}/vx_vy_{type_data}.png")

        return fig, ax

    def plot_vx_vy_overlaid(
        self,
        colors: List[str] = ["orange", "blue"],
        type_data: str = "invert",
        zoom_on_results: bool = False,
        block_plot: bool = True,
    ):
        """
        Plot vx and vy in two plots of the same figure where inverted/interpolated results overlay the observations (raw data).

        :param colors: [List[str]] [default is ['orange', 'blue']] --- List of the colors used for the plot (first : raw data, second : overlaying data)
        :param type_data: [str] [default is 'obs'] --- If 'obs' dataf corresponds to obsevations, if 'invert', it corresponds to inverted velocity
        :param zoom_on_results: [bool] [default is False] --- If True set the limits of the axis according to the results min and max
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.

        :return fig, ax: Axis and Figures of the plot
        """

        data, label = self.get_dataf_invert_or_obs_or_interp(type_data)

        show = copy.copy(self.show)
        save = copy.copy(self.save)
        self.show, self.save = False, False
        fig, ax = self.plot_vx_vy(color=colors[0], type_data="obs")

        self.show, self.save = show, save

        if zoom_on_results:
            ax[0].set_ylim(data.vxymin, data.vxymax)
        ax[0].plot(
            data.dataf["date_cori"], data.dataf["vx"], linestyle="", marker="o", markersize=2, color=colors[1]
        )  # Display the vx components
        ax[0].errorbar(
            data.dataf["date_cori"],
            data.dataf["vx"],
            xerr=data.dataf["offset_bar"],
            color=colors[1],
            alpha=0.5,
            fmt=",",
            zorder=1,
        )
        if zoom_on_results:
            ax[1].set_ylim(data.vyymin, data.vyymax)
        ax[1].plot(
            data.dataf["date_cori"],
            data.dataf["vy"],
            linestyle="",
            marker="o",
            markersize=2,
            color=colors[1],
            label=label,
        )  # Display the vy components
        ax[1].errorbar(
            data.dataf["date_cori"],
            data.dataf["vy"],
            xerr=data.dataf["offset_bar"],
            color="b",
            alpha=0.2,
            fmt=",",
            zorder=1,
        )
        ax[1].legend(loc="lower left", bbox_to_anchor=(0.0, -0.65), fontsize=14)
        fig.suptitle(
            f"X and Y components of {'interpolated' if type_data == 'interp' else 'inverted'} results, along with raw data",
            y=0.95,
            fontsize=16,
        )

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            if zoom_on_results:
                fig.savefig(f"{self.path_save}/vx_vy_overlaid_zoom_on_results_{type_data}.png")
            else:
                fig.savefig(f"{self.path_save}/vx_vy_overlaid_{type_data}.png")

        return fig, ax

    def plot_vv(
        self, color: str = "orange", type_data: str = "invert", block_plot: bool = True, vminmax: list | None = None
    ):
        """
        Plot the velocity magnitude.

        :param color: [str] [default is 'orange'] --- Color used for the plot
        :param type_data: [str] [default is 'invert'] --- If 'obs' dataf corresponds to obsevations, if 'invert', it corresponds to inverted velocity
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.
        :param vminmax: List[int] [default is None] --- Min and max values for the y-axis of the plots

        :return fig, ax: Axis and Figure of the plot
        """

        data, label = self.get_dataf_invert_or_obs_or_interp(type_data)

        fig, ax = plt.subplots(figsize=self.figsize)
        if vminmax is None:
            ax.set_ylim(data.vvymin, data.vvymax)
        else:
            ax.set_ylim(vminmax[0], vminmax[1])
        ax.set_ylabel(f"Velocity magnitude  [{self.unit}]", fontsize=14)
        ax.plot(
            data.dataf["date_cori"],
            data.dataf["vv"],
            linestyle="",
            zorder=1,
            marker="o",
            lw=0.7,
            markersize=2,
            color=color,
            label=label,
        )
        ax.errorbar(
            data.dataf["date_cori"],
            data.dataf["vv"],
            xerr=data.dataf["offset_bar"],
            color=color,
            alpha=0.2,
            fmt=",",
            zorder=1,
        )
        plt.subplots_adjust(bottom=0.2)
        ax.legend(loc="lower left", bbox_to_anchor=(0.02, -0.25), fontsize=14)
        ax.set_xlabel("Central dates", fontsize=14)

        if type_data == "obs":
            fig.suptitle("Magnitude of raw data velocities", y=0.95, fontsize=16)
        elif type_data == "invert":
            fig.suptitle("Magnitude of inverted velocities", y=0.95, fontsize=16)
        elif type_data == "interp":
            fig.suptitle("Magnitude of interpolated velocities", y=0.95, fontsize=16)

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig.savefig(f"{self.path_save}/vv_{type_data}.png")

        return fig, ax

    def plot_vv_overlaid(
        self,
        colors: List[str] = ["orange", "blue"],
        type_data: str = "invert",
        zoom_on_results: bool = False,
        block_plot: bool = True,
        vminmax: list | None = None,
    ):
        """
        Plot the velocity magnitude of inverted/interpolated results, overlaying the velocity magnitude of the observations (raw data).

        :param colors: [List[str]] [default is ['orange', 'blue']] --- List of the colors used for the plot (first : raw data, second : overlaying data)
        :param type_data: [str] [default is 'invert'] --- If 'obs' dataf corresponds to obsevations, if 'invert', it corresponds to inverted velocity
        :param zoom_on_results: [bool] [default is False] --- Set the limites of the axis according to the results min and max
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.
        :param vminmax: List[int] [default is None] --- Min and max values for the y-axis of the plots

        :return fig, ax: Axis and Figure of the plots
        """

        data, label = self.get_dataf_invert_or_obs_or_interp(type_data)

        show = copy.copy(self.show)
        save = copy.copy(self.save)
        self.show, self.save = False, False
        fig, ax = self.plot_vv(color=colors[0], type_data="obs", vminmax=vminmax)
        self.show, self.save = show, save

        if zoom_on_results:
            ax.set_ylim(data.vvymin, data.vvymax)
        ax.plot(
            data.dataf["date_cori"],
            data.dataf["vv"],
            linestyle="",
            zorder=1,
            marker="o",
            lw=0.7,
            markersize=2,
            color=colors[1],
            label="Results from the inversion",
        )
        ax.errorbar(
            data.dataf["date_cori"],
            data.dataf["vv"],
            xerr=data.dataf["offset_bar"],
            color=colors[1],
            alpha=0.2,
            fmt=",",
            zorder=1,
        )
        ax.legend(loc="lower left", bbox_to_anchor=(0, -0.3), fontsize=14)
        fig.suptitle(
            f"Magnitude of {'interpolated' if type_data == 'interp' else 'inverted'} results, along with raw data magnitude",
            y=0.95,
            fontsize=16,
        )

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            if zoom_on_results:
                fig.savefig(f"{self.path_save}/vv_overlaid_zoom_on_results_{type_data}.png")
            else:
                fig.savefig(f"{self.path_save}/vv_overlaid_{type_data}.png")

        return fig, ax

    def plot_vv_quality(self, cmap: str = "viridis", type_data: str = "obs", block_plot: bool = True):
        """
        Plot error on top of velocity vx and vy.

        :param cmap: [str] [default is 'viridis''] --- Color map used to mark the errors in the plots
        :param type_data: [str] [default is 'obs'] --- If 'obs' dataf corresponds to obsevations, if 'invert', it corresponds to inverted velocity
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting
        :param vminmax: List[int] [default is None] --- Min and max values for the y-axis of the plots

        :return fig, ax: Axis and Figure of the plots
        """

        assert "errorx" in self.dataobs.dataf.columns and "errory" in self.dataobs.dataf.columns, (
            "'errorx' and/or 'errory' values are missing in the data, impossible to plot the errors"
        )

        data, label = self.get_dataf_invert_or_obs_or_interp(type_data)

        qualityx = data.dataf["errorx"]
        qualityy = data.dataf["errory"]
        qualityv = np.sqrt(
            (qualityx / data.dataf["vx"] * qualityx) ** 2 + (qualityy / data.dataf["vy"] * qualityy) ** 2
        )

        fig, ax = plt.subplots(figsize=self.figsize)
        # First subplot
        ax.set_ylabel(f"Vx [{self.unit}]", fontsize=14)
        scat = ax.scatter(data.dataf["date_cori"], data.dataf["vv"], c=qualityv, s=5, cmap=cmap)
        cbar = fig.colorbar(scat, ax=ax, orientation="horizontal", pad=0.2)  # Increased pad for spacing
        cbar.set_label("Errors [m/y]", fontsize=14)
        # Adjustments
        plt.subplots_adjust(hspace=0.5, bottom=0.3)  # Increase hspace and bottom padding
        fig.suptitle("Error associated to the velocity data", y=0.98, fontsize=16)  # Adjusted title position
        # Use tight layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig.savefig(f"{self.path_save}/vxvy_quality_bas_{type_data}.png")

        return fig, ax

    def plot_vx_vy_quality(self, cmap: str = "viridis", type_data: str = "obs", block_plot: bool = True):
        """
        Plot error on top of velocity magnitude vv

        :param cmap: [str] [default is 'viridis''] --- Color map used to mark the errors in the plots
        :param type_data: [str] [default is 'obs'] --- If 'obs' dataf corresponds to obsevations, if 'invert', it corresponds to inverted velocity
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting

        :return fig, ax: Axis and Figure of the plots
        """

        assert "errorx" in self.dataobs.dataf.columns and "errory" in self.dataobs.dataf.columns, (
            "'errorx' and/or 'errory' values are missing in the data, impossible to plot the errors"
        )

        data, label = self.get_dataf_invert_or_obs_or_interp(type_data)

        qualityx = data.dataf["errorx"]
        qualityy = data.dataf["errory"]

        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        # First subplot
        ax[0].set_ylabel(f"Vx [{self.unit}]", fontsize=14)
        scat = ax[0].scatter(data.dataf["date_cori"], data.dataf["vx"], c=qualityx, s=5, cmap=cmap)
        cbar = fig.colorbar(scat, ax=ax[0], orientation="horizontal", pad=0.2)  # Increased pad for spacing
        cbar.set_label("Errors [m/y]", fontsize=14)

        # Second subplot
        ax[1].set_ylabel(f"Vy [{self.unit}]", fontsize=14)
        scat = ax[1].scatter(data.dataf["date_cori"], data.dataf["vy"], c=qualityy, s=5, cmap=cmap)
        cbar = fig.colorbar(scat, ax=ax[1], orientation="horizontal", pad=0.2)  # Increased pad for spacing
        cbar.set_label("Errors [m/y]", fontsize=14)

        # Adjustments
        plt.subplots_adjust(hspace=0.5, bottom=0.3)  # Increase hspace and bottom padding
        fig.suptitle("Error associated to the velocity data", y=0.98, fontsize=16)  # Adjusted title position

        # Use tight layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig.savefig(f"{self.path_save}/vxvy_quality_bas_{type_data}.png")

        return fig, ax

    def plot_direction(
        self, color: str = "orange", type_data: str = "obs", block_plot: bool = True, plot_mean: bool = True
    ):
        """
        Plot the direction of the velocities for each of the data at this point.

        :param color: [str] [default is 'orange'] --- Color used for the plot
        :param type_data: [str] [default is 'obs'] --- If 'obs' dataf corresponds to obsevations, if 'invert', it corresponds to inverted velocity
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.
        :param plot_mean:  [bool] [default is True] --- If True, plot the mean velocity direction
        :return fig, ax: Axis and Figure of the plot
        """

        data, label = self.get_dataf_invert_or_obs_or_interp(type_data)

        directionm, directionm_mean = self.get_direction(data)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(data.dataf["date_cori"], directionm, linestyle="", marker="o", markersize=2, color=color, label=label)
        if plot_mean:
            ax.hlines(
                directionm_mean,
                np.min(data.dataf["date_cori"]),
                np.max(data.dataf["date_cori"]),
                label=f"Mean direction of {label}",
            )
        ax.set_ylim(0, 360)
        ax.set_ylabel("Direction [°]")
        ax.set_xlabel("Central Dates")
        plt.subplots_adjust(bottom=0.25)
        ax.legend(loc="lower left", bbox_to_anchor=(0, -0.4), ncol=2, fontsize=14)
        fig.suptitle("Direction of the observations", y=0.95, fontsize=16)

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig.savefig(f"{self.path_save}/direction_{type_data}.png")

        return fig, ax

    def plot_direction_overlaid(
        self,
        colors: List[str] = ["orange", "blue"],
        type_data: str = "interp",
        block_plot: bool = True,
        plot_mean: bool = True,
    ):
        """
        Plot the velocity direction of inverted/interpolated results, overlaying the velocity direction of the observations (raw data).

        :param colors: [List[str]] [default is ['orange', 'blue']] --- List of the colors used for the plot (first : raw data, second : overlaying data)
        :param type_data: [str] [default is 'invert'] --- If 'obs' dataf corresponds to obsevations, if 'invert', it corresponds to inverted velocity
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.
                :param plot_mean:  [bool] [default is True] --- If True, plot the mean velocity direction

        :return fig, ax: Axis and Figure of the plot
        """

        data, label = self.get_dataf_invert_or_obs_or_interp(type_data)

        show = copy.copy(self.show)
        save = copy.copy(self.save)
        self.show, self.save = False, False
        fig, ax = self.plot_direction(color=colors[0], type_data="obs", plot_mean=plot_mean)
        self.show, self.save = show, save

        directionm, directionm_mean = self.get_direction(data)

        ax.plot(
            data.dataf["date_cori"], directionm, linestyle="", marker="o", markersize=2, color=colors[1], label=label
        )
        ax.set_ylim(0, 360)
        ax.set_ylabel("Direction [°]", fontsize=14)
        ax.set_xlabel("Central Dates", fontsize=14)
        ax.legend(loc="lower left", bbox_to_anchor=(0, -0.4), ncol=2, fontsize=14)
        fig.suptitle(
            f"Direction of the {'interpolated' if type_data == 'interp' else 'inverted'} results, along with raw data direction",
            y=0.95,
            fontsize=16,
        )

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig.savefig(f"{self.path_save}/direction_overlaid_{type_data}.png")

        return fig, ax

        return fig, ax

    def plot_quality_metrics(self, color: str = "orange"):
        """
        Plot quality metrics on top of velocity magnitude. It can be the number of observations used for each estimation, and/or the confidence intervals.
        :param color: [str] [default is 'orange'] --- Color used for the plot
        :return:
        """

        dataf, label = self.get_dataf_invert_or_obs_or_interp(type_data="interp")
        data = dataf.dataf.dropna(subset=["vx", "vy"])  # drop rows where with no velocity values

        assert "error_x" and "x_count" not in data.columns, (
            "No quality metrics to display, please re run ticoi using the options Error_propagation or X_contribution"
        )

        if "error_x" in data.columns:
            data["error_x"] = np.sqrt(data["error_x"])
            data["error_y"] = np.sqrt(data["error_y"])
            data["error_v"] = np.sqrt(
                (data["vx"] / data["vv"] * data["error_x"]) ** 2 + (data["vy"] / data["vv"] * data["error_y"]) ** 2
            )

            data["confidence_x"] = data["sigma0"].iloc[2] * data["error_x"]
            data["confidence_y"] = data["sigma0"].iloc[3] * data["error_y"]
            data["confidence_v"] = np.nanmean(data["sigma0"].iloc[2:4]) * data["error_v"]

        if "xcount_x" in data.columns:
            xcount_mean = np.nanmean([data["xcount_x"], data["xcount_y"]], axis=0)  # Mean of xcount_x and xcount_y
            max_xcount = int(np.max(xcount_mean))
            if max_xcount > 100:
                bounds = [0, 100, 1000, max_xcount]
                cmap = mcolors.ListedColormap(["lightcoral", "red", "darkred"])  # Light red, red, dark red
                # Boundaries for color ranges
            else:
                bounds = [0, 100, max_xcount]
                cmap = mcolors.ListedColormap(["lightcoral", "red"])  # Light red, red, dark red

            norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Apply the custom colormap to the scatter plot based on xcount

        fig, ax = plt.subplots(figsize=(10, 6))
        if "error_x" in data.columns:
            if "xcount_x" not in data.columns:
                ax.plot(
                    data["date_cori"],
                    data["vv"],
                    linestyle="",
                    zorder=1,
                    marker="o",
                    lw=0.7,
                    markersize=2,
                    color=color,
                    label=label,
                )
            # Plot confidence interval using fill_between
            ax.fill_between(
                data["date_cori"],
                data["vv"] - data["confidence_v"],
                data["vv"] + data["confidence_v"],
                color="purple",
                alpha=0.4,
            )
            # Create custom legend entries for confidence interval
            conf_legend = malines.Line2D([], [], color="purple", alpha=0.4, lw=6, label="95% confidence interval")
            if "xcount_x" in data.columns:
                plt.subplots_adjust(bottom=-0.01)
            # Add the legends for confidence interval and GPS
            ax.legend(
                [conf_legend],
                ["95% confidence interval"],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fontsize=15,
                ncol=3,
                markerscale=1.5,
            )

        if "xcount_x" in data.columns:
            scat = ax.scatter(data["date_cori"], data["vv"], c=xcount_mean, cmap=cmap, norm=norm, s=7)
            # Add the colorbar for xcount
            cbar = fig.colorbar(scat, ax=ax, boundaries=bounds, orientation="horizontal", pad=0.15, shrink=0.7)
            cbar.set_label("Number of image-pair velocities used", fontsize=14)

        ax.set_ylabel("Velocity magnitude [m/y]", fontsize=18)
        # Show plot if specified
        if self.show:
            plt.show(block=False)

        # Save the figure
        if self.save:
            fig.savefig(f"{self.path_save}/confidence_intervals_and_quality.png")

        return fig, ax

    # %%========================================================================= #
    #                       PLOTS ABOUT INVERSION RESULTS                         #
    # =========================================================================%% #

    def plot_xcount_vx_vy(self, cmap: str = "viridis", block_plot: bool = True):
        """
        Plot the observation contribution to the inversion on top of velocities x and y components.

        :param cmap: [str] [default is 'rainbow] --- Color map used to mark the xcount values in the plots.
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.

        :return fig, ax: Axis and Figure of the plot
        """

        assert self.datainvert is not None, (
            "No inverted data found, think of loading the results of an inversion to this pixel_class before calling plot_xcount_vx_vy()"
        )
        assert "xcount_x" in self.datainvert.dataf.columns and "xcount_y" in self.datainvert.dataf.columns, (
            "'xcount_x' and/or 'xount_y' values are missing in the data, impossible to plot the xcount values"
        )

        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        ax[0].set_ylabel(f"Vx [{self.unit}]", fontsize=14)
        ax[0].scatter(
            self.datainvert.dataf["date_cori"],
            self.datainvert.dataf["vx"],
            c=self.datainvert.dataf["xcount_x"],
            s=8,
            cmap=cmap,
            label="Y_contribution",
        )
        ax[1].set_ylabel(f"Vy [{self.unit}]", fontsize=14)
        ax[1].set_xlabel("Central dates", fontsize=14)
        scat = ax[1].scatter(
            self.datainvert.dataf["date_cori"],
            self.datainvert.dataf["vy"],
            c=self.datainvert.dataf["xcount_y"],
            s=8,
            cmap=cmap,
        )
        plt.subplots_adjust(bottom=0.1)
        cbar = fig.colorbar(scat, ax=ax.ravel().tolist(), orientation="horizontal", pad=0.15)
        cbar.set_label("Amount of contributing observations", fontsize=14)
        fig.suptitle(
            "Contribution of the observations to the resulting inverted velocity x and y components",
            y=0.95,
            fontsize=16,
        )

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig.savefig(f"{self.path_save}/X_dates_contribution_vx_vy.png")

        return fig, ax

    def plot_xcount_vv(self, cmap: str = "viridis", block_plot: bool = True):
        """
        Plot the observation contribution to the inversion on top of the velocity magnitude.

        :param cmap: [str] [default is 'rainbow''] --- Color map used in the plots
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.

        :return fig, ax: Axis and Figure of the plot
        """

        assert self.datainvert is not None, (
            "No inverted data found, think of loading the results of an inversion to this pixel_class before calling plot_xcount_vv()"
        )
        assert "xcount_x" in self.datainvert.dataf.columns and "xcount_y" in self.datainvert.dataf.columns, (
            "'xcount_x' and/or 'xount_y' values are missing in the data, impossible to plot the xcount values"
        )

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_ylabel(f"Velocity magnitude [{self.unit}]", fontsize=14)
        ax.set_xlabel("Central dates", fontsize=14)
        scat = ax.scatter(
            self.datainvert.dataf["date_cori"],
            self.datainvert.dataf["vv"],
            c=(self.datainvert.dataf["xcount_x"] + self.datainvert.dataf["xcount_y"]) / 2,
            s=8,
            cmap=cmap,
        )
        # Adding a colorbar for the scatter plot
        cbar = plt.colorbar(scat, ax=ax, pad=0.02)
        cbar.set_label("Amount of contributing observations", fontsize=14)
        plt.subplots_adjust(bottom=0.2)
        fig.suptitle("Contribution of the observations to the resulting inverted velocities", y=0.95, fontsize=16)

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig.savefig(f"{self.path_save}/X_dates_contribution_vv.png")

        return fig, ax

    def plot_weights_inversion(self, cmap: str = "plasma_r", block_plot: bool = True):
        """
        Plot initial and final weights used in the inversion.

        :param cmap: [str] [default is 'plasma_r'] --- Color map used in the plots
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.

        :return ax_f, fig_f, ax_l, fig_l: Axis and Figure of the plots (weights from f: the first inversion, l: the last inversion)
        """

        assert self.datainvert is not None, (
            "No inverted data found, think of loading the results of an inversion to this pixel_class before calling plot_xcount_vv()"
        )

        ## ----------------------- Weights used during the first inversion ------------------------- ##
        fig_f, ax_f = plt.subplots(2, 1, figsize=(8, 4))
        ax_f[0].set_ylabel(f"Vx [{self.unit}]", fontsize=14)
        ax_f[0].set_xticklabels([])
        scat1 = ax_f[0].scatter(
            self.dataobs.dataf["date_cori"],
            self.dataobs.dataf["vx"],
            c=abs(self.dataobs.dataf["weightinix"]),
            s=5,
            cmap=cmap,
            edgecolors="k",
            linewidth=0.1,
        )
        ax_f[1].set_ylabel(f"Vy [{self.unit}]", fontsize=14)
        ax_f[1].set_xlabel("Central dates", fontsize=14)
        scat2 = ax_f[1].scatter(
            self.dataobs.dataf["date_cori"],
            self.dataobs.dataf["vx"],
            c=abs(self.dataobs.dataf["weightiniy"]),
            s=5,
            cmap=cmap,
            edgecolors="k",
            linewidth=0.1,
        )
        plt.subplots_adjust(bottom=0.32)
        legend1 = ax_f[1].legend(
            *scat1.legend_elements(num=5),
            loc="lower left",
            bbox_to_anchor=(0.05, -1.25),
            ncol=3,
            title="Initial weights for Vx",
        )
        legend2 = ax_f[1].legend(
            *scat2.legend_elements(num=5),
            loc="lower right",
            bbox_to_anchor=(0.95, -1.25),
            ncol=3,
            title="Initial weights for Vy",
        )
        ax_f[1].add_artist(legend1)
        ax_f[1].add_artist(legend2)
        fig_f.suptitle("Initial weights before the inversion", y=0.95, fontsize=16)

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig_f.savefig(f"{self.path_save}/weightini_vx_vy.png")

        ## ------------------------ Weights used during the last inversion ------------------------- ##
        fig_l, ax_l = plt.subplots(2, 1, figsize=(8, 4))
        ax_l[0].set_ylabel(f"Vx [{self.unit}]", fontsize=14)
        ax_l[0].set_xticklabels([])
        scat1 = ax_l[0].scatter(
            self.dataobs.dataf["date_cori"],
            self.dataobs.dataf["vx"],
            c=abs(self.dataobs.dataf["weightlastx"]),
            s=5,
            cmap=cmap,
            edgecolors="k",
            linewidth=0.1,
        )
        ax_l[1].set_ylabel(f"Vy [{self.unit}]", fontsize=14)
        ax_l[1].set_xlabel("Central dates", fontsize=14)
        scat2 = ax_l[1].scatter(
            self.dataobs.dataf["date_cori"],
            self.dataobs.dataf["vx"],
            c=abs(self.dataobs.dataf["weightlasty"]),
            s=5,
            cmap=cmap,
            edgecolors="k",
            linewidth=0.1,
        )
        plt.subplots_adjust(bottom=0.32)
        legend1 = ax_l[1].legend(
            *scat1.legend_elements(num=5),
            loc="lower left",
            bbox_to_anchor=(0.05, -1.25),
            ncol=3,
            title="Final weights for Vx",
        )
        legend2 = ax_l[1].legend(
            *scat2.legend_elements(num=5),
            loc="lower right",
            bbox_to_anchor=(0.95, -1.25),
            ncol=3,
            title="Final weights for Vy",
        )
        ax_l[1].add_artist(legend1)
        ax_l[1].add_artist(legend2)
        fig_l.suptitle("Final weights after the inversion", y=0.95, fontsize=16)

        if self.show:
            plt.show(block=block_plot)
        if self.save:
            fig_l.savefig(f"{self.path_save}/weightlast_vx_vy.png")

        return ax_f, fig_f, ax_l, fig_l

    def plot_residuals(self, log_scale: bool = False, block_plot: bool = True):
        """
        Statistics about the residuals from the inversion:
            - Plot of the final residuals overlaid in colors on vx and vy measurements ('residuals_vx_vy_final_residual.png').
            - Plot of the reconstructed velocity observations (from AX) overlaid on the original velocity observations ('residuals_vx_vy_mismatch.png').
            - Comparison of residuals according to the temporal baseline (residuals_tempbaseline.png),
            - the type of sensor and authors (residuals_author_abs.png,residuals_vy_author.png,residuals_vx_author_abs.png),
            - and the quality indicators (residuals_quality.png).

        :param log_scale: [bool] [default is False] --- if True, plot the figure in a log scale
        :param block_plot: [bool] [default is True] --- If True, the plot persists on the screen until the user manually closes it. If False, it disappears instantly after plotting.
        """

        assert self.datainvert is not None, (
            "No inverted data found, think of loading the results of an inversion to this pixel_class before calling plot_xcount_vv()"
        )
        assert self.A is not None, "Please provide A (design matrix) when loading the pixel_class"

        dataf = self.dataobs.dataf.replace("L. Charrier, J. Mouginot, R.Millan, A.Derkacheva", "IGE")
        dataf = dataf.replace("S. Leinss, L. Charrier", "Leinss")

        dataf["abs_residux"] = abs(dataf["residux"])
        dataf["abs_residuy"] = abs(dataf["residuy"])

        dataf = dataf.rename(columns={"author": "Author"})

        conversion = self.get_conversion()

        ###RECONSTRUCTION PLOT : reconstruct the observation from AX
        Y_reconstruct_x = (
            np.dot(self.A, self.datainvert.dataf["vx"] * self.datainvert.dataf["temporal_baseline"] / conversion)
            / self.dataobs.dataf["temporal_baseline"]
            * conversion
        )
        Y_reconstruct_y = (
            np.dot(self.A, self.datainvert.dataf["vy"] * self.datainvert.dataf["temporal_baseline"] / conversion)
            / self.dataobs.dataf["temporal_baseline"]
            * conversion
        )

        show = copy.copy(self.show)
        save = copy.copy(self.save)
        self.show, self.save = False, False
        fig, ax = self.plot_vx_vy(type_data="obs")
        self.show, self.save = show, save

        # fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].plot(
            self.dataobs.dataf["date_cori"],
            Y_reconstruct_x,
            linestyle="",
            marker="o",
            color="r",
            markersize=3,
            alpha=0.2,
        )  # Display the vx components
        ax[0].errorbar(
            self.dataobs.dataf["date_cori"],
            Y_reconstruct_x,
            xerr=self.dataobs.dataf["offset_bar"],
            color="r",
            alpha=0.2,
            fmt=",",
            zorder=1,
        )
        ax[0].set_ylabel(f"Vx [{self.unit}]", fontsize=18)
        ax[1].plot(
            self.dataobs.dataf["date_cori"],
            Y_reconstruct_y,
            linestyle="",
            marker="o",
            color="r",
            markersize=3,
            alpha=0.2,
            label="Reconstructed Data",
        )  # Display the vy components
        ax[1].errorbar(
            self.dataobs.dataf["date_cori"],
            Y_reconstruct_y,
            xerr=self.dataobs.dataf["offset_bar"],
            color="r",
            alpha=0.3,
            fmt=",",
            zorder=1,
        )
        ax[1].legend(bbox_to_anchor=(0.55, -0.3), ncol=3, fontsize=15)
        if self.show:
            plt.show()
        if self.save:
            fig.savefig(f"{self.path_save}/residuals_vx_vy_mismatch.png")

        ###RESIDUALS FROM THE LAST INVERSION
        fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].set_ylabel(f"Vx [{self.unit}]")
        scat1 = ax[0].scatter(
            self.dataobs.dataf["date_cori"],
            self.dataobs.dataf["vx"],
            c=abs(self.dataobs.dataf["residux"]),
            s=5,
            cmap="plasma_r",
            edgecolors="k",
            linewidth=0.1,
        )
        ax[1].set_ylabel(f"Vy [{self.unit}]")
        scat2 = ax[1].scatter(
            self.dataobs.dataf["date_cori"],
            self.dataobs.dataf["vy"],
            c=abs(self.dataobs.dataf["residuy"]),
            s=5,
            cmap="plasma_r",
            edgecolors="k",
            linewidth=0.1,
        )
        plt.subplots_adjust(bottom=0.3)
        legend1 = ax[1].legend(
            *scat1.legend_elements(num=5),
            loc="lower left",
            bbox_to_anchor=(0.05, 0),
            bbox_transform=fig.transFigure,
            ncol=3,
            title="Absolute residual Vx",
        )
        legend2 = ax[1].legend(
            *scat2.legend_elements(num=5),
            loc="lower right",
            bbox_to_anchor=(0.95, 0),
            bbox_transform=fig.transFigure,
            ncol=3,
            title="Absolute residual Vy",
        )
        ax[1].add_artist(legend1)
        ax[1].add_artist(legend2)
        if self.show:
            plt.show(block=False)
        if self.save:
            fig.savefig(f"{self.path_save}/residuals_vx_vy_final_residual.png")

        ###RESIDUALS FOR VX AND VY, ACCORDING TO THE SENSOR
        ax = sns.catplot(data=dataf, x="sensor", y="abs_residux", hue="Author", kind="box")
        ax.set(xlabel="Sensor", ylabel="Absolute residual vx [m/y]")
        if self.save:
            plt.savefig(f"{self.path_save}/residuals_vx_author_abs.png")
        if self.show:
            plt.show()

        ax = sns.catplot(data=dataf, x="sensor", y="abs_residuy", hue="Author", kind="box")
        ax.set(xlabel="Sensor", ylabel="Absolute residual vy [m/y]")
        if self.save:
            plt.savefig(f"{self.path_save}/residuals_author_abs.png")
        if self.show:
            plt.show()

        ###RESIDUALS FROM VX AND VY, ACCORDING TO THE AUTHOR
        ax = sns.catplot(data=dataf, x="sensor", y="residux", hue="Author", kind="box")
        ax.set(xlabel="Sensor", ylabel="Residual vx [m/y]")
        if self.save:
            plt.savefig(f"{self.path_save}/residuals_vx_author.png")
        if self.show:
            plt.show()

        ax = sns.catplot(data=dataf, x="sensor", y="residuy", hue="Author", kind="box")
        ax.set(xlabel="Sensor", ylabel="Residual vy [m/y]")
        if self.save:
            plt.savefig(f"{self.path_save}/residuals_vy_author.png")
        if self.show:
            plt.show()

        ###RESIDUALS FROM VX AND VY, ACCORDING TO THE QUALITY INDICATOR
        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        color_list = ["b", "m", "k", "g", "m"]
        for i, auth in enumerate(dataf["Author"].unique()):
            ax[0].plot(
                dataf[dataf["Author"] == auth]["weightinix"],
                dataf[dataf["Author"] == auth]["residux"],
                linestyle="",
                marker="o",
                color=color_list[i],
                markersize=3,
            )
            ax[1].plot(
                dataf[dataf["Author"] == auth]["weightiniy"],
                dataf[dataf["Author"] == auth]["residuy"],
                linestyle="",
                marker="o",
                color=color_list[i],
                markersize=3,
                label=auth,
            )
        if log_scale:
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
        ax[0].set_ylabel(f"Residual vx [{self.unit}]", fontsize=16)
        ax[1].set_ylabel(f"Residual vy [{self.unit}]", fontsize=16)
        ax[1].set_xlabel("Quality indicator", fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc="lower left", bbox_to_anchor=(0.12, 0), bbox_transform=fig.transFigure, fontsize=12, ncol=5)
        if self.show:
            plt.show()
        if self.save:
            if log_scale:
                fig.savefig(f"{self.path_save}/residu_qualitylog.png")
            else:
                fig.savefig(f"{self.path_save}/residuals_quality.png")

        ###RESIDUALS FROM VX AND VY, ACCORDING TO THE TEMPORAL BASELINE
        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        color_list = ["b", "m", "k", "g", "m"]
        for i, auth in enumerate(dataf["Author"].unique()):
            ax[0].plot(
                np.array(dataf["temporal_baseline"])[dataf["Author"] == auth] * 2,
                dataf[dataf["Author"] == auth]["residux"],
                linestyle="",
                marker="o",
                color=color_list[i],
                markersize=3,
            )
            ax[1].plot(
                np.array(dataf["temporal_baseline"])[dataf["Author"] == auth] * 2,
                dataf[dataf["Author"] == auth]["residuy"],
                linestyle="",
                marker="o",
                color=color_list[i],
                markersize=3,
                label=auth,
            )
        if log_scale:
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
        ax[0].set_ylabel(f"Residual vx [{self.unit}]", fontsize=16)
        ax[1].set_ylabel(f"Residual vy [{self.unit}]", fontsize=16)
        ax[1].set_xlabel("Temporal baseline [days]", fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc="lower left", bbox_to_anchor=(0.12, 0), bbox_transform=fig.transFigure, fontsize=12, ncol=5)
        if self.show:
            plt.show()
        if self.save:
            if log_scale:
                fig.savefig(f"{self.path_save}/residu_tempbaseline_log.png")
            else:
                fig.savefig(f"{self.path_save}/residuals_tempbaseline.png")

    # %%========================================================================= #
    #                         PLOTS ABOUT THE SEASONALITY                         #
    # =========================================================================%% #

    def plot_filtered_results(self, filt: str | None = None, impose_frequency: bool = True):
        """
        Plot the filtered TICOI results, with a given filter.

        :param filt: [str | None] [default is None] --- Filter to be used ('highpass' for a highpass filtering removing the trend over several years, 'lowpass' to just respect Shannon criterium, or None to don't apply any filter)
        :param impose_frequency: [bool] [default is True] --- If True, impose the frequency to 1/365.25 days-1 (one year seasonality). If False, look for the best matching frequency too, using the Fourier Transform in the first place

        :return fig, ax: Axis and Figure of the plot
        """

        vv_filt, vv_c, dates_c, dates = self.get_filtered_results(filt=filt)

        if impose_frequency:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
            axe = ax
        else:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=self.figsize)
            axe = ax[0]

        axe.plot(dates_c, vv_c, "blue", label="Before filtering")
        axe.plot(dates_c, vv_filt, "red", label="After filtering")
        axe.set_xlabel("Centered velocity [m/y]", fontsize=16)
        axe.set_ylabel("Central date", fontsize=16)
        axe.set_title("Effect of filtering", fontsize=16)
        axe.legend(loc="lower left")

        if not impose_frequency:
            ax[1].plot(dates_c, vv_filt * signal.windows.hann(len(dates)), "blue", label="With Hanning windowing")
            ax[1].plot(dates_c, vv_filt, "black", label="Without windowing")
            ax[1].set_xlabel("Centered velocity [m/y]", fontsize=16)
            ax[1].set_ylabel("Central date", fontsize=16)
            ax[1].set_title("Effect of Hanning windowing", fontsize=16)
            ax[1].legend(loc="best")

            fig.tight_layout()

        if self.show:
            plt.show()
        if self.save:
            fig.savefig(f"{self.path_save}/filtered_results.png")

        return fig, ax

    def plot_TF(self, filt=None, verbose=False):
        """
        Plot the Fourier Transform (TF) of the TICOI results after filtering with a given filter.

        :param filt: [str | None] [default is None] --- Filter to be used ('highpass' for a highpass filtering removing the trend over several years, 'lowpass' to just respect Shannon criterium, or None to don't apply any filter)
        :param verbose:[bool] [default is False] --- If True, print the maximum and the amplitude of the TF

        :return fig, ax: Axis and Figure of the plot
        """

        vv_tf, vv_win_tf, freq, N = self.get_TF(filt=filt, verbose=verbose)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        ax.plot(freq, 2 / N * np.abs(vv_tf), "blue", label="TF without windowing")
        ax.plot(freq, 2 / N * np.abs(vv_win_tf), "red", label="TF after Hanning windowing")
        ax.vlines(
            [i / 365 for i in range(1, 4)],
            0,
            1.1 * 2 / N * max(np.max(np.abs(vv_tf)), np.max(np.abs(vv_win_tf))),
            color="black",
            label="365d periodicity",
        )
        ax.set_xlim([0, 0.01])
        ax.set_ylim([0, 1.1 * 2 / N * max(np.max(np.abs(vv_tf)), np.max(np.abs(vv_win_tf)))])
        ax.set_xlabel("Frequency [day-1]", fontsize=16)
        ax.set_ylabel("Amplitude [m/y]", fontsize=16)
        ax.legend(loc="best")
        ax.set_title("Fourier Transform of the TICOI-resulting velocities", fontsize=16)

        if self.show:
            plt.show()
        if self.save:
            fig.savefig(f"{self.path_save}TF.png")

        return fig, ax

    def plot_best_matching_sinus(
        self,
        filt: str | None = None,
        impose_frequency: bool = True,
        raw_seasonality: bool = False,
        several_freq: int = 1,
        verbose: bool = False,
    ):
        """
        Plot the best matching sinus to the TICOI results (and to the raw data if required), by fixing the frequency to 1/365.25 days-1 or looking for the best matching one.

        :param filt: [str | None] [default is None] --- Filter to be used ('highpass' for a highpass filtering removing the trend over several years, 'lowpass' to just respect Shannon criterium, or None to don't apply any filter)
        :param impose_frequency: [bool] [default is True] --- If True, impose the frequency to 1/365.25 days-1 (one year seasonality). If False, look for the best matching frequency too, using the Fourier Transform in the first place
        :param raw_seasonality: [bool] [default is False] --- Also look for the best matching sinus directly on the raw data
        :param several_freq: [int] [default is 1] --- Number of harmonics to be computed (combination of sinus at frequencies 1/365.25, 2/365.25, etc...). If 1, only compute the fundamental
        :param verbose: [bool] [default is False] --- If True, print the amplitude, the position of the maximum and the RMSE between the best matching sinus and the original data (TICOI results and raw data), and the best matching frequency if impose_frequency is False

        :return fig, ax: Axis and Figure of the plots
        """

        sine_f, popt, popt_raw, dates, vv_filt, stats, stats_raw = self.get_best_matching_sinus(
            filt=filt,
            impose_frequency=impose_frequency,
            raw_seasonality=raw_seasonality,
            several_freq=several_freq,
            verbose=verbose,
        )

        sine = sine_f(dates[0], *popt, freqs=several_freq)
        f = popt[1] if not impose_frequency else 1 / 365.25

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.plot(
            self.dataobs.dataf.index,
            self.dataobs.dataf["vv"],
            linestyle="",
            marker="x",
            markersize=2,
            color="orange",
            label="Raw data",
        )
        ax.plot(dates[1], self.datainterp.dataf["vv"], "black", alpha=0.6, label="TICOI velocities")
        if filt is not None:
            ax.plot(
                dates[1],
                vv_filt + np.mean(self.datainterp.dataf["vv"]),
                "red",
                alpha=0.6,
                label="Filtered TICOI velocities",
            )
        if impose_frequency and raw_seasonality:
            sine_raw = sine_f(dates[2], *popt_raw, freqs=several_freq) if impose_frequency else sine_f(dates[2], *popt)
            ax.plot(
                self.dataobs.dataf.index,
                sine_raw + self.dataobs.dataf["vv"].mean(),
                linewidth=3,
                color="forestgreen",
                label="Best matching sinus to raw data",
            )
        ax.plot(
            dates[1],
            sine + np.mean(self.datainterp.dataf["vv"]),
            color="deepskyblue",
            linewidth=3,
            label="Best matching sinus to TICOI results",
        )
        ax.vlines(
            pd.date_range(start=stats[0], end=self.datainterp.dataf["date2"].max(), freq=f"{int(1 / f)}D"),
            np.min(self.datainterp.dataf["vv"]),
            np.max(self.datainterp.dataf["vv"]),
            "black",
            label="Maximum (TICOI)",
        )
        ax.set_xlabel("Central dates", fontsize=16)
        ax.set_ylabel("Velocity", fontsize=16)
        ax.legend(loc="best")
        ax.set_title("Best matching sinus around an annual seasonality")

        if self.show:
            plt.show()
        if self.save:
            fig.savefig(f"{self.path_save}matching_sine.png")

        return fig, ax

    def plot_annual_curves(
        self,
        normalize: bool = False,
        statistics: List[str] = [
            "min",
            "max",
            "mean",
            "median",
            "std",
            "amplitude",
            "max_day",
            "nb_peaks",
            "relative_max",
        ],
        cmap: str = "hsv",
        markers: List[str] = [".", "p", "s", "v", "D", "*", "x", "1", "+"],
        markers_size: List[int] = [5, 4, 3, 4, 3, 4, 4, 7, 4],
        verbose: bool = True,
    ):
        """
        Plot the velocity curves of each year on top of ones another and compute some statistics about it ().

        :param normalize: [bool] [default is False] --- Normalize the curves to [0-1] before plotting
        :param statistics: [List[str]] [default is everything] --- List of the statistics to compute and return (in ['min_max', 'mean', 'median', 'std', 'amplitude', 'max_day', 'nb_peaks', 'relative_max'])
        :param cmap: [str] [default is 'hsv'] --- Color map among which the colors for plotting the annual curves are picked
        :param markers: [List[str]] [default is ['.', 'p', 's', 'v', 'D', '*', 'x', '1', '+']] --- Symbols of the markers for the plot
        :param markers_size: [List[int]] [default is [5, 4, 3, 4, 3, 4, 4, 6, 4]] --- Marker size to use for each marker
        :param verbose: [bool] [default is False] --- Print a recap of the year statistics for each year

        :return fig, ax: Axis and Figure of the plots
        :return stats: [dict] --- dictionary of the statistics (each key is associated to a list with every year's value of the statistic related to the key)
        """

        dates_c = (
            self.datainterp.dataf["date1"] + (self.datainterp.dataf["date2"] - self.datainterp.dataf["date1"]) // 2
        )  # Central dates
        vv = np.sqrt(
            self.datainterp.dataf["vx"] ** 2 + self.datainterp.dataf["vy"] ** 2
        ).to_numpy()  # Velocity magnitude

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

        stats = {
            "min": [],
            "max": [],
            "mean": [],
            "median": [],
            "std": [],
            "amplitude": [],
            "max_day": [],
            "nb_peaks": [],
            "relative_max": [],
        }

        cmap = matplotlib.cm.get_cmap(cmap)
        colors = [cmap(i) for i in np.linspace(0, 1, len(years))]
        fig, ax = plt.subplots(figsize=(12, 4))
        for y in range(len(years)):
            dates = dates_c[[dates_c.iloc[i].year == years[y] for i in range(dates_c.size)]] - pd.Timestamp(
                year=years[y], month=1, day=1
            )
            dates = np.array([dates.iloc[i].days for i in range(dates.size)])
            vv_y = vv[[dates_c.iloc[i].year == years[y] for i in range(dates_c.size)]]

            if verbose:
                print(f"Year {years[y]} :")

            if "min" in statistics:  # Min value of the velocities over the year
                stats["min"].append(np.min(vv_y))
                if verbose:
                    print("   Min = {:.1f} m/y".format(stats["min"][y]))
            if "max" in statistics:  # Max value of the velocities over the year
                stats["max"].append(np.max(vv_y))
                if verbose:
                    print("   Max = {:.1f} m/y".format(stats["max"][y]))
            if "mean" in statistics:
                stats["mean"].append(np.mean(vv_y))  # Mean value of the velocities over the year
                if verbose:
                    print("   Mean = {:.1f} m/y".format(stats["mean"][y]))
            if "median" in statistics:
                stats["median"].append(np.median(vv_y))  # Median value of the velocities over the year
                if verbose:
                    print("   Median = {:.1f} m/y".format(stats["median"][y]))
            if "std" in statistics:
                stats["std"].append(np.std(vv_y, ddof=0))  # Standard deviation of the velocities over the year
                if verbose:
                    print("   Standard deviation = {:.1f} m/y".format(stats["std"][y]))
            if "amplitude" in statistics:
                stats["amplitude"].append(
                    (np.max(vv_y) - np.min(vv_y)) / 2
                )  # Amplitude of the velocity variations (computed as (max - min)/2)
                if verbose:
                    print("   Amplitude = {:.1f} m/y".format(stats["amplitude"][y]))
            if "max_day" in statistics:
                stats["max_day"].append(dates[np.argmax(vv_y)])  # Position of the maximum (in day)
                if verbose:
                    diff_month = stats["max_day"][y] - np.array(list(months_start.values()))
                    month = list(months_start.keys())[np.argmin(diff_month[diff_month > 0])]
                    day = np.min(diff_month[diff_month > 0]) + 1
                    print(f"   Day of the maximum = {stats['max_day'][y]}th day of the year ({month}, {day})")

            if "nb_peaks" in statistics or "relative_max" in statistics or "start_accel" in statistics:
                deriv = np.diff(vv_y) / np.diff(dates)  # Compute the derivative of the velocities
                peak_pos = (
                    [False]
                    + [(np.sign(deriv[i + 1]) == -1 and np.sign(deriv[i]) == 1) for i in range(len(deriv) - 1)]
                    + [False]
                )
                peak_dates = dates[peak_pos]
                peak_amplitudes = vv_y[peak_pos] - np.mean(vv_y)  # This time, the amplitudes are compute as max - mean

                if "nb_peaks" in statistics:
                    stats["nb_peaks"].append(len(peak_dates))  # Number of velocitiy peaks during the year
                    if verbose:
                        print("   Number of maximum = {}".format(stats["nb_peaks"][y]))
                if (
                    "relative_max" in statistics
                ):  # Amplitude of the second maximum divided by the amplitude of the first maximum
                    if len(peak_dates) == 0:
                        stats["relative_max"].append(None)
                    else:
                        stats["relative_max"].append(
                            np.max(peak_amplitudes[np.arange(len(peak_amplitudes)) != np.argmax(peak_amplitudes)])
                            / np.max(peak_amplitudes)
                        )
                        if verbose:
                            print("   Relative maximum value = {:.2f}".format(stats["relative_max"][y]))
                if "start_accel" in statistics:
                    pass

            if normalize:
                vv_y = (vv_y - np.min(vv_y)) / (np.max(vv_y) - np.min(vv_y))

            ax.plot(
                dates,
                vv_y,
                linestyle="",
                marker=markers[y],
                markersize=markers_size[y],
                label=str(years[y]),
                color=colors[y],
            )

        ax.set_xticks(list(months_start.values()), list(months_start.keys()))
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Day of the year", fontsize=14)
        ax.set_ylabel("Velocity magnitude [m/y]", fontsize=14)
        ax.legend(loc="best")
        ax.set_title("Superposed annual TICOI resulting velocities", fontsize=16)
        plt.subplots_adjust(bottom=0.2)

        if self.show:
            plt.show()
        if self.save:
            fig.savefig(f"{self.path_save}annual_curves.png")

        return fig, ax, {key: stats[key] for key in statistics}
