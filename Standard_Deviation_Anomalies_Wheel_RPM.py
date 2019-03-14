# Standard modules
import datetime
import progressbar
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas import datetime
from pandas import read_csv

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from wheel rpm data
    return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for wheel rpm data


def detect_anomalies_with_mean(ts, num_stds, verbose):
    """Detect outliers in the wheel RPM data by comparing points against [num_stds] standard deviations from the mean.

       Inputs:
           ts [pd Series]:   A pandas Series with a DatetimeIndex and a column for RPM.
           num_stds [float]: The number of standard deviations away from the mean used to define point outliers.
           verbose [bool]:   When True, a plot of the dataset mean will be displayed before outliers are detected.

       Optional Inputs:
           None

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, and columns for RPM (real values) and Outlier (True or False).
           outliers [pd Series]: The detected outliers, as a pandas Series with a DatetimeIndex and a column for the outlier value.

       Optional Outputs:
           None

       Example:
           wheel_rpm_with_outliers, outliers = detect_anomalies_with_mean(time_series, 2, True)
       """

    # Gather statistics in preparation for outlier detection
    mean = float(ts.values.mean())
    mean_line = pd.Series(([mean] * len(ts)), index=ts.index)
    std = float(ts.values.std(ddof=0))
    X = ts.values
    outliers = pd.Series()
    time_series_with_outliers = pd.DataFrame({'RPM': ts})
    time_series_with_outliers['Outlier'] = 'False'

    if verbose:
        pyplot.plot(ts, color='blue', label='Time Series')
        pyplot.plot(mean_line, color='black', label='Time Series Mean')
        pyplot.legend(loc='best')
        pyplot.title('Time Series & Mean')
        pyplot.show()

    # Start a progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    progress_bar_sliding_window = progressbar.ProgressBar(
        widgets=[progressbar.FormatLabel('Wheel RPM Outliers ')] + widgets,
        max_value=int(len(X))).start()

    # Label outliers using standard deviation
    for t in range(len(X)):
        obs = X[t]
        if abs(mean-obs) > std*num_stds:
            time_series_with_outliers.at[ts.index[t], 'Outlier'] = 'True'
            outlier = pd.Series(obs, index=[ts.index[t]])
            outliers = outliers.append(outlier)
        progress_bar_sliding_window.update(t)  # advance progress bar

    return time_series_with_outliers, outliers


def detect_anomalies_with_rolling_mean(ts, num_stds, window, verbose):
    """Detect outliers in the wheel RPM data by comparing points against [num_stds] standard deviations from a rolling mean.

       Inputs:
           ts [pd Series]:   A pandas Series with a DatetimeIndex and a column for RPM.
           num_stds [float]: The number of standard deviations away from the mean used to define point outliers.
           window [int]:     Window size; the number of samples to include in the rolling mean.
           verbose [bool]:   When True, a plot of the rolling mean will be displayed before outliers are detected.

       Optional Inputs:
           None

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, and columns for RPM (real values) and Outlier (True or False).
           outliers [pd Series]: The detected outliers, as a pandas Series with a DatetimeIndex and a column for the outlier value.

       Optional Outputs:
           None

       Example:
           wheel_rpm_with_outliers, outliers = detect_anomalies_with_rolling_mean(time_series, 2, window, False)
    """

    if window <= 0:
        raise ValueError('\'window\' must be given a value greater than 0 when using rolling mean.')
    else:
        # Gather statistics in preparation for outlier detection
        rolling_mean = ts.rolling(window=window, center=False).mean()
        first_window_mean = ts.iloc[:window].mean()
        for i in range(window):  # fill first 'window' samples with mean of those samples
            rolling_mean[i] = first_window_mean
        std = float(ts.values.std(ddof=0))
        X = ts.values
        outliers = pd.Series()
        time_series_with_outliers = pd.DataFrame({'RPM': ts})
        time_series_with_outliers['Outlier'] = 'False'

        if verbose:
            pyplot.plot(ts, color='blue', label='Time Series')
            pyplot.plot(rolling_mean, color='black', label='Rolling Mean')
            pyplot.legend(loc='best')
            pyplot.title('Time Series & Rolling Mean')
            pyplot.show()

        # Start a progress bar
        widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
        progress_bar_sliding_window = progressbar.ProgressBar(
            widgets=[progressbar.FormatLabel('Wheel RPM Outliers ')] + widgets,
            max_value=int(len(X))).start()

        # Label outliers using standard deviation
        for t in range(len(X)):
            obs = X[t]
            y = rolling_mean[t]
            if abs(y-obs) > std*num_stds:
                time_series_with_outliers.at[ts.index[t], 'Outlier'] = 'True'
                outlier = pd.Series(obs, index=[ts.index[t]])
                outliers = outliers.append(outlier)
            progress_bar_sliding_window.update(t)  # advance progress bar

        return time_series_with_outliers, outliers


def standard_deviation_anomalies_wheel_rpm(dataset_path='Data/WheelRPM.csv', plots_save_path=None,
                                            verbose=False, use_rolling_mean=False, window=0, num_stds=2):
    """Detect outliers in the wheel RPM data by one of the following methods:
       1) comparing points against [num_stds] standard deviations from the dataset mean
       2) comparing points against [num_stds] standard deviations from a rolling mean with a specified window

       Inputs:
           None

       Optional Inputs:
           dataset_path [str]:      A string path to the wheel RPM data. Data is read as a pandas Series with a DatetimeIndex and a column for RPM.
           plots_save_path [str]:   Set to a path in order to save the plot of the data with outliers to disk.
                                    Default is None, meaning no plots will be saved to disk.
           verbose [bool]:          Set to display extra dataset information (plot of the data, its head, and statistics).
                                    Default is False.
           use_rolling_mean [bool]: Set to compare points against a rolling mean. Requires setting a value for "window".
                                    Default is False.
           window [int]:            Window size; the number of samples to include in a rolling mean.
           num_stds [float]:        The number of standard deviations away from the mean used to define point outliers.

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, and columns for RPM (real values) and Outlier (True or False).

       Optional Outputs:
           None

       Example:
           wheel_rpm_with_outliers = standard_deviation_anomalies_wheel_rpm(verbose=True, plots_save_path='Plots/',
                                                                                use_rolling_mean=True, window=100)
       """

    # Load the dataset
    print('Reading the dataset: ' + dataset_path)
    time_series = read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

    # Preprocess
    # time_series.fillna(0, inplace=True)
    # time_series = time_series.resample('5min').mean()
    # time_series.to_csv('Wheel1RPMAveraged5Minutely.csv')

    if verbose:
        # describe the loaded dataset
        print(time_series.head())
        print(time_series.describe())
        time_series.plot(title=dataset_path + ' Dataset')  # plots the data
        pyplot.show()

    if use_rolling_mean:
        time_series_with_outliers, outliers = detect_anomalies_with_rolling_mean(time_series, num_stds, window, verbose)
    else:
        time_series_with_outliers, outliers = detect_anomalies_with_mean(time_series, num_stds, verbose)

    # Plot the outliers
    time_series.plot(color='blue', title=dataset_path.split('/')[-1] + ' Dataset with Outliers')
    pyplot.xlabel('Time')
    pyplot.ylabel('RPM')
    if len(outliers) > 0:
        print('\nDetected Outliers: ' + str(len(outliers)) + "\n")
        outliers.plot(color='red', style='.')
    if plots_save_path:
        current_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        filename = plots_save_path + dataset_path.split('/')[-1] + ' with Outliers (' + current_time + ').png'
        pyplot.savefig(filename, dpi=500)
    pyplot.show()

    return time_series_with_outliers


if __name__ == "__main__":
    print('Standard_Deviation_Anomalies_Wheel_RPM.py is being run directly')
    wheel_rpm_with_outliers = standard_deviation_anomalies_wheel_rpm(verbose=True, use_rolling_mean=True,
                                                                     window=1000, num_stds=2)

else:
    print('Standard_Deviation_Anomalies_Wheel_RPM.py is being imported into another module')
