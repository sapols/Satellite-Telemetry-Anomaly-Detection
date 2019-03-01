# Standard modules
import datetime
import progressbar
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas import datetime
from pandas import read_csv

# Custom modules
from detect_anomalies_with_mean import detect_anomalies_with_mean
from detect_anomalies_with_rolling_mean import detect_anomalies_with_rolling_mean

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data


def detect_standard_deviation_anomalies(dataset_path, var_name='Value', plots_save_path=None,
                                        verbose=False, use_rolling_mean=False, window=0, num_stds=2):
    """Detect outliers in the time series data by one of the following methods:
       1) comparing points against [num_stds] standard deviations from the dataset mean
       2) comparing points against [num_stds] standard deviations from a rolling mean with a specified window

       Inputs:
           dataset_path [str]: A string path to the time series data. Data is read as a pandas Series with a DatetimeIndex and a column for numerical values.

       Optional Inputs:
           var_name [str]:          The name of the dependent variable in the time series.
                                    Default is 'Value'.
           plots_save_path [str]:   Set to a path in order to save the plot of the data with outliers to disk.
                                    Default is None, meaning no plots will be saved to disk.
           verbose [bool]:          Set to display extra dataset information (plot of the data, its head, and statistics).
                                    Default is False.
           use_rolling_mean [bool]: Set to compare points against a rolling mean. Requires setting a value for "window".
                                    Default is False.
           window [int]:            Window size; the number of samples to include in a rolling mean.
           num_stds [float]:        The number of standard deviations away from the mean used to define point outliers.

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, a columns for numerical values, and an Outlier column (True or False).

       Optional Outputs:
           None

       Example:
           time_series_with_outliers = detect_standard_deviation_anomalies(dataset_path='Data/BatteryTemperature.csv',
                                                                           verbose=True, plots_save_path='Plots/',
                                                                           use_rolling_mean=True, window=100)
       """

    # Load the dataset
    print('Reading the dataset: ' + dataset_path)
    print('Using rolling mean? ' + str(use_rolling_mean))
    print('Standard deviations for point outliers: ' + str(num_stds))
    time_series = read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

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
    pyplot.ylabel(var_name)
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
    print('Detect_Standard_Deviation_Anomalies.py is being run directly\n')

    ds_num = 3  # used to select dataset path and variable name together

    dataset = ['Data/BusVoltage.csv', 'Data/TotalBusCurrent.csv', 'Data/BatteryTemperature.csv',
               'Data/WheelTemperature.csv', 'Data/WheelRPM.csv'][ds_num]
    name = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Temperature (C)', 'RPM'][ds_num]

    time_series_with_outliers = detect_standard_deviation_anomalies(dataset_path=dataset, var_name=name, verbose=True,
                                                                    use_rolling_mean=True, window=10000, num_stds=2)

else:
    print('Detect_Standard_Deviation_Anomalies.py is being imported into another module\n')
