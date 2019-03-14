# Standard modules
import datetime
from math import sqrt
import os
import progressbar
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas import datetime
from pandas import read_csv
from sklearn.metrics import mean_squared_error


__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data


def fit_rolling_mean(ts, window, verbose=False):
    if window <= 0:
        raise ValueError('\'window\' must be given a value greater than 0 when using rolling mean.')

    # Gather statistics in preparation for outlier detection
    rolling_mean = ts.rolling(window=window, center=False).mean()
    first_window_mean = ts.iloc[:window].mean()
    for i in range(window):  # fill first 'window' samples with mean of those samples
        rolling_mean[i] = first_window_mean

    if verbose:
        pyplot.plot(ts, color='blue', label='Time Series')
        pyplot.plot(rolling_mean, color='black', label='Rolling Mean')
        pyplot.legend(loc='best')
        pyplot.title('Time Series & Rolling Mean')
        pyplot.show()

    rmse = sqrt(mean_squared_error(ts, rolling_mean))
    return rmse


def generate_rolling_mean_rmses():
    """Fit all datasets in this study with a rolling mean, where the window size is the length of the dataset/100,
       and return the RMSEs for those fits.

       Inputs:
           None

       Optional Inputs:
           None

       Outputs:
           datasets_with_rmses [list(tuple)]: A list of tuples that are the datasets in this study paired with the RMSEs
                                              from the rolling mean fitting.

       Optional Outputs:
           None

       Example:
           datasets_with_rmses = generate_rolling_mean_rmses()
       """

    datasets = ['Data/BusVoltage.csv', 'Data/TotalBusCurrent.csv', 'Data/BatteryTemperature.csv',
               'Data/WheelTemperature.csv', 'Data/WheelRPM.csv']
    var_names = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Temperature (C)', 'RPM']
    rmses = [0, 0, 0, 0, 0]
    datasets_with_rmses = datasets

    for ds in range(len(datasets)):
        dataset = datasets[ds]
        var_name = var_names[ds]

        # Load the dataset
        print('Reading the dataset: ' + dataset)
        time_series = read_csv(dataset, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
        print('Length of dataset: ' + str(len(time_series)))
        window_size = len(time_series) / 100
        print('Length of window: ' + str(window_size) + '\n')

        rmse = fit_rolling_mean(time_series, window_size, verbose=True)
        rmses[ds] = rmse
        datasets_with_rmses[ds] = (datasets[ds], rmse)

    return datasets_with_rmses


if __name__ == "__main__":
    print('Generate_Rolling_Mean_RMSEs.py is being run directly\n')

    datasets_with_rmses = generate_rolling_mean_rmses()
    print(str(datasets_with_rmses))


else:
    print('Generate_Rolling_Mean_RMSEs.py is being imported into another module\n')
