# Standard modules
import datetime
import os
import pandas as pd
from pandas import datetime
from pandas import read_csv
import numpy as np
from matplotlib import pyplot

# Custom modules
from detect_anomalies_with_ARIMA import detect_anomalies_with_arima

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data


def detect_arima_anomalies(dataset_path, train_size, order, seasonal_order=(), seasonal_freq=None, trend=None,
                           grid_search=False, path_to_model=None, var_name='Value', plots_save_path=None,
                           verbose=False):
    """Detect outliers in the given time series by comparing points against an ARIMA forecast, then plot the outliers.

       Inputs:
           dataset_path [str]: A string path to the time series data. Data is read as a pandas Series with a DatetimeIndex and a column for numerical values.
           train_size [float]: The percentage of data to use for training, as a float (e.g., 0.66).
           order [tuple]:      The order hyperparameters (p,d,q) for the ARIMA model.

       Optional Inputs:
           seasonal_order [tuple]: The seasonal order hyperparameters (P,D,Q) for the ARIMA model. When specifying these, 'seasonal_freq' must also be given.
           seasonal_freq [int]:    The freq hyperparameter for this ARIMA model, i.e., the number of samples that make up one seasonal cycle.
           grid_search [bool]:     When True, perform a grid search to set values for the 'order' and 'seasonal order' hyperparameters.
                                   Note this overrides any given (p,d,q)(P,D,Q) hyperparameter values. Default is False.
           path_to_model [str]:    Path to a *.pkl file of a trained ARIMA model. When set, no training will be done because that model will be used.
           var_name [str]:         The name of the dependent variable in the time series.
                                   Default is 'Value'.
           verbose [bool]:         When True, describe the time series dataset upon loading it, and pass 'verbose=True' down the chain to any other functions called during outlier detection.
                                   Default is False.

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, a columns for numerical values, and an Outlier column (True or False).

       Optional Outputs:
           None

       Example:
           time_series_with_outliers = detect_arima_anomalies(dataset_path='Data/BusVoltage.csv', train_size=0.66,
                                                              order=(12, 0, 0), seasonal_order=(0, 1, 0),
                                                              seasonal_freq=2920, var_name='Voltage', verbose=True)
       """

    # Load the dataset
    print('Reading the dataset: ' + dataset_path)
    time_series = read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

    # TODO: don't use just this chunk of data; delete me
    split = int(len(time_series) * 0.15)
    time_series = time_series[0:split]

    if verbose:
        # describe the loaded dataset
        print(time_series.head())
        print(time_series.describe())
        time_series.plot(title=dataset_path + ' Dataset')
        pyplot.show()

    # Detect outliers
    time_series_with_outliers, outliers = detect_anomalies_with_arima(time_series, train_size=train_size, order=order,
                                                                      seasonal_order=seasonal_order,
                                                                      seasonal_freq=seasonal_freq, trend=trend,
                                                                      grid_search=grid_search,
                                                                      path_to_model=path_to_model, verbose=verbose,
                                                                      var_name=var_name)

    # Plot the outliers
    time_series.plot(color='blue', title=dataset_path.split('/')[-1] + ' Dataset with Outliers')
    pyplot.xlabel('Time')
    pyplot.ylabel(var_name)
    if len(outliers) > 0:
        print('\nDetected outliers: ' + str(len(outliers)) + '\n')
        outliers.plot(color='red', style='.')
    if plots_save_path:
        if not os.path.exists(plots_save_path):
            os.makedirs(plots_save_path)
        current_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        filename = plots_save_path + dataset_path.split('/')[-1] + ' with Outliers (' + current_time + ').png'
        pyplot.savefig(filename, dpi=500)
    pyplot.show()

    return time_series_with_outliers


if __name__ == "__main__":
    print('Detect_ARIMA_Anomalies.py is being run directly\n')

    ds_num = 0  # used to select dataset path and variable name together

    dataset = ['Data/BusVoltage.csv', 'Data/TotalBusCurrent.csv', 'Data/BatteryTemperature.csv',
               'Data/WheelTemperature.csv', 'Data/WheelRPM.csv'][ds_num]
    name = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Temperature (C)', 'RPM'][ds_num]

    time_series_with_outliers = detect_arima_anomalies(dataset_path=dataset, train_size=0.66,
                                                       order=(12, 0, 0), seasonal_order=(0, 1, 0),
                                                       seasonal_freq=2920, trend=None, var_name=name, verbose=True,
                                                       grid_search=False)

else:
    print('Detect_ARIMA_Anomalies.py is being imported into another module\n')
