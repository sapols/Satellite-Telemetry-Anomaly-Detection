# Standard modules
import os
import progressbar
import pandas as pd
import numpy as np
from matplotlib import pyplot

# Custom modules
import nonparametric_dynamic_thresholding as ndt

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def detect_anomalies(ts, normal_model, var_name, ds_name, outlier_def='std', num_stds=2, verbose=False):
    """Detect outliers in the time series data by comparing points against a "normal" model.

       Inputs:
           ts [pd Series]:           A pandas Series with a DatetimeIndex and a column for numerical values.
           normal_model [pd Series]: ts [pd Series]:   A pandas Series with a DatetimeIndex and a column for numerical values.
           var_name [str]:           The name of the dependent variable in the time series.
           ds_name [str]:            The name of the time series dataset.


       Optional Inputs:
           outlier_def [str]: {'std', 'errors', 'dynamic'} The definition of an outlier to be used. Can be 'std' for [num_stds] from the data's mean,
                              'errors' for [num_stds] from the mean of the errors, or 'dynamic' for nonparametric dynamic thresholding
                              Default is 'std'.
           num_stds [float]:  The number of standard deviations away from the mean used to define point outliers (when applicable).
                              Default is 2.
           verbose [bool]:

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, a columns for numerical values, and an Outlier column (True or False).

       Optional Outputs:
           None

       Example:
           time_series_with_outliers = detect_anomalies(time_series, model,
                                                        'Temperature (C)', 'BatteryTemperature', 'dynamic')
    """

    X = ts.values
    Y = normal_model.values
    outliers = pd.Series()
    errors = pd.Series()
    time_series_with_outliers = pd.DataFrame({var_name: ts})
    time_series_with_outliers['Outlier'] = 'False'

    # Start a progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    progress_bar_sliding_window = progressbar.ProgressBar(
        widgets=[progressbar.FormatLabel('Time Series Outliers ')] + widgets,
        max_value=int(len(X))).start()

    # Define outliers by distance from "normal" model
    if outlier_def == 'std':
        # Label outliers using standard deviations
        std = float(X.std(ddof=0))
        for t in range(len(X)):
            obs = X[t]
            y = Y[t]
            error = abs(y - obs)
            error_point = pd.Series(error, index=[ts.index[t]])
            errors = errors.append(error_point)
            if error > std*num_stds:
                time_series_with_outliers.at[ts.index[t], 'Outlier'] = 'True'
                outlier = pd.Series(obs, index=[ts.index[t]])
                outliers = outliers.append(outlier)
            progress_bar_sliding_window.update(t)  # advance progress bar

    # Define outliers by distance from mean of errors
    elif outlier_def == 'errors':
        # Populate errors
        for t in range(len(X)):
            obs = X[t]
            y = Y[t]
            error = abs(y - obs)
            error_point = pd.Series(error, index=[ts.index[t]])
            errors = errors.append(error_point)

            progress_bar_sliding_window.update(t)  # advance progress bar

        mean_of_errors = float(errors.values.mean())
        std_of_errors = float(errors.values.std(ddof=0))
        threshold = mean_of_errors + (std_of_errors*num_stds)

        # Label outliers using standard deviations from the errors' mean
        for t in range(len(X)):
            obs = X[t]
            y = Y[t]
            error = errors[t]
            if error > threshold:
                time_series_with_outliers.at[ts.index[t], 'Outlier'] = 'True'
                outlier = pd.Series(obs, index=[ts.index[t]])
                outliers = outliers.append(outlier)
            progress_bar_sliding_window.update(t)  # advance progress bar

    # Define outliers using JPL's nonparamatric dynamic thresholding technique
    elif outlier_def == 'dynamic':
        smoothed_errors = ndt.get_errors(X, Y)

        # These are the results of the nonparametric dynamic thresholding
        E_seq, anom_scores = ndt.process_errors(X, smoothed_errors)

        # Convert sets of outlier start/end indices into outlier points
        for anom in E_seq:
            start = anom[0]
            end = anom[1]
            for i in range(start, end+1):
                outlier = pd.Series(X[i], index=[ts.index[i]])
                outliers = outliers.append(outlier)


    # TODO: save data and plots
    # Save plot to proper directory with encoded file name
    ax = ts.plot(color='#192C87', title=ds_name + ' with Rolling Mean', label=var_name, figsize=(14, 6))
    # TODO: pass in a str for the algorithm used to get model, for the label below
    normal_model.plot(color='#0CCADC', label='Model', linewidth=1.5)
    if len(outliers) > 0:
        print('\nDetected outliers: ' + str(len(outliers)) + '\n')
        outliers.plot(color='red', style='.')
    ax.set(xlabel='Time', ylabel=var_name)
    pyplot.legend(loc='best')

    # plot_filename = ds_name + '_with_rolling_mean.png'
    # plot_path = './save/datasets/' + ds_name + '/rolling mean/plots/'
    # if not os.path.exists(plot_path):
    #     os.makedirs(plot_path)
    # pyplot.savefig(plot_path + plot_filename, dpi=500)

    pyplot.show()

    return time_series_with_outliers
