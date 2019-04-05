# Standard modules
import progressbar
import pandas as pd
import numpy as np
from matplotlib import pyplot

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def detect_anomalies_with_rolling_mean(ts, num_stds, window, verbose, var_name='Value', outlier_def='std'):
    """Detect outliers in the time series data by comparing points against [num_stds] standard deviations from a rolling mean.

       Inputs:
           ts [pd Series]:   A pandas Series with a DatetimeIndex and a column for numerical values.
           num_stds [float]: The number of standard deviations away from the mean used to define point outliers.
           window [int]:     Window size; the number of samples to include in the rolling mean.
           verbose [bool]:   When True, a plot of the rolling mean will be displayed before outliers are detected.

       Optional Inputs:
           var_name [str]:    The name of the dependent variable in the time series.
                              Default is 'Value'.
           outlier_def [str]: {'std', 'errors'} The definition of an outlier to be used. Can be 'std' for [num_stds] from the data's mean,
                              or 'errors' for [num_stds] from the mean of the errors.
                              Default is 'std'.

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, a columns for numerical values, and an Outlier column (True or False).
           predictions [pd Series]:                 The rolling mean, as a pandas Series with a DatetimeIndex and a column for the rolling mean.
           outliers [pd Series]:                    The detected outliers, as a pandas Series with a DatetimeIndex and a column for the outlier value.
           errors [pd Series]:                      The errors at each point, as a pandas Series with a DatetimeIndex and a column for the errors.

       Optional Outputs:
           None

       Example:
           time_series_with_outliers, predictions, outliers, errors = detect_anomalies_with_rolling_mean(time_series, 2, window, False)
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
        predictions = pd.Series(rolling_mean, index=ts.index)
        outliers = pd.Series()
        errors = pd.Series()
        time_series_with_outliers = pd.DataFrame({var_name: ts})
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
            widgets=[progressbar.FormatLabel('Time Series Outliers ')] + widgets,
            max_value=int(len(X))).start()

        if outlier_def == 'std':
            # Label outliers using standard deviations
            for t in range(len(X)):
                obs = X[t]
                y = rolling_mean[t]
                error = abs(y-obs)
                error_point = pd.Series(error, index=[ts.index[t]])
                errors = errors.append(error_point)
                if error > std*num_stds:
                    time_series_with_outliers.at[ts.index[t], 'Outlier'] = 'True'
                    outlier = pd.Series(obs, index=[ts.index[t]])
                    outliers = outliers.append(outlier)
                progress_bar_sliding_window.update(t)  # advance progress bar

        elif outlier_def == 'errors':
            # Populate errors
            for t in range(len(X)):
                obs = X[t]
                y = rolling_mean[t]
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
                y = rolling_mean[t]
                error = errors[t]
                if error > threshold:
                    time_series_with_outliers.at[ts.index[t], 'Outlier'] = 'True'
                    outlier = pd.Series(obs, index=[ts.index[t]])
                    outliers = outliers.append(outlier)
                progress_bar_sliding_window.update(t)  # advance progress bar

        return time_series_with_outliers, predictions, outliers, errors
