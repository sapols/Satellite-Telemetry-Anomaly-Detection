# Standard modules
import progressbar
import pandas as pd
import numpy as np
from matplotlib import pyplot

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def detect_anomalies_with_mean(ts, num_stds, verbose, var_name='Value'):
    """Detect outliers in the time series data by comparing points against [num_stds] standard deviations from the mean.

       Inputs:
           ts [pd Series]:   A pandas Series with a DatetimeIndex and a column for numerical values.
           num_stds [float]: The number of standard deviations away from the mean used to define point outliers.
           verbose [bool]:   When True, a plot of the dataset mean will be displayed before outliers are detected.

       Optional Inputs:
           var_name [str]: The name of the dependent variable in the time series.
                           Default is 'Value'.

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, a columns for numerical values, and an Outlier column (True or False).
           outliers [pd Series]: The detected outliers, as a pandas Series with a DatetimeIndex and a column for the outlier value.

       Optional Outputs:
           None

       Example:
           time_series_with_outliers, outliers = detect_anomalies_with_mean(time_series, 2, True)
       """

    # Gather statistics in preparation for outlier detection
    mean = float(ts.values.mean())
    mean_line = pd.Series(([mean] * len(ts)), index=ts.index)
    std = float(ts.values.std(ddof=0))
    X = ts.values
    outliers = pd.Series()
    time_series_with_outliers = pd.DataFrame({var_name: ts})
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
        widgets=[progressbar.FormatLabel('Time Series Outliers ')] + widgets,
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
