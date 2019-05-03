# Standard modules
import progressbar
import pandas as pd
import numpy as np
from matplotlib import pyplot


__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def model_with_rolling_mean(ts, window, verbose, var_name='Value'):
    """Model the time series data with a rolling mean.

       Inputs:
           ts [pd Series]: A pandas Series with a DatetimeIndex and a column for numerical values.
           window [int]:   Window size; the number of samples to include in the rolling mean.
           verbose [bool]: When True, a plot of the rolling mean will be displayed.

       Optional Inputs:
           var_name [str]: The name of the dependent variable in the time series.
                           Default is 'Value'.

       Outputs:
           rolling_mean [pd Series]: The rolling mean, as a pandas Series with a DatetimeIndex and a column for the rolling mean.
           errors [pd Series]:       The errors at each point, as a pandas Series with a DatetimeIndex and a column for the errors.

       Optional Outputs:
           None

       Example:
           rolling_mean, errors = detect_anomalies_with_rolling_mean(time_series, window_size, False)
    """

    if window <= 0:
        raise ValueError('\'window\' must be given a value greater than 0 when using rolling mean.')

    # Gather statistics
    rolling_mean = ts.rolling(window=window, center=False).mean()
    first_window_mean = ts.iloc[:window].mean()
    for i in range(window):  # fill first 'window' samples with mean of those samples
        rolling_mean[i] = first_window_mean
    X = ts.values
    rolling_mean = pd.Series(rolling_mean, index=ts.index)
    errors = pd.Series()

    # TODO: save data & plots to proper directories with encoded file names
    if verbose:
        # TODO: finalize coloring
        pyplot.plot(ts, color='black', label='Time Series')
        pyplot.plot(rolling_mean, color='blue', label='Rolling Mean')
        pyplot.legend(loc='best')
        pyplot.title('Time Series & Rolling Mean')
        pyplot.show()

    # Start a progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    progress_bar_sliding_window = progressbar.ProgressBar(
        widgets=[progressbar.FormatLabel('Rolling Mean errors ')] + widgets,
        maxval=int(len(X))).start()

    # Get errors
    for t in range(len(X)):
        obs = X[t]
        y = rolling_mean[t]
        error = abs(y-obs)
        error_point = pd.Series(error, index=[ts.index[t]])
        errors = errors.append(error_point)

        progress_bar_sliding_window.update(t)  # advance progress bar

    return rolling_mean, errors
