# Standard modules
import os
import progressbar
import pandas as pd
from pandas import datetime
import numpy as np
from matplotlib import pyplot

# Custom modules
import nonparametric_dynamic_thresholding as ndt

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data


def detect_anomalies(ts, normal_model, ds_name, var_name, alg_name, outlier_def='std', num_stds=2,
                     plot_save_path=None, data_save_path=None):
    """Detect outliers in the time series data by comparing points against a "normal" model.

       Inputs:
           ts [pd Series]:           A pandas Series with a DatetimeIndex and a column for numerical values.
           normal_model [pd Series]: A pandas Series with a DatetimeIndex and a column for numerical values.
           ds_name [str]:            The name of the time series dataset.
           var_name [str]:           The name of the dependent variable in the time series.
           alg_name [str]:           The name of the algorithm used to create 'normal_model'.

       Optional Inputs:
           outlier_def [str]:    {'std', 'errors', 'dynamic'} The definition of an outlier to be used. Can be 'std' for [num_stds] from the data's mean,
                                 'errors' for [num_stds] from the mean of the errors, or 'dynamic' for nonparametric dynamic thresholding
                                 Default is 'std'.
           num_stds [float]:     The number of standard deviations away from the mean used to define point outliers (when applicable).
                                 Default is 2.
           plot_save_path [str]: The file path (ending in file name *.png) for saving plots of outliers.
           data_save_path [str]: The file path (ending in file name *.csv) for saving CSVs with outliers.

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, two columns for numerical values, and an Outlier column (True or False).

       Optional Outputs:
           None

       Example:
           time_series_with_outliers = detect_anomalies(time_series, model, 'BatteryTemperature', 'Temperature (C)',
                                                        'ARIMA', 'dynamic', plot_path, data_path)
    """

    X = ts.values
    Y = normal_model.values
    outliers = pd.Series()
    errors = pd.Series()
    time_series_with_outliers = pd.DataFrame({var_name: ts, alg_name: normal_model})
    time_series_with_outliers['Outlier'] = 'False'
    column_names = [var_name, alg_name, 'Outlier']  # column order
    time_series_with_outliers = time_series_with_outliers.reindex(columns=column_names)  # sort columns in specified order

    # Start a progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    progress_bar_sliding_window = progressbar.ProgressBar(
        widgets=[progressbar.FormatLabel('Outliers (' + ds_name + ')')] + widgets,
        maxval=int(len(X))).start()

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

    # Plot anomalies
    ax = ts.plot(color='#192C87', title=ds_name + ' with ' + alg_name + ' Outliers', label=var_name, figsize=(14, 6))
    normal_model.plot(color='#0CCADC', label=alg_name, linewidth=1.5)
    if len(outliers) > 0:
        print('Detected outliers (' + ds_name + '): ' + str(len(outliers)))
        outliers.plot(color='red', style='.', label='Outliers')
    ax.set(xlabel='Time', ylabel=var_name)
    pyplot.legend(loc='best')

    # Save plot
    if plot_save_path is not None:
        plot_dir = plot_save_path[:plot_save_path.rfind('/')+1]
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        pyplot.savefig(plot_save_path, dpi=500)

    pyplot.show()
    pyplot.clf()

    # Save data
    if data_save_path is not None:
        data_dir = data_save_path[:data_save_path.rfind('/')+1]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        time_series_with_outliers.to_csv(data_save_path)

    return time_series_with_outliers


def detect_anomalies_with_many_stds(ts, normal_model, ds_name, var_name, alg_name, outlier_def='std', stds=[2,4,8],
                                    plot_save_path=None, data_save_path=None):
    """Detect outliers in the time series data by comparing points against a "normal" model, using a set of three standard deviations as thresholds.

       Inputs:
           ts [pd Series]:           A pandas Series with a DatetimeIndex and a column for numerical values.
           normal_model [pd Series]: A pandas Series with a DatetimeIndex and a column for numerical values.
           ds_name [str]:            The name of the time series dataset.
           var_name [str]:           The name of the dependent variable in the time series.
           alg_name [str]:           The name of the algorithm used to create 'normal_model'.

       Optional Inputs:
           outlier_def [str]:    {'std', 'errors'} The definition of an outlier to be used. Can be 'std' for [num_stds] from the data's mean
                                 or 'errors' for [num_stds] from the mean of the errors.
                                 Default is 'std'.
           stds [3 floats]:      Exactly 3 numbers which will be successively used as the number of standard deviations away from the mean that constitutes an outlier.
                                 Plots will show outliers with all 3 stds and 3 CSVs will be saved.
                                 Default is [2,4,6].
           plot_save_path [str]: The file path (ending in file name *.png) for saving plots of outliers.
           data_save_path [str]: The file path (ending in file name *.csv) for saving. 3 CSVs get saved, each with the std used appended to the name.

       Outputs:
           time_series_with_outliers1 [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, two columns for numerical values, and an Outlier column (True or False).
                                                      Uses the first number in stds.
           time_series_with_outliers2 [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, two columns for numerical values, and an Outlier column (True or False).
                                                      Uses the second number in stds.
           time_series_with_outliers3 [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, two columns for numerical values, and an Outlier column (True or False).
                                                      Uses the third number in stds.

       Optional Outputs:
           None

       Example:
           df1, df2, df3 = detect_anomalies(time_series, model, 'BatteryTemperature', 'Temperature (C)', [2,4,6],
                                                        'ARIMA', 'dynamic', plot_path, data_path)
    """

    X = ts.values
    Y = normal_model.values
    outliers1 = pd.Series()
    outliers2 = pd.Series()
    outliers3 = pd.Series()
    errors = pd.Series()
    time_series_with_outliers = pd.DataFrame({var_name: ts, alg_name: normal_model})
    time_series_with_outliers['Outlier'] = 'False'
    column_names = [var_name, alg_name, 'Outlier']  # column order
    time_series_with_outliers1 = time_series_with_outliers.reindex(columns=column_names)  # sort columns in specified order
    time_series_with_outliers2 = time_series_with_outliers1
    time_series_with_outliers3 = time_series_with_outliers1

    # Start a progress bar
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    progress_bar_sliding_window = progressbar.ProgressBar(
        widgets=[progressbar.FormatLabel('Outliers (' + ds_name + ')')] + widgets,
        maxval=int(len(X)*len(stds))).start()

    for i in range(len(stds)):
        num_stds = stds[i]

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
                if error > std * num_stds:
                    if i == 0:
                        time_series_with_outliers1.at[ts.index[t], 'Outlier'] = 'True'
                    elif i == 1:
                        time_series_with_outliers2.at[ts.index[t], 'Outlier'] = 'True'
                    elif i == 2:
                        time_series_with_outliers3.at[ts.index[t], 'Outlier'] = 'True'
                    outlier = pd.Series(obs, index=[ts.index[t]])
                    if i == 0:
                        outliers1 = outliers1.append(outlier)
                    elif i == 1:
                        outliers2 = outliers2.append(outlier)
                    elif i == 2:
                        outliers3 = outliers3.append(outlier)
                progress_bar_sliding_window.update(t*(i+1))  # advance progress bar

        # Define outliers by distance from mean of errors
        elif outlier_def == 'errors':
            # Populate errors
            for t in range(len(X)):
                obs = X[t]
                y = Y[t]
                error = abs(y - obs)
                error_point = pd.Series(error, index=[ts.index[t]])
                errors = errors.append(error_point)

                progress_bar_sliding_window.update(t*(i+1))  # advance progress bar

            mean_of_errors = float(errors.values.mean())
            std_of_errors = float(errors.values.std(ddof=0))
            threshold = mean_of_errors + (std_of_errors * num_stds)

            # Label outliers using standard deviations from the errors' mean
            for t in range(len(X)):
                obs = X[t]
                y = Y[t]
                error = errors[t]
                if error > threshold:
                    if i == 0:
                        time_series_with_outliers1.at[ts.index[t], 'Outlier'] = 'True'
                    elif i == 1:
                        time_series_with_outliers2.at[ts.index[t], 'Outlier'] = 'True'
                    elif i == 2:
                        time_series_with_outliers3.at[ts.index[t], 'Outlier'] = 'True'
                    outlier = pd.Series(obs, index=[ts.index[t]])
                    if i == 0:
                        outliers1 = outliers1.append(outlier)
                    elif i == 1:
                        outliers2 = outliers2.append(outlier)
                    elif i == 2:
                        outliers3 = outliers3.append(outlier)
                progress_bar_sliding_window.update(t)  # advance progress bar

    # Plot anomalies
    ax = ts.plot(color='#192C87', title=ds_name + ' with ' + alg_name + ' Outliers', label=var_name, figsize=(14, 6))
    normal_model.plot(color='#0CCADC', label=alg_name, linewidth=1.5)
    if len(outliers1) > 0:
        print('Detected outliers (' + ds_name + ', ' + str(stds[0]) + ' stds): ' + str(len(outliers1)))
        outliers1.plot(color='orange', style='.', label='Outliers (' + str(stds[0]) + '$\sigma$)')
    if len(outliers2) > 0:
        print('Detected outliers (' + ds_name + ', ' + str(stds[1]) + ' stds): ' + str(len(outliers2)))
        outliers2.plot(color='orangered', style='.', label='Outliers (' + str(stds[1]) + '$\sigma$)')
    if len(outliers3) > 0:
        print('Detected outliers (' + ds_name + ', ' + str(stds[2]) + ' stds): ' + str(len(outliers3)))
        outliers3.plot(color='crimson', style='.', label='Outliers (' + str(stds[2]) + '$\sigma$)')
    ax.set(xlabel='Time', ylabel=var_name)
    pyplot.legend(loc='best')

    # Save plot
    if plot_save_path is not None:
        plot_dir = plot_save_path[:plot_save_path.rfind('/') + 1]
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        pyplot.savefig(plot_save_path, dpi=500)

    pyplot.show()
    pyplot.clf()

    # Save data
    if data_save_path is not None:
        data_dir = data_save_path[:data_save_path.rfind('/') + 1]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        file1 = data_save_path[:data_save_path.rfind('.csv')] + '_' + str(stds[0]) + '_stds.csv'
        file2 = data_save_path[:data_save_path.rfind('.csv')] + '_' + str(stds[1]) + '_stds.csv'
        file3 = data_save_path[:data_save_path.rfind('.csv')] + '_' + str(stds[2]) + '_stds.csv'
        time_series_with_outliers1.to_csv(file1)
        time_series_with_outliers2.to_csv(file2)
        time_series_with_outliers3.to_csv(file3)

    return time_series_with_outliers1, time_series_with_outliers2, time_series_with_outliers3



if __name__ == "__main__":

    datasets = ['Data/BusVoltage.csv', 'Data/TotalBusCurrent.csv', 'Data/BatteryTemperature.csv',
                'Data/WheelTemperature.csv', 'Data/WheelRPM.csv']

    # Rolling Mean
    for ds in range(len(datasets)):
        ds_name = datasets[ds][5:-4]  # drop 'Data/' and '.csv'

        file = 'save/datasets/' + ds_name + '/rolling mean/data/' + ds_name + '_with_rolling_mean.csv'
        ts_with_model = pd.read_csv(file, header=0, parse_dates=[0], index_col=0, date_parser=parser)
        var_name = ts_with_model.columns[0]
        alg_name = ts_with_model.columns[1]

        x = ts_with_model[var_name]
        y = ts_with_model[alg_name]

        # ts_with_outliers = detect_anomalies(x, y, ds_name, var_name, alg_name=alg_name, outlier_def='std', num_stds=2,
        #                                     plot_save_path='./test/plot.png', data_save_path='./test/data.csv')

        ts_with_outliers = detect_anomalies_with_many_stds(x, y, ds_name, var_name, alg_name=alg_name, outlier_def='std', stds=[2, 4, 8],
                                            plot_save_path='./test/plot.png', data_save_path='./test/data.csv')


else:
    print('\n')
