# Standard modules
import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def detect_anomalies_with_arima(ts, size, order, seasonal_order=(), grid_search=False, path_to_model='',
                                 verbose=False, var_name='Value'):
    """Detect outliers in the time series data by comparing points against [num_stds] standard deviations from a rolling mean.

       Inputs:
           ts [pd Series]:         A pandas Series with a DatetimeIndex and a column for numerical values.
           size [float]:           The percentage of data to use for training, as a float (e.g., 0.66).
           order [tuple]:          The order hyperparameters (p,d,q) for this ARIMA model.


       Optional Inputs:
           seasonal_order [tuple]: The seasonal order hyperparameters (P,D,Q,freq) for this (S)ARIMA model.
           grid_search [bool]:     When True, perform a grid search to set values for the 'order' and 'seasonal order' hyperparameters.
                                   Note this overrides any given values. Default is False.
           path_to_model [str]:    Path to a *.pkl file of a trained SARIMA model. When set, no training will be done because that model will be used.
           verbose [bool]:         ??? When True, ???
           var_name [str]:         The name of the dependent variable in the time series.
                                   Default is 'Value'.

       Outputs:
           time_series_with_outliers [pd DataFrame]: A pandas DataFrame with a DatetimeIndex, a columns for numerical values, and an Outlier column (True or False).
           outliers [pd Series]: The detected outliers, as a pandas Series with a DatetimeIndex and a column for the outlier value.

       Optional Outputs:
           None

       Example:
           time_series_with_outliers, outliers = detect_anomalies_with_arima(time_series, order=(12, 0, 0),
                                                                              seasonal_order=(0, 1, 0, 365),
                                                                              size=0.8, verbose=False)
    """

    # Forecast with training data
    X = ts
    split = int(len(X) * size)
    train, test = X[0:split], X[split:len(X)]
    std = float(train.values.std(ddof=0))

    if grid_search:  #TODO: decide how to differentiate ARIMA vs. SARIMA for grid search
        if verbose:
            lag_acf = acf(ts, nlags=20)
            lag_pacf = pacf(ts, nlags=20, method='ols')
            pyplot.show()
        pass  # TODO: grid search (make a separate file for it?)
        #order, seasonal_order = grid_search_sarima_params(???) #TODO: can freq be found?? Or must it be given upfront?
        print('Grid search found hyperparameters:')
        print(str(order) + ' ' + str(seasonal_order))

    if len(seasonal_order) < 4:
        trained_model = ARIMA(train, order=order)
    else:
        trained_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)

    if path_to_model != '':
        # load pre-trained model
        trained_model_fit = ARIMAResults.load(path_to_model)
    else:
        trained_model_fit = trained_model.fit(disp=1)
        # save just-trained model
        try:
            current_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            filename = 'SARIMA_' + str(order) + '_' + str(seasonal_order) + '_' + current_time + '.pkl'
            trained_model_fit.save(filename)
        except Exception as e:
            print("Saving model failed:")
            print(e)

    print(trained_model_fit.summary())

    if verbose:
        # plot residual errors
        residuals = pd.DataFrame(trained_model_fit.resid)
        residuals.plot(title='Training Model Fit Residual Errors')
        pyplot.show()
        residuals.plot(kind='kde', title='Training Model Fit Residual Error Density')
        pyplot.show()
        print(residuals.describe())

    predictions = trained_model_fit.predict(start=1, end=len(X) - 1, typ='levels')
    predict_index = pd.Index(X.index[1:len(X)])
    predictions_with_dates = pd.Series(predictions.values, index=predict_index)
    outliers = pd.Series()
    time_series_with_outliers = pd.DataFrame({var_name: ts}) # TODO: use this correctly
    time_series_with_outliers['Outlier'] = 'False'

    # Label outliers using SARIMA forecast
    for t in range(len(test)):
        obs = test[t]
        yhat = predictions_with_dates[test.index[t]]
        print('predicted=%f, expected=%f' % (yhat, obs))
        if abs(yhat - obs) > std:  #TODO: decide std scheme
            time_series_with_outliers.at[ts.index[t], 'Outlier'] = 'True'
            outlier = pd.Series(obs, index=[test.index[t]])
            outliers = outliers.append(outlier)

    model_error = mean_squared_error(X[1:len(X)], predictions_with_dates)
    test_error = mean_squared_error(test, predictions_with_dates[test.index[0]:test.index[-1]])

    print('Test MSE: %.3f' % test_error)
    print('MSE: %.3f' % model_error)

    # plot the forecast TODO: move this under verbose
    X.plot(color='black', title='Dataset with Forecast and Outliers')  #TODO: pass in dataset name?
    test.plot(color='blue')
    predictions_with_dates.plot(color='green')
    if len(outliers) > 0:
        print('Outliers: ' + str(len(outliers)) + "\n")
        outliers.plot(color='red', style='.')
    pyplot.show()

    return time_series_with_outliers, outliers
