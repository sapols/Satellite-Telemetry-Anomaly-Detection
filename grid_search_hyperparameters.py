# Standard modules
# TODO: use a progressbar?
# import progressbar
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


# Evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)  # TODO: specify size instead of hardcoding 0.66?
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)  # TODO: pass in verbose and put this under "if verbose" for disp=1?
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# Evaluate an SARIMA model for a given order (p,d,q)(P,D,Q,freq)
def evaluate_sarima_model(X, sarima_order, sarima_seasonal_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)  # TODO: specify size instead of hardcoding 0.66?
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = SARIMAX(history, order=sarima_order, seasonal_order=sarima_seasonal_order)
        model_fit = model.fit(disp=0)  # TODO: pass in verbose and put this under "if verbose" for disp=1?
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


def grid_search_arima_params(ts):
    """Perform a grid search to return ARIMA hyperparameters (p,d,q) for the given time series.

       Inputs:
           ts [pd Series]: A pandas Series with a DatetimeIndex and a column for numerical values.

       Optional Inputs:
           None

       Outputs:
           order [tuple]: The order hyperparameters (p,d,q) for this ARIMA model.

       Optional Outputs:
           None

       Example:
           order = grid_search_arima_params(time_series)
       """

    warnings.filterwarnings("ignore")  # helps prevent junk from cluttering up console output
    # TODO: don't hardcode these values? pass them in? increase range for p_values and q_values?
    p_values = range(0, 9)
    d_values = range(0, 3)
    q_values = range(0, 6)
    # Evaluate combinations of p, d and q values for an ARIMA model
    dataset = ts.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    order = best_cfg  # TODO: always returning the best score doesn't lead to constant overfitting, does it?
    return order


def grid_search_sarima_params(ts, freq):
    """Perform a grid search to return SARIMA hyperparameters (p,d,q)(P,D,Q,freq) for the given time series.

       Inputs:
           ts [pd Series]: A pandas Series with a DatetimeIndex and a column for numerical values.
           freq [int]:     The freq hyperparameter for this SARIMA model, i.e., toohe number of time steps for a single seasonal period.

       Optional Inputs:
           None

       Outputs:
           order [tuple]: The order hyperparameters (p,d,q) for this SARIMA model.
           seasonal_order [tuple]: The seasonal order hyperparameters (P,D,Q,freq) for this SARIMA model.

       Optional Outputs:
           None

       Example:
           order, seasonal_order = grid_search_sarima_params(time_series, seasonal_freq)
       """

    # see: https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
    return order, seasonal_order
