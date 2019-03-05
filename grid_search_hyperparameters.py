# Standard modules
# TODO: use a progressbar?
# import progressbar
from ast import literal_eval as make_tuple
import pandas as pd
from math import sqrt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
import warnings
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


#----Helper Functions---------------------------------------------------------------------------------------------------

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


# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# split a univariate dataset into train/test sets
def train_test_split(data, train_size):
    split = int(len(data) * train_size)
    return data[0:split], data[split:len(data)]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# walk-forward validation for univariate data
def walk_forward_validation(data, train_size, cfg):  #TODO: make this 5-fold cross validation?
    predictions = list()
    # split dataset
    train, test = train_test_split(data, train_size)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, train_size, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, train_size, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, train_size, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    # TODO: increase range of these lists? These are defaults from: https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    # p_params = [0, 1]
    # d_params = [0]
    # q_params = [0]
    # t_params = ['n']
    # P_params = [0]
    # D_params = [0]
    # Q_params = [0]
    freq_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in freq_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models


# grid search configs
def grid_search(data, cfg_list, train_size, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, train_size, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, train_size, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


#----Grid Search Functions----------------------------------------------------------------------------------------------

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
       See: https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/

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

    #data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    data = ts.values
    train_size = 0.66  # TODO: specify size instead of hardcoding 0.66?
    cfg_list = sarima_configs([freq])

    scores = grid_search(data, cfg_list, train_size)

    best_cfg = scores[0]

    order_str = best_cfg[0][1:best_cfg[0].find(')')+1]
    order = make_tuple(order_str)

    seasonal_order_str = best_cfg[0][1+len(order_str):]
    seasonal_order_str = seasonal_order_str[seasonal_order_str.find('('):seasonal_order_str.find(')')+1]
    seasonal_order = make_tuple(seasonal_order_str)

    trend = ''  # TODO: get trend

    return order, seasonal_order  # TODO: return trend too