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
from matplotlib import pyplot
import statistics

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
        model_fit = model.fit(disp=1)  # TODO: pass in verbose and put this under "if verbose" for disp=1 else 0?
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# root mean squared error or rmse
# def measure_rmse(actual, predicted): #TODO: don't use this unnecessary func
#     return sqrt(mean_squared_error(actual, predicted))


# create a set of sarima configs to try
def generate_sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    # TODO: increase range of these lists? These are defaults from: https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
    # TODO: log/binary searching? (doubling/halving while errors go down; think big jumps through U-shape until finding local minimum, then refining)
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    freq_params = seasonal
    # create config instances (1,296 of them in total, but many will error and will get discarded)
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
def get_cross_validation_scores(data, order_configs, parallel=False):  # TODO: parallel should be True
    configs_with_scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, config) for config in order_configs)
        configs_with_scores = executor(tasks)
    else:
        configs_with_scores = [score_model(data, config) for config in order_configs]

    # remove empty results
    configs_with_scores = [r for r in configs_with_scores if (r[1] is not None and len(r[1])>0)]
    # sort configs by error, asc
    configs_with_scores.sort(key=lambda tup: float(statistics.mean(tup[1])))
    return configs_with_scores


# score a model, return None on failure
def score_model(data, config, debug=False):
    rmses = []

    # show all warnings and fail on exception if debugging
    if debug:
        rmses = nested_cross_validation(data, config)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                rmses = nested_cross_validation(data, config)
        except Exception as e:
            print(e)
            error = None
    # check for an interesting result
    if len(rmses) > 0:
        print(' > Model[%s] %s' % (str(config), str(rmses))) # TODO: "if verbose" or don't print
    return (config, rmses)


def nested_cross_validation(data, config, n_folds=5):
    # Split the data into n_folds+1 chunks
    data = pd.Series(data)
    # data.plot(color='blue', title='Holdout Data (before splitting into folds)') # TODO: delete me
    # pyplot.show()
    folds = []
    fold_size = len(data) / (n_folds+1)
    for i in range(n_folds+1):  # 0 through 5 when n_folds=5
        if i == n_folds:
            folds.append(pd.Series(data[i*fold_size:]))  # last fold gets any off-by-one remainder point
        else:
            folds.append(pd.Series(data[i*fold_size:(i*fold_size)+fold_size]))

    # I can trust that this logic splits the data into perfect folds
    # data.plot(color='black', title='Holdout Data (after splitting into folds)')  # TODO: delete me
    # folds[0].plot(color='blue')
    # folds[1].plot(color='green')
    # folds[2].plot(color='red')
    # folds[3].plot(color='purple')
    # folds[4].plot(color='orange')
    # folds[5].plot(color='pink')
    # pyplot.show()

    RMSEs = train_and_validate(folds, config)
    return RMSEs


def train_and_validate(folds, config):
    num_folds = len(folds)
    RMSEs = []

    for i in range(num_folds-1):  # 0 through 5 when num_folds=6
        num_training_folds = i+1
        training_folds = folds[:num_training_folds]
        training_data = pd.Series([])
        training_data = training_data.append(training_folds, verify_integrity=True)
        validation_data = folds[i+1]
        RMSEs.append(sarima_forecast_and_score(training_data, validation_data, config))

    return RMSEs


def sarima_forecast_and_score(training, validation, config):
    X = training.append(validation, verify_integrity=True)
    order = config[0]
    seasonal_order = config[1]
    trend = config[2]

    trained_model = SARIMAX(training, order=order, seasonal_order=seasonal_order, trend=trend)
    trained_model_fit = trained_model.fit(disp=1)

    predictions = trained_model_fit.predict(start=1, end=len(X)-1, typ='levels')
    predict_index = pd.Index(X.index[1:len(X)])
    predictions_with_index = pd.Series(predictions.values, index=predict_index)

    model_rmse = sqrt(mean_squared_error(X[1:len(X)], predictions_with_index))
    return model_rmse



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
    """Perform a grid search to return SARIMA hyperparameters (p,d,q)(P,D,Q,freq) and trend for the given time series.
       See: https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/

       Inputs:
           ts [pd Series]: A pandas Series with a DatetimeIndex and a column for numerical values.
           freq [int]:     The freq hyperparameter for this SARIMA model, i.e., toohe number of time steps for a single seasonal period.

       Optional Inputs:
           None

       Outputs:
           order [tuple]:          The order hyperparameters (p,d,q) for this SARIMA model.
           seasonal_order [tuple]: The seasonal order hyperparameters (P,D,Q,freq) for this SARIMA model.
           trend [str]:            The trend hyperparameter for this SARIMA model.

       Optional Outputs:
           None

       Example:
           order, seasonal_order, trend = grid_search_sarima_params(time_series, seasonal_freq)
       """

    trivial_data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0]
    # data = ts.values
    # holdout_size = 0.2
    # split = int(len(data) * holdout_size)
    # holdout_data = data[0:split]

    possible_order_configs = generate_sarima_configs([freq])

    configs_with_scores = get_cross_validation_scores(trivial_data, possible_order_configs)  # get cross validation scores for each order_config

    best_order_config = configs_with_scores[0][0]  # TODO: always returning the best score doesn't lead to constant overfitting, does it?

    order = best_order_config[0]
    seasonal_order = best_order_config[1]
    trend = best_order_config[2]

    return order, seasonal_order, trend

