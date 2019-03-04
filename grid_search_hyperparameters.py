# Standard modules
import progressbar
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


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

    return order

def grid_search_sarima_params(ts, freq):
    """Perform a grid search to return SARIMA hyperparameters (p,d,q)(P,D,Q,freq) for the given time series.

       Inputs:
           ts [pd Series]: A pandas Series with a DatetimeIndex and a column for numerical values.
           freq [int]:     The freq hyperparameter for this SARIMA model, i.e., the number of samples that make up one seasonal cycle.

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

    return order, seasonal_order