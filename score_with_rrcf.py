# Standard modules
import datetime
import os
import pandas as pd
from pandas import datetime
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
import numpy as np
from matplotlib import pyplot
import rrcf
import progressbar
register_matplotlib_converters()

# Custom modules
from model_with_autoencoder import *


__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data


def score_with_rrcf(dataset_path, ds_name, var_name, num_trees=100, shingle_size=18, tree_size=256):
    """Get anomaly scores for each point in the given time series using a robust random cut forest.

       Inputs:
           dataset_path [str]: A string path to the time series data. Data is read as a pandas Series with a DatetimeIndex and a column for numerical values.
           ds_name [str]:      The name of the dataset.
           var_name [str]:     The name of the dependent variable in the time series.

       Optional Inputs:
           num_trees [int]:    The number of trees in the generated forest.
                               Default is 100.
           shingle_size [int]: The size of each shingle when shingling the time series.
                               Default is 18.
           tree_size [int]:    The size of each tree in the generated forest.
                               Default is 256.

       Outputs:
            ts_with_scores [pd DataFrame]: The original time series with an added column for anomaly scores.

       Optional Outputs:
           None

       Example:
           time_series_with_anomaly_scores = score_with_rrcf(dataset, ds_name, var_name)
       """

    ts = pd.read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

    # Set tree parameters
    num_trees = num_trees
    shingle_size = shingle_size
    tree_size = tree_size

    # Create a forest of empty trees
    forest = []
    for _ in range(num_trees):
        tree = rrcf.RCTree()
        forest.append(tree)

    # Use the "shingle" generator to create rolling window
    points = rrcf.shingle(ts, size=shingle_size)

    # Create a dict to store anomaly score of each point
    avg_codisp = {}

    # For each shingle...
    for index, point in enumerate(points):
        # for each tree in the forest...
        for tree in forest:
            # if tree is above permitted size, drop the oldest point (FIFO)
            if len(tree.leaves) > tree_size:
                tree.forget_point(index - tree_size)
            # insert the new point into the tree
            tree.insert_point(point, index=index)
            # compute codisp on the new point and take the average among all trees
            if not index in avg_codisp:
                avg_codisp[index] = 0
            avg_codisp[index] += tree.codisp(index) / num_trees

    # Plot
    fig, ax1 = pyplot.subplots(figsize=(14, 6))

    score_color = '#0CCADC'
    ax1.set_ylabel('CoDisp', color=score_color)
    ax1.set_xlabel('Time')
    anom_score_series = pd.Series(list(avg_codisp.values()),
                                  index=ts.index[:-(shingle_size - 1)])  # TODO: ensure data and index line up perfectly
    lns1 = ax1.plot(anom_score_series.sort_index(), label='RRCF Anomaly Score',
                    color=score_color)  # Plot this series to get dates on the x-axis instead of number indices
    ax1.tick_params(axis='y', labelcolor=score_color)
    ax1.grid(False)
    max_ylim = float(anom_score_series.max())
    ax1.set_ylim(0, max_ylim)
    ax2 = ax1.twinx()
    data_color = '#192C87'
    ax2.set_ylabel(var_name, color=data_color)
    lns2 = ax2.plot(ts, label=var_name, color=data_color)
    ax2.tick_params(axis='y', labelcolor=data_color)
    ax2.grid(False)
    pyplot.title(ds_name + ' and Anomaly Score')
    # make the legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    # Save plot
    plot_filename = ds_name + '_with_rrcf_scores.png'
    plot_path = './save/datasets/' + ds_name + '/rrcf/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    pyplot.savefig(plot_path + plot_filename, dpi=500)

    pyplot.show()
    pyplot.clf()

    # Save data
    ts_with_scores = pd.DataFrame({'RRCF Anomaly Score': anom_score_series, var_name: ts})
    ts_with_scores.rename_axis('Time', axis='index', inplace=True)  # name index 'Time'
    column_names = [var_name, 'RRCF Anomaly Score']  # column order
    ts_with_scores = ts_with_scores.reindex(columns=column_names)  # sort columns in specified order

    data_filename = ds_name + '_with_rrcf_scores.csv'
    data_path = './save/datasets/' + ds_name + '/rrcf/data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    ts_with_scores.to_csv(data_path + data_filename)

    return ts_with_scores


def score_features_with_rrcf(dataset_path, features_path, ds_name, var_name, num_trees=100, shingle_size=18,
                             tree_size=256, chunk_size=18):
    """Get anomaly scores for each point in the given time series by feeding a feature vector representation of it into a robust random cut forest.

       Inputs:
           dataset_path [str]:  A string path to the time series data. Data is read as a pandas Series with a DatetimeIndex and a column for numerical values.
           features_path [str]: A string path to a feature vector representation of the time series. Must be a saved numpy ndarray where each element is an array of numbers.
                                The length of the ndarray must be equal to the length of the time series after chunking.
           ds_name [str]:       The name of the dataset.
           var_name [str]:      The name of the dependent variable in the time series.

       Optional Inputs:
           num_trees [int]:    The number of trees in the generated forest.
                               Default is 100.
           shingle_size [int]: The size of each shingle when shingling the time series.
                               Default is 18.
           tree_size [int]:    The size of each tree in the generated forest.
                               Default is 256.
           chunk_size [int]:   The size of each elemental list when "chunking" the time series into an ndarray.
                               The length of the chunked time series must be equal to the length of the feature vector representation.

       Outputs:
            ts_with_scores [pd DataFrame]: The original time series with an added column for anomaly scores.

       Optional Outputs:
           None

       Example:
           time_series_with_anomaly_scores = score_with_rrcf(dataset, compressed_feature_vectors, ds_name, var_name)
       """

    ts = pd.read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    features = np.load(features_path)
    ts_chunked = chunk(ts, chunk_size)

    assert len(features) == len(ts_chunked),'The length of the chunked time series must be equal to the length of the feature vector representation.'

    # Set tree parameters
    num_trees = num_trees
    shingle_size = shingle_size
    tree_size = tree_size

    # Create a forest of empty trees
    forest = []
    for _ in range(num_trees):
        tree = rrcf.RCTree()
        forest.append(tree)

    # Create a dict to store anomaly score of each point (each feature vector)
    avg_codisp = {}

    # For each shingle...
    for index, point in enumerate(features):
        # for each tree in the forest...
        for tree in forest:
            # if tree is above permitted size, drop the oldest point (FIFO)
            if len(tree.leaves) > tree_size:
                tree.forget_point(index - tree_size)
            # insert the new point into the tree
            tree.insert_point(point, index=index)
            # compute codisp on the new point and take the average among all trees
            if not index in avg_codisp:
                avg_codisp[index] = 0
            avg_codisp[index] += tree.codisp(index) / num_trees

    anom_scores = list(avg_codisp.values())
    assert len(ts_chunked) == len(anom_scores)

    # Assign feature vector anomaly scores to each corresponding time series chunk (actually not necessary!)
    # ts_chunked_with_scores = []
    # for i in range(len(ts_chunked)):
    #     ts_chunked_with_scores.append((ts_chunked[i], anom_scores[i]))

    # Reassociate anomaly scores with un-chunked time series by duplicating each score chunk_size at a time
    unchunked_scores = []
    for score in anom_scores:
        duplicated_score = [score] * chunk_size
        for s in duplicated_score:
            unchunked_scores.append(s)

    remainder = len(ts) % chunk_size
    ts = ts.iloc[remainder:]  # if ts isn't divisible by chunk_size, chunk() drops the first [remainder] data points; do the same here
    assert len(unchunked_scores) == len(ts)

    anom_score_series = pd.Series(unchunked_scores, index=ts.index)

    # Plot
    fig, ax1 = pyplot.subplots(figsize=(14, 6))

    score_color = '#0CCADC'
    ax1.set_ylabel('CoDisp', color=score_color)
    ax1.set_xlabel('Time')
    lns1 = ax1.plot(anom_score_series.sort_index(), label='RRCF Anomaly Score',
                    color=score_color)  # Plot this series to get dates on the x-axis instead of number indices
    ax1.tick_params(axis='y', labelcolor=score_color)
    ax1.grid(False)
    max_ylim = float(anom_score_series.max())
    ax1.set_ylim(0, max_ylim)
    ax2 = ax1.twinx()
    data_color = '#192C87'
    ax2.set_ylabel(var_name, color=data_color)
    lns2 = ax2.plot(ts, label=var_name, color=data_color)
    ax2.tick_params(axis='y', labelcolor=data_color)
    ax2.grid(False)
    pyplot.title(ds_name + ' and Anomaly Score')
    # make the legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    # Save plot
    plot_filename = ds_name + '_with_rrcf_scores_from_feature_vectors.png'
    plot_path = './save/datasets/' + ds_name + '/rrcf/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    pyplot.savefig(plot_path + plot_filename, dpi=500)

    pyplot.show()
    pyplot.clf()

    # Save data
    ts_with_scores = pd.DataFrame({'RRCF Anomaly Score': anom_score_series, var_name: ts})
    ts_with_scores.rename_axis('Time', axis='index', inplace=True)  # name index 'Time'
    column_names = [var_name, 'RRCF Anomaly Score']  # column order
    ts_with_scores = ts_with_scores.reindex(columns=column_names)  # sort columns in specified order

    data_filename = ds_name + '_with_rrcf_scores_from_feature_vectors.csv'
    data_path = './save/datasets/' + ds_name + '/rrcf/data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    ts_with_scores.to_csv(data_path + data_filename)

    return ts_with_scores



# if __name__ == "__main__":
#     print('score_with_rrcf.py is being run directly\n')
#
#     ds_num = 1  # used to select dataset path and variable name together
#
#     dataset = ['Data/BusVoltage.csv', 'Data/TotalBusCurrent.csv', 'Data/BatteryTemperature.csv',
#                'Data/WheelTemperature.csv', 'Data/WheelRPM.csv'][ds_num]
#     var_name = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Temperature (C)', 'RPM'][ds_num]
#
#     ds_name = dataset[5:-4]  # drop 'Data/' and '.csv'
#
#     cfv = 'save/datasets/' + ds_name + '/autoencoder/data/50 percent/' + ds_name + '_compressed_by_autoencoder_half.npy'
#
#     time_series_with_scores = score_features_with_rrcf(dataset, cfv, ds_name, var_name)

