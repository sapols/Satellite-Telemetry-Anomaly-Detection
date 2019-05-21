# Standard modules
import datetime
import os
import pandas as pd
from pandas import datetime
from pandas import read_csv
import numpy as np
from matplotlib import pyplot
import rrcf
import progressbar

# Custom modules
import model_with_autoencoder as mwauto
from model_with_autoencoder import shingle

__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data



if __name__ == "__main__":
    print('rrcf_anomaly_scores.py is being run directly\n')

    ds_num = 0  # used to select dataset path and variable name together

    dataset = ['Data/BusVoltage.csv', 'Data/TotalBusCurrent.csv', 'Data/BatteryTemperature.csv',
               'Data/WheelTemperature.csv', 'Data/WheelRPM.csv'][ds_num]
    var_name = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Temperature (C)', 'RPM'][ds_num]

    ds_name = dataset[5:-4]  # drop 'Data/' and '.csv'

    ts = pd.read_csv(dataset, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    ts = ts.iloc[:int(len(ts)/4)]  #TODO: remove
    ts_shingled = mwauto.shingle(ts, 18)
    compressed_feature_vectors = np.load('save/datasets/' + ds_name + '/autoencoder/data/50 percent/' + ds_name + '_compressed_by_autoencoder_half.npy')

    # RRCF code
    # tree = rrcf.RCTree(compressed_feature_vectors)

    # # Generate data
    # n = 730
    # A = 50
    # center = 100
    # phi = 30
    # T = 2 * np.pi / 100
    # t = np.arange(n)
    # sin = A * np.sin(T * t - phi * T) + center
    # sin[235:255] = 80

    # Set tree parameters
    num_trees = 2  # TODO: was 100
    shingle_size = 18
    tree_size = 256

    # Create a forest of empty trees
    forest = []
    for _ in range(num_trees):
        tree = rrcf.RCTree()
        forest.append(tree)

    # Use the "shingle" generator to create rolling window
    points = rrcf.shingle(ts, size=shingle_size) #points = rrcf.shingle(sin, size=shingle_size)

    # Create a dict to store anomaly score of each point
    avg_codisp = {}

    # # Start a progress bar
    # widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
    # progress_bar = progressbar.ProgressBar(
    #     widgets=[progressbar.FormatLabel('RRCF (' + ds_name + ')')] + widgets,
    #     maxval=int(len(ts)/shingle_size)).start()
    # progress = 0

    # For each shingle...
    for index, point in enumerate(points):
        # For each tree in the forest...
        for tree in forest:
            # If tree is above permitted size, drop the oldest point (FIFO)
            if len(tree.leaves) > tree_size:
                tree.forget_point(index - tree_size)
            # Insert the new point into the tree
            tree.insert_point(point, index=index)
            # Compute codisp on the new point and take the average among all trees
            if not index in avg_codisp:
                avg_codisp[index] = 0
            avg_codisp[index] += tree.codisp(index) / num_trees
        # progress = progress + 1
        # progress_bar.update(progress)  # advance progress bar

    # Plot
    fig, ax1 = pyplot.subplots(figsize=(14, 6))

    score_color = '#0CCADC'
    ax1.set_ylabel('CoDisp', color=score_color)
    ax1.set_xlabel('Time')
    lns1 = ax1.plot(pd.Series(avg_codisp).sort_index(), label='Anomaly Score', color=score_color)
    ax1.tick_params(axis='y', labelcolor=score_color)
    ax1.grid(False)
    # ax1.set_ylim(0, 160)
    ax2 = ax1.twinx()
    data_color = '#192C87'
    ax2.set_ylabel(var_name, color=data_color)
    lns2 = ax2.plot(ts.values, label=var_name, color=data_color)
    ax2.tick_params(axis='y', labelcolor=data_color)
    ax2.grid(False)
    pyplot.title(ds_name + ' (navy blue) and Anomaly Score (light blue)')
    # make the legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    pyplot.show()
    pyplot.clf()
