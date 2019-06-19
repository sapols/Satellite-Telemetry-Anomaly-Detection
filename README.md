# Satellite-Telemetry-Anomaly-Detection
This is the repository for my research into anomaly detection for LASP spacecraft telemetry. It began as a semester of Independent Study for my master’s program at CU Boulder. The purpose was to explore unsupervised machine learning techniques that could be used to improve [WebTCAD](http://lasp.colorado.edu/home/mission-ops-data/tools-and-technologies/webtcad/)'s anomaly detection abilities.

## Link to the Paper
Read my paper, [_Unsupervised Machine Learning for Spacecraft Anomaly Detection in WebTCAD_](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/Paper/Unsupervised%20Machine%20Learning%20for%20Spacecraft%20Anomaly%20Detection%20in%20WebTCAD.pdf).

### Abstract
> This paper explores unsupervised machine learning techniques for anomaly detection in spacecraft telemetry with the aim of improving WebTCAD's automated detection abilities. WebTCAD is a tool for ad-hoc visualization and analysis of telemetry data that is built and maintained at the Laboratory for Atmospheric and Space Physics. This paper attempts to answer the question: "How good could machine learning for anomaly detection in WebTCAD be?" The techniques are applied to five representative time series datasets. Four algorithms are examined in depth: rolling means, ARIMA, autoencoders, and robust random cut forests. Then, three unsupervised anomaly definitions are examined: thresholding outlier scores with standard deviations from the data's mean, thresholding outlier scores with standard deviations from the scores' mean, and nonparametric dynamic thresholding. Observations from this exploration and suggestions for incorporating these ideas into WebTCAD and future work are included in the final two sections.

### Example 
A common pattern for unsupervised anomaly detection is to create a model of what is “normal” for a dataset, then when data “differs greatly” from that model, that is anomalous. Using the code in this repo, a time series dataset is first fed into an algorithm that models what is “normal” for that data, e.g., an autoencoder. Then, for a given anomaly definition, e.g., “an anomaly is any data point whose deviation from the model is greater than four standard deviations from the mean of the model errors,” the output is the dataset with an added column for labeling points as anomalies (true or false). A plot is optionally produced to visualize the results (see example below).

```python
import pandas as pd
from model_with_autoencoder import *
from detect_anomalies import *

dataset = 'Data/WheelTemperature.csv'
ds_name = 'WheelTemperature'
var_name = 'Temperature (C)'
alg_name = 'Autoencoder'

ts_with_model = autoencoder_prediction(dataset, ds_name, var_name=var_name, train_size=0.5)
X = ts_with_model[var_name]
Y = ts_with_model[alg_name]
ts_with_anomalies = detect_anomalies(X, Y, ds_name, var_name, alg_name, outlier_def='errors', num_stds=[2,4,8])
```

![pic](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/save/datasets/WheelTemperature/autoencoder/plots/50%20percent/WheelTemperature_autoencoder_half_outliers_from_error_mean.png)

## Brief Directory Descriptions

Below is a brief description of each directory in this repo:
 - [Data](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/Data): Contains the five datasets used in this research.
 - [Metadata](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/Metadata): Contains metadata for the five datasets used in this research.
 - [Paper](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/Paper): The research paper and the LaTeX code behind it.
 - [demo](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/demo): Contains the two Jupyter notebooks in which the demonstrations in the paper were performed.
 - [explore](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/explore): Contains various Jupyter notebooks that were not a part of the paper. They detect anomalies in datasets using a subset of the paper’s techniques (ARIMA and simple statistics).
 - [save](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/save): Contains all the data and plots produced by the demonstrations in the paper.

## Brief Code File Descriptions
Below is a brief description of each code file in this repo:
 - [model_with_arima.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/model_with_arima.py): Models a given time series dataset with an ARIMA forecast.
 - [model_with_autoencoder.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/model_with_autoencoder.py): Models a given time series dataset with an autoencoder's predictions.
 - [model_with_rolling_mean.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/model_with_rolling_mean.py): Models a given time series dataset with a rolling mean.
 - [score_with_rrcf.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/score_with_rrcf.py): Uses a robust random cut forest to assign anomaly scores to each data point in a given time series dataset.
 - [detect_anomalies.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/detect_anomalies.py): Detects anomalies in a given time series dataset by comparing points against a “normal” model and one of three anomaly definitions.
 - [nonparametric_dynamic_thresholding.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/nonparametric_dynamic_thresholding.py): Contains the code for the anomaly definition “nonparametric dynamic thresholding,” which is a thresholding approach [recently proposed by NASA'S JPL](https://arxiv.org/abs/1802.04431).
 - [grid_search_hyperparameters.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/grid_search_hyperparameters.py): Performs a grid search with nested cross-validation to return ARIMA hyperparameters _(p,d,q)(P,D,Q,m)_ and _trend_ for the given time series.
  - [correlate.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/correlate.py): Calculates the correlation coefficient between two sequences of numbers of equal length, given two CSV file paths and a column name for each file.

