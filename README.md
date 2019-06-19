# Satellite-Telemetry-Anomaly-Detection
This is a repository for my research into anomaly detection for LASP spacecraft telemetry. It began as a semester of Independent Study for my master’s program at CU Boulder. The purpose was to explore unsupervised machine learning techniques that could be used to improve [WebTCAD](http://lasp.colorado.edu/home/mission-ops-data/tools-and-technologies/webtcad/)'s anomaly detection abilities.

## Link to the Paper
Read my paper, [_Unsupervised Machine Learning for Spacecraft Anomaly Detection in WebTCAD_](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/Paper/Unsupervised%20Machine%20Learning%20for%20Spacecraft%20Anomaly%20Detection%20in%20WebTCAD.pdf).

### Abstract
> This paper explores unsupervised machine learning techniques for anomaly detection in spacecraft telemetry with the aim of improving WebTCAD's automated detection abilities. WebTCAD is a tool for ad-hoc visualization and analysis of telemetry data that is built and maintained at the Laboratory for Atmospheric and Space Physics. This paper attempts to answer the question: "How good could machine learning for anomaly detection in WebTCAD be?" The techniques are applied to five representative time series datasets. Four algorithms are examined in depth: rolling means, ARIMA, autoencoders, and robust random cut forests. Then, three unsupervised anomaly definitions are examined: thresholding outlier scores with standard deviations from the data's mean, thresholding outlier scores with standard deviations from the scores' mean, and nonparametric dynamic thresholding. Observations from this exploration and suggestions for incorporating these ideas into WebTCAD and future work are included in the final two sections.

## Overview

![pic](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/save/datasets/WheelTemperature/autoencoder/plots/50%20percent/WheelTemperature_autoencoder_half_outliers_from_error_mean.png)

## Brief Directory Descriptions

Below is a brief description of each directory in this repo:
 - [Data](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/Data): Contains the five datasets used in this research 
 - [Metadata](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/Metadata): Contains metadata for the five datasets used in this research 
 - [Paper](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/Paper): The research paper and the LaTex code behind it
 - [demo](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/demo): Contains the two Jupyter notebooks that executed the demonstrations in this research
 - [explore](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/explore): Contains various Jupyter notebooks that were not actually a part of the research but detect anomalies in the five datasets using a subset of techniques (ARIMA and simple statistics)
 - [save](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/save): Contains all the data and plots produced by the demonstrations in demo

## Brief Code File Descriptions
Below is a brief description of each code file in this repo:
 - [Detect_ARIMA_Anomalies.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/Detect_ARIMA_Anomalies.py): j
 - [Detect_Standard_Deviation_Anomalies.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/Detect_Standard_Deviation_Anomalies.py): j
 - [correlate.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/correlate.py): Calculates the correlation coefficients between two sequences of numbers of equal length, given two CSV file paths and a column name within each file.
 - [detect_anomalies.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/detect_anomalies.py): Detects outliers in time series data by comparing points against a “normal” model and a given outlier definition.
 - [detect_anomalies_with_ARIMA.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/detect_anomalies_with_ARIMA.py): j
 - [detect_anomalies_with_mean.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/detect_anomalies_with_mean.py): j
 - [detect_anomalies_with_rolling_mean.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/detect_anomalies_with_rolling_mean.py): j
 - [grid_search_hyperparameters.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/grid_search_hyperparameters.py): j
 - [model_with_arima.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/model_with_arima.py): j
 - [model_with_autoencoder.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/model_with_autoencoder.py): j
 - [model_with_rolling_mean.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/model_with_rolling_mean.py): j
 - [nonparametric_dynamic_thresholding.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/nonparametric_dynamic_thresholding.py): j
 - [score_with_rrcf.py](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/score_with_rrcf.py): j
