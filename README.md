# Satellite-Telemetry-Anomaly-Detection
Code repo for my independent study into anomaly detection for LASP spacecraft telemetry. The aim was to explore ways to improve [WebTCAD](http://lasp.colorado.edu/home/mission-ops-data/tools-and-technologies/webtcad/)'s anomaly detection capabilities.


## Link to the Paper
Read the research paper, _Unsupervised Machine Learning for Spacecraft Anomaly Detection in WebTCAD_, [here](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/blob/master/Paper/Unsupervised%20Machine%20Learning%20for%20Spacecraft%20Anomaly%20Detection%20in%20WebTCAD.pdf).

### Abstract
> This paper explores unsupervised machine learning techniques for anomaly detection in spacecraft telemetry with the aim of improving WebTCAD's automated detection abilities. WebTCAD is a tool for ad-hoc visualization and analysis of telemetry data that is built and maintained at the Laboratory for Atmospheric and Space Physics. This paper attempts to answer the question: "How good could machine learning for anomaly detection in WebTCAD be?" The techniques are applied to five representative time series datasets. Four algorithms are examined in depth: rolling means, ARIMA, autoencoders, and robust random cut forests. Then, three unsupervised anomaly definitions are examined: thresholding outlier scores with standard deviations from the data's mean, thresholding outlier scores with standard deviations from the scores' mean, and nonparametric dynamic thresholding. Observations from this exploration and suggestions for incorporating these ideas into WebTCAD and future work are included in the final two sections.

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
