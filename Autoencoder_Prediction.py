# Standard modules
import datetime
import os
import pandas as pd
from pandas import datetime
from pandas import read_csv
import numpy as np
from matplotlib import pyplot
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from tensorflow import set_random_seed
from sklearn import preprocessing
from sklearn.preprocessing import normalize


__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data


def shingle(ts, window_size=18):
    remainder = len(ts) % window_size  # if ts isn't divisible by window_size, drop the first [remainder] data points

    shingled_ts = np.array([ts.values[remainder:window_size+remainder]])
    for i in range(window_size+remainder, len(ts), window_size):
        shingle = []
        for j in range(i, i+window_size):
            shingle.append(ts.values[j])
        shingled_ts = np.append(shingled_ts, [shingle], axis=0)

    return shingled_ts


def get_autoencoder_predictions(encoder, decoder, ts):
    predictions = []

    for i in range(len(ts)):
        inputs = np.array([ts[i]])
        x = encoder.predict(inputs)  # the compressed representation
        y = decoder.predict(x)[0]    # the decoded output
        predictions = predictions + y.tolist()

    return predictions


def seedy(s):
    np.random.seed(s)
    set_random_seed(s)


class AutoEncoder:
    # TODO: with shingles of 18, the shape of this net is (18,10,18). Make it (18,150,75,10,75,150,18)
    def __init__(self, data, encoding_dim=3):
        self.encoding_dim = encoding_dim
        r = lambda: np.random.randint(1, 3)
        # self.x = np.array([[r(), r(), r()] for _ in range(1000)])
        # self.x = np.array([[r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r()] for _ in range(1000)])
        self.x = data
        print(self.x)

    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)
        #h1 = Dense(150, activation='tanh')(inputs)
        h2 = Dense(75, activation='tanh')(inputs)
        encoded = Dense(self.encoding_dim, activation='tanh')(h2)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        h3 = Dense(75, activation='tanh')(inputs)
        # h4 = Dense(150, activation='tanh')(h3)
        num_outputs = self.x[0].shape[0]
        decoded = Dense(num_outputs)(h3)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        print(self.model.summary())
        return model

    def fit(self, batch_size=10, epochs=300):
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = './log/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tbCallBack])

    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')



def autoencoder_prediction(dataset_path, train_size, path_to_model=None, var_name='Value', plots_save_path=None,
                           verbose=False):
    """Predict the given time series with an autoencoder.

       Inputs:
           dataset_path [str]: A string path to the time series data. Data is read as a pandas Series with a DatetimeIndex and a column for numerical values.
           train_size [float]: The percentage of data to use for training, as a float (e.g., 0.66).

       Optional Inputs:
           path_to_model [str]:    Path to a file of a trained autoencoder model. When set, no training will be done because that model will be used.
           var_name [str]:         The name of the dependent variable in the time series.
                                   Default is 'Value'.
           plots_save_path [str]:  Path to save any generated plots.
           verbose [bool]:         When True, describe the time series dataset upon loading it, and pass 'verbose=True' down the chain to any other functions called during outlier detection.
                                   Default is False.

       Outputs:
            time_series [pd Series]: A pandas Series with a DatetimeIndex and a column for numerical values.
            predictions [pd Series]: A pandas Series with a DatetimeIndex and a column for the autoencoder's predictions.

       Optional Outputs:
           None

       Example:
           time_series, predictions = autoencoder_prediction(dataset_path=dataset, train_size=0.5, var_name=name,
                                                             verbose=True)
       """

    # Load the dataset
    print('Reading the dataset: ' + dataset_path)
    time_series = read_csv(dataset_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

    # Normalize data values between 0 and 1
    X = time_series.values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X).reshape(1, -1).tolist()[0]
    normalized = pd.Series(X_scaled, index=time_series.index)
    time_series = normalized

    # Shingle the dataset
    window_size = 18
    shingled_ts = shingle(time_series, window_size)

    if verbose:
        # describe the loaded dataset
        print(time_series.head())
        print(time_series.describe())
        time_series.plot(title=dataset_path + ' Dataset')
        pyplot.show()

    predictions = []

    seedy(2)
    ae = AutoEncoder(shingled_ts, encoding_dim=10)
    ae.encoder_decoder()
    ae.fit(batch_size=50, epochs=1000)
    ae.save()

    encoder = ae.encoder  # load_model(r'weights/encoder_weights.h5')
    decoder = ae.decoder  # load_model(r'weights/decoder_weights.h5')

    print('\nDone fitting the autoencoder. Testing it with an arbitrary input:')

    #inputs = np.array([[2, 1, 1]])
    inputs = np.array([[30, 31, 32, 30, 28, 31, 32, 34, 32, 32, 31, 31, 31, 30, 30, 15, 31, 31 ]])
    #inputs = np.array([[2, 1, 100, 20, 50, 3, 3, 3, 3, 2, 3, 3, 3, 0, 30, 5, 31, 31]])
    x = encoder.predict(inputs)
    y = decoder.predict(x)

    print('Input: {}'.format(inputs))
    print('Encoded: {}'.format(x))  # NOTE: if network always encodes to 0s, the relu's have "died"
    print('Decoded: {}'.format(y))

    print('\nActually predicting the time series now.')

    predictions = time_series.values[:len(time_series)%window_size].tolist()
    autoencoder_predictions = get_autoencoder_predictions(encoder, decoder, shingled_ts)  # feed data through network
    predictions = predictions + autoencoder_predictions
    predictions = pd.Series(predictions, index=time_series.index)

    #model_predict = ae.model.predict(np.array([shingled_ts[0]]))
    s = np.array([shingled_ts[-1]])
    encoded = encoder.predict(s)
    decoded = decoder.predict(encoded)
    print('Model prediction on last shingle:')
    print('Input: {}'.format(s))
    print('Encoded: {}'.format(encoded))  # NOTE: if network always encodes to 0s, the relu's have "died"
    print('Decoded: {}'.format(decoded))

    return time_series, predictions



if __name__ == "__main__":
    print('Autoencoder_Prediction.py is being run directly\n')

    ds_num = 3  # used to select dataset path and variable name together

    dataset = ['Data/BusVoltage.csv', 'Data/TotalBusCurrent.csv', 'Data/BatteryTemperature.csv',
               'Data/WheelTemperature.csv', 'Data/WheelRPM.csv'][ds_num]
    name = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Temperature (C)', 'RPM'][ds_num]

    time_series, predictions = autoencoder_prediction(dataset_path=dataset, var_name=name,
                                                      train_size=0.5, verbose=True)

    # Plot time series with autoencoder predictions
    time_series.plot(color='black', title='Time Series with Autoencoder Predictions', label=name)
    predictions.plot(color='blue', label='Predictions')
    pyplot.legend(loc='best')
    pyplot.show()


else:
    print('Autoencoder_Prediction.py is being imported into another module\n')
