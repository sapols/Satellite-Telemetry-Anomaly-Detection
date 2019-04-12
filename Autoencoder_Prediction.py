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
import numpy as np
from tensorflow import set_random_seed


__author__ = 'Shawn Polson'
__contact__ = 'shawn.polson@colorado.edu'


def parser(x):
    new_time = ''.join(x.split('.')[0])  # remove microseconds from time data
    try:
        return datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')  # for bus voltage, battery temp, wheel temp, and wheel rpm data
    except:
        return datetime.strptime(new_time, '%Y-%m-%d')  # for total bus current data


def seedy(s):
    np.random.seed(s)
    set_random_seed(s)


class AutoEncoder:
    def __init__(self, encoding_dim=3):
        self.encoding_dim = encoding_dim
        r = lambda: np.random.randint(1, 3)                       # TODO: nope
        self.x = np.array([[r(), r(), r()] for _ in range(1000)]) # TODO: use time series instead of random values
        print(self.x)

    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(3)(inputs)
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

    if verbose:
        # describe the loaded dataset
        print(time_series.head())
        print(time_series.describe())
        time_series.plot(title=dataset_path + ' Dataset')
        pyplot.show()

    predictions = pd.Series()
    # predictions_with_dates = pd.Series(predictions.values, index=time_series.index)

    seedy(2)
    ae = AutoEncoder(encoding_dim=2)
    ae.encoder_decoder()
    ae.fit(batch_size=50, epochs=300)
    ae.save()

    encoder = load_model(r'weights/encoder_weights.h5')
    decoder = load_model(r'weights/decoder_weights.h5')

    inputs = np.array([[2, 1, 1]])
    x = encoder.predict(inputs)
    y = decoder.predict(x)

    print('Input: {}'.format(inputs))
    print('Encoded: {}'.format(x))
    print('Decoded: {}'.format(y))

    #return time_series, predictions
    return inputs, y



if __name__ == "__main__":
    print('Autoencoder_Prediction.py is being run directly\n')

    ds_num = 0  # used to select dataset path and variable name together

    dataset = ['Data/BusVoltage.csv', 'Data/TotalBusCurrent.csv', 'Data/BatteryTemperature.csv',
               'Data/WheelTemperature.csv', 'Data/WheelRPM.csv'][ds_num]
    name = ['Voltage (V)', 'Current (A)', 'Temperature (C)', 'Temperature (C)', 'RPM'][ds_num]

    time_series, predictions = autoencoder_prediction(dataset_path=dataset, var_name=name,
                                                      train_size=0.5, verbose=True)



else:
    print('Autoencoder_Prediction.py is being imported into another module\n')
