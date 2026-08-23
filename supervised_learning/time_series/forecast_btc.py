#!/usr/bin/env python3
"""Creates, trains and validates an RNN model to forecast BTC price

Loads the data preprocessed by preprocess_data.py, builds sliding
24-hour windows with tf.data.Dataset, and trains a stacked LSTM model
to predict the closing price of the following hour.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


WINDOW_SIZE = 24
BATCH_SIZE = 32
EPOCHS = 20


def load_preprocessed(path='btc_data.npz'):
    """Loads the preprocessed BTC data saved by preprocess_data.py

    path -- path to the .npz file produced by preprocess_data.py

    Returns: train, val, test, columns
    train, val, test -- numpy.ndarrays of normalized hourly BTC data
    columns          -- numpy.ndarray of the column names, in order
    """
    data = np.load(path, allow_pickle=True)
    return data['train'], data['val'], data['test'], data['columns']


def make_dataset(
        array, close_index, window_size=WINDOW_SIZE,
        batch_size=BATCH_SIZE, shuffle=False):
    """Builds a tf.data.Dataset of (24-hour window, next close) pairs

    array        -- numpy.ndarray of shape (n, f) of normalized hourly
                    BTC data
    close_index  -- index of the Close column within array
    window_size  -- number of past hours used to predict the next hour
    batch_size   -- batch size of the dataset
    shuffle      -- whether to shuffle the windows

    Returns: a tf.data.Dataset yielding (X, y) batches
    X -- windows of shape (window_size, f)
    y -- the following hour's normalized Close price
    """
    features = tf.data.Dataset.from_tensor_slices(array)
    windows = features.window(
        window_size + 1, shift=1, drop_remainder=True)
    windows = windows.flat_map(lambda w: w.batch(window_size + 1))

    def split(window):
        """Splits a (window_size + 1, f) window into (X, y)"""
        return window[:-1], window[-1, close_index]

    dataset = windows.map(split)

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def build_model(window_size, num_features):
    """Builds an LSTM based model to forecast the next hour's close

    window_size  -- number of past hours in each input window
    num_features -- number of features per hour

    Returns: an uncompiled keras.Model
    """
    inputs = keras.Input(shape=(window_size, num_features))
    hidden = keras.layers.LSTM(64, return_sequences=True)(inputs)
    hidden = keras.layers.LSTM(32)(hidden)
    outputs = keras.layers.Dense(1)(hidden)

    return keras.Model(inputs, outputs, name="btc_forecaster")


if __name__ == '__main__':
    train, val, test, columns = load_preprocessed()
    close_index = list(columns).index('Close')

    train_ds = make_dataset(train, close_index, shuffle=True)
    val_ds = make_dataset(val, close_index)
    test_ds = make_dataset(test, close_index)

    model = build_model(WINDOW_SIZE, train.shape[1])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    loss, mae = model.evaluate(test_ds)
    print("Test loss (MSE): {:.6f}, Test MAE: {:.6f}".format(loss, mae))

    model.save('btc_forecaster.h5')
