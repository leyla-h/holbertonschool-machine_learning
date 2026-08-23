#!/usr/bin/env python3
"""Preprocesses the Coinbase and Bitstamp BTC datasets for forecasting

Loads the two raw per-minute datasets, cleans out missing time
windows, combines them into a single continuous series, resamples the
result to an hourly resolution, splits it chronologically into
train / validation / test sets, standardizes each set (using only the
training statistics) and saves everything to a single .npz file that
forecast_btc.py can load directly.
"""
import numpy as np
import pandas as pd


def load_data(filepath):
    """Loads a raw per-minute BTC dataset from a CSV file

    filepath -- path to the CSV file

    Returns: a pandas DataFrame indexed by datetime
    """
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    return df


def clean_data(df):
    """Cleans a raw per-minute BTC DataFrame

    df -- DataFrame with columns Open, High, Low, Close,
         Volume_(BTC), Volume_(Currency), Weighted_Price. Minutes with
         no trades are recorded as rows of NaNs in this dataset.

    Returns: the cleaned DataFrame
    """
    price_cols = ['Open', 'High', 'Low', 'Close', 'Weighted_Price']
    volume_cols = ['Volume_(BTC)', 'Volume_(Currency)']

    # a minute with no trades did not move the price: carry the last
    # traded price forward instead of dropping the row, and treat the
    # untraded volume as 0
    df[price_cols] = df[price_cols].ffill()
    df[volume_cols] = df[volume_cols].fillna(0)

    # drop any leading rows that still have no price to forward fill
    df = df.dropna()

    return df


def resample_hourly(df):
    """Resamples a per-minute DataFrame into an hourly DataFrame

    df -- cleaned per-minute DataFrame

    Returns: an hourly DataFrame with Open, High, Low, Close and
    Volume_(BTC) columns
    """
    hourly = pd.DataFrame()
    hourly['Open'] = df['Open'].resample('1h').first()
    hourly['High'] = df['High'].resample('1h').max()
    hourly['Low'] = df['Low'].resample('1h').min()
    hourly['Close'] = df['Close'].resample('1h').last()
    hourly['Volume_(BTC)'] = df['Volume_(BTC)'].resample('1h').sum()

    hourly = hourly.dropna()

    return hourly


def normalize(df, mean=None, std=None):
    """Standardizes a DataFrame's columns to zero mean and unit variance

    df    -- DataFrame to normalize
    mean  -- optional Series of column means to use (e.g. computed on
            a training set); computed from df if not given
    std   -- optional Series of column standard deviations to use;
            computed from df if not given

    Returns: normalized_df, mean, std
    """
    if mean is None:
        mean = df.mean()
    if std is None:
        std = df.std()

    normalized_df = (df - mean) / std

    return normalized_df, mean, std


def preprocess(coinbase_path, bitstamp_path, output_path='btc_data.npz'):
    """Loads, cleans, resamples and normalizes the BTC datasets, then
    saves the result to a compressed numpy file

    coinbase_path  -- path to the coinbase CSV file
    bitstamp_path  -- path to the bitstamp CSV file
    output_path    -- path of the .npz file to save the result to
    """
    coinbase = clean_data(load_data(coinbase_path))
    bitstamp = clean_data(load_data(bitstamp_path))

    # the two exchanges cover the same time range: coinbase is used
    # as the primary source, and any minute missing from it entirely
    # is filled in from bitstamp
    combined = coinbase.combine_first(bitstamp)

    hourly = resample_hourly(combined)

    # split chronologically (never shuffle time series data): 80%
    # train, 10% validation, 10% test
    n = len(hourly)
    train_end = int(n * .8)
    val_end = int(n * .9)

    train_df = hourly.iloc[:train_end]
    val_df = hourly.iloc[train_end:val_end]
    test_df = hourly.iloc[val_end:]

    # normalize using only the training set's statistics, so that no
    # information about the validation/test future leaks into training
    train_norm, mean, std = normalize(train_df)
    val_norm, _, _ = normalize(val_df, mean, std)
    test_norm, _, _ = normalize(test_df, mean, std)

    np.savez(
        output_path,
        train=train_norm.values,
        val=val_norm.values,
        test=test_norm.values,
        columns=train_norm.columns.values,
        mean=mean.values,
        std=std.values,
    )

    print("Saved preprocessed data to {}".format(output_path))
    print("train: {}, val: {}, test: {}".format(
        train_norm.shape, val_norm.shape, test_norm.shape))


if __name__ == '__main__':
    preprocess('coinbase.csv', 'bitstamp.csv')
