#!/usr/bin/env python3
"""
Module to extract specific data from a DataFrame into a numpy array.
"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns from a DataFrame
    and converts them into a numpy.ndarray.

    Args:
        df: pd.DataFrame containing columns named High and Close

    Returns:
        The newly created numpy.ndarray
    """
    return df[['High', 'Close']].tail(10).to_numpy()
