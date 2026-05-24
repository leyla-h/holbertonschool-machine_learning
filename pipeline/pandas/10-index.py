#!/usr/bin/env python3
"""
Module to set a column as the index of a pandas DataFrame.
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.

    Args:
        df: The input pd.DataFrame

    Returns:
        The modified pd.DataFrame
    """
    return df.set_index('Timestamp')
