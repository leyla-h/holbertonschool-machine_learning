#!/usr/bin/env python3
"""
Module to remove rows with missing values in specific columns.
"""


def prune(df):
    """
    Removes any entries where Close has NaN values.

    Args:
        df: The input pd.DataFrame

    Returns:
        The modified pd.DataFrame
    """
    return df.dropna(subset=['Close'])
