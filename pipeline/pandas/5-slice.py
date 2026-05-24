#!/usr/bin/env python3
"""
Module to slice specific columns and rows from a pandas DataFrame.
"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row.

    Args:
        df: The input pd.DataFrame

    Returns:
        The sliced pd.DataFrame
    """
    columns = ['High', 'Low', 'Close', 'Volume_(BTC)']
    return df[columns].iloc[::60]
