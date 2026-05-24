#!/usr/bin/env python3
"""
Module to sort a DataFrame by the High price column.
"""


def high(df):
    """
    Sorts a DataFrame by the High column in descending order.

    Args:
        df: The input pd.DataFrame

    Returns:
        The sorted pd.DataFrame
    """
    return df.sort_values(by='High', ascending=False)
