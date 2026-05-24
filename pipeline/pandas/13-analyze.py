#!/usr/bin/env python3
"""
Module to calculate descriptive statistics on a pandas DataFrame.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except the Timestamp column.

    Args:
        df: The input pd.DataFrame

    Returns:
        A new pd.DataFrame containing the computed statistics
    """
    return df.drop(columns=['Timestamp']).describe()
