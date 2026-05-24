#!/usr/bin/env python3
"""
Module to calculate descriptive statistics on a pandas DataFrame.
"""


def analyze(df):
    """
    Computes descriptive statistics for columns except Timestamp.

    Args:
        df: The input pd.DataFrame

    Returns:
        A new pd.DataFrame containing the computed statistics
    """
    return df.drop(columns=['Timestamp']).describe()
