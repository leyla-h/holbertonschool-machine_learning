#!/usr/bin/env python3
"""
Module to sort a DataFrame in reverse chronological order and transpose it.
"""


def flip_switch(df):
    """
    Sorts data in reverse chronological order and transposes the dataframe.

    Args:
        df: The input pd.DataFrame

    Returns:
        The transformed pd.DataFrame
    """
    return df.sort_index(ascending=False).T
