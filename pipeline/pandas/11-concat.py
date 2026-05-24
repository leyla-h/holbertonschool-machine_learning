#!/usr/bin/env python3
"""
Module to slice and concatenate two pandas DataFrames with unique keys.
"""
import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes both dataframes, filters df2 up to a specific timestamp,
    and stacks them with labels 'bitstamp' and 'coinbase'.

    Args:
        df1: The coinbase pd.DataFrame
        df2: The bitstamp pd.DataFrame

    Returns:
        The concatenated pd.DataFrame
    """
    df1_idx = index(df1)
    df2_idx = index(df2)

    df2_filtered = df2_idx.loc[:1417411920]

    return pd.concat([df2_filtered, df1_idx], keys=['bitstamp', 'coinbase'])
