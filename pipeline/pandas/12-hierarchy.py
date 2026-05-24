#!/usr/bin/env python3
"""
Module to rearrange MultiIndex structures and filter chronological tables.
"""
import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates tables with custom source keys, swaps index hierarchy,
    slices specific inclusive timestamps, and sorts chronologically.

    Args:
        df1: The coinbase pd.DataFrame
        df2: The bitstamp pd.DataFrame

    Returns:
        The concatenated and structured pd.DataFrame
    """
    df1_idx = index(df1)
    df2_idx = index(df2)

    df = pd.concat([df2_idx, df1_idx], keys=['bitstamp', 'coinbase'])

    df = df.swaplevel(0, 1, axis=0)

    df = df.sort_index()

    return df.loc[1417411980:1417417980]
