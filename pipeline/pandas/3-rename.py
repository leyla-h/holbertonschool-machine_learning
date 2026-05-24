#!/usr/bin/env python3
"""
Module to rename columns, convert timestamps, and filter columns.
"""
import pandas as pd


def rename(df):
    """
    Renames Timestamp to Datetime, converts values to datetime objects,
    and filters the DataFrame to show only Datetime and Close.

    Args:
        df: pd.DataFrame containing a column named Timestamp

    Returns:
        The modified pd.DataFrame containing only Datetime and Close
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})

    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    return df[['Datetime', 'Close']]
