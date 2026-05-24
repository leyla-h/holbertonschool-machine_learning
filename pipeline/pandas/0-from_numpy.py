#!/usr/bin/env python3
"""
Module to convert a numpy array to a pandas DataFrame.
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    Args:
        array: np.ndarray from which to create the pd.DataFrame

    Returns:
        The newly created pd.DataFrame with columns labeled in alphabetical
        order and capitalized.
    """
    num_cols = array.shape[1]

    columns = [chr(i) for i in range(65, 65 + num_cols)]

    return pd.DataFrame(array, columns=columns)
