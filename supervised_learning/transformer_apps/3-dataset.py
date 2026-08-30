#!/usr/bin/env python3
"""
Dataset class for Transformer application with data pipeline setup.
"""

import tensorflow as tf
tensorflow_datasets = __import__('0-dataset').Dataset


class Dataset(tensorflow_datasets):
    """Sets up the data pipeline for training and validation datasets."""

    def __init__(self, batch_size, max_len):
        """Initializes the Dataset and sets up data pipelines.

        Args:
            batch_size: batch size for training/validation
            max_len: maximum number of tokens allowed per example sentence
        """
        super().__init__()

        # Filter out examples where either sentence has more than max_len tokens
        def filter_max_len(pt, en):
            return (
                tf.shape(pt)[0] <= max_len
                and tf.shape(en)[0] <= max_len
            )

        # Training data pipeline
        self.data_train = self.data_train.filter(filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        # Validation data pipeline
        self.data_valid = self.data_valid.filter(filter_max_len)
        self.data_valid = self.data_valid.padded_batch(batch_size)
