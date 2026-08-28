#!/usr/bin/env python3
"""Dataset class"""

import tensorflow as tf
from setup import load_pt2en


class Dataset:
    """Dataset class for transformer applications"""

    def __init__(self):
        """Class constructor"""
        self.data_train, self.data_valid, self.tokenizer_pt, \
            self.tokenizer_en = load_pt2en()
        
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method"""
        result_pt, result_en = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en

    def encode(self, pt, en):
        """Encodes a translation into tokens"""
        pt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8')) + [self.tokenizer_pt.vocab_size + 1]

        en = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy().decode('utf-8')) + [self.tokenizer_en.vocab_size + 1]

        return pt, en
