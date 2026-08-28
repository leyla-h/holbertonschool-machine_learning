#!/usr/bin/env python3
"""Dataset class for Transformer applications."""

import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """Dataset class that loads and tokenizes translation data."""

    def __init__(self):
        """Class constructor."""
        # Load the Portuguese-English translation dataset
        self.data_train, self.data_validate = tfds.load(
            'ted_hrlr_translate/pt_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        # Tokenize the datasets
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Update data_train and data_validate by tokenizing the examples
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_validate = self.data_validate.map(self.tf_encode)

        # Cache and prefetch for performance (standard practice for tf.data)
        self.data_train = self.data_train.cache()
        # Set buffer size or prefetch as needed

    def tokenize_dataset(self, data):
        """
        Creates subword tokenizers for our dataset.
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2**15
        )

        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Wraps the tokenizer into a tf.data.Dataset compatible method.
        """
        pt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8')
        ) + [self.tokenizer_pt.vocab_size + 1]

        en = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy().decode('utf-8')
        ) + [self.tokenizer_en.vocab_size + 1]

        return pt, en

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method.
        """
        result_pt, result_en = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
