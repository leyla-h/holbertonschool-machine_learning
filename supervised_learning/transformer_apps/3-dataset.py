#!/usr/bin/env python3
"""
Dataset class for Transformer application with data pipeline setup.
"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a Portuguese-English translation dataset
    for machine translation, including tokenization and a
    training/validation data pipeline."""

    def __init__(self, batch_size, max_len):
        """Initializes the Dataset and sets up data pipelines.

        Args:
            batch_size: batch size for training/validation
            max_len: maximum number of tokens allowed per example
                sentence
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_len(pt, en):
            """Filters out examples where either sentence exceeds
            max_len tokens."""
            return tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )

        self.data_train = self.data_train.filter(filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.filter(filter_max_len)
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset.

        Args:
            data: a tf.data.Dataset whose examples are formatted as
                a tuple (pt, en)

        Returns:
            tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        def pt_gen():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_gen():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_gen(), vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_gen(), vocab_size=2 ** 13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation into tokens, adding start/end
        of sentence tokens.

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the English sentence

        Returns:
            pt_tokens, en_tokens
        """
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'), add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'), add_special_tokens=False)

        pt_tokens = [pt_vocab_size] + pt_tokens + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_tokens + [en_vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance
        method.

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the English sentence

        Returns:
            pt_result, en_result
        """
        pt_result, en_result = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        pt_result.set_shape([None])
        en_result.set_shape([None])
        return pt_result, en_result
