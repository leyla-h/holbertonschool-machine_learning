#!/usr/bin/env python3
"""Defines the Dataset class used to load and prepare a machine
translation dataset for a Transformer model.
"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation

    Attributes:
        data_train: contains the ted_hrlr_translate/pt_to_en tf.data.Dataset
            train split, loaded as_supervised, tokenized
        data_valid: contains the ted_hrlr_translate/pt_to_en tf.data.Dataset
            validate split, loaded as_supervised, tokenized
        tokenizer_pt: the Portuguese tokenizer created from the
            training set
        tokenizer_en: the English tokenizer created from the
            training set
    """

    def __init__(self):
        """Class constructor"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                     split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                     split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset

        Args:
            data: a tf.data.Dataset whose examples are formatted as a
                tuple (pt, en)
                pt: the tf.Tensor containing the Portuguese sentence
                en: the tf.Tensor containing the corresponding English
                    sentence

        Returns:
            tokenizer_pt, tokenizer_en
                tokenizer_pt: the Portuguese tokenizer
                tokenizer_en: the English tokenizer
        """
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder

        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation into tokens

        Args:
            pt: the tf.Tensor containing the Portuguese sentence
            en: the tf.Tensor containing the corresponding English
                sentence

        Returns:
            pt_tokens, en_tokens
                pt_tokens: a list containing the Portuguese tokens
                en_tokens: a list containing the English tokens
        """
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [en_vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method

        Args:
            pt: the tf.Tensor containing the Portuguese sentence
            en: the tf.Tensor containing the corresponding English
                sentence

        Returns:
            pt_result, en_result: the encoded pt and en tensors, with
                their shapes set
        """
        pt_result, en_result = tf.py_function(
            func=self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
        pt_result.set_shape([None])
        en_result.set_shape([None])

        return pt_result, en_result
