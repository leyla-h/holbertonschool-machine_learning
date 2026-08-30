#!/usr/bin/env python3
"""
Dataset class that loads and preps a dataset for machine translation.
"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a Portuguese-English translation dataset
    for machine translation."""

    def __init__(self):
        """Initializes the Dataset, creating instance attributes:

        data_train: the ted_hrlr_translate/pt_to_en train split as
            a tf.data.Dataset
        data_valid: the ted_hrlr_translate/pt_to_en validation split
            as a tf.data.Dataset
        tokenizer_pt: the Portuguese tokenizer created from the
            training set
        tokenizer_en: the English tokenizer created from the
            training set
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset.

        Args:
            data: a tf.data.Dataset whose examples are formatted as
                a tuple (pt, en)
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English
                    sentence

        Returns:
            tokenizer_pt, tokenizer_en
                tokenizer_pt: the Portuguese tokenizer
                tokenizer_en: the English tokenizer
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
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
