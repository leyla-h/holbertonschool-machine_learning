#!/usr/bin/env python3
"""Dataset module for machine translation"""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """
        Class constructor

        creates the instance attributes:
            data_train - the ted_hrlr_translate/pt_to_en train split
            data_valid - the ted_hrlr_translate/pt_to_en validation split
            tokenizer_pt - the Portuguese tokenizer
            tokenizer_en - the English tokenizer
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset

        data is a tf.data.Dataset whose examples are formatted as a
            tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English
                sentence

        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        pairs = [(pt.decode('utf-8'), en.decode('utf-8'))
                 for pt, en in data.as_numpy_iterator()]
        pt_sentences = [pair[0] for pair in pairs]
        en_sentences = [pair[1] for pair in pairs]

        pt_base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        en_base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        tokenizer_pt = pt_base_tokenizer.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13)
        tokenizer_en = en_base_tokenizer.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        pt is the tf.Tensor containing the Portuguese sentence
