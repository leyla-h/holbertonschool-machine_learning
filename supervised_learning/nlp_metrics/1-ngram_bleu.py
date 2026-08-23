#!/usr/bin/env python3
"""Defines a function that calculates the n-gram BLEU score"""
from collections import Counter
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Args:
        references (list): list of reference translations
            each reference translation is a list of the words in the
            translation
        sentence (list): list containing the model proposed sentence
        n (int): size of the n-gram to use for evaluation

    Returns:
        the n-gram BLEU score
    """
    def get_ngrams(words, n):
        """Builds a list of n-grams (as tuples) from a list of words"""
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    sentence_ngrams = get_ngrams(sentence, n)
    sentence_counts = Counter(sentence_ngrams)
    max_ref_counts = {}

    for reference in references:
        ref_ngrams = get_ngrams(reference, n)
        ref_counts = Counter(ref_ngrams)
        for ngram, count in ref_counts.items():
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

    clipped_count = 0
    for ngram, count in sentence_counts.items():
        clipped_count += min(count, max_ref_counts.get(ngram, 0))

    precision = clipped_count / len(sentence_ngrams)

    sentence_len = len(sentence)
    ref_lens = [len(reference) for reference in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (
            abs(ref_len - sentence_len), ref_len))

    if sentence_len >= closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    return brevity_penalty * precision
