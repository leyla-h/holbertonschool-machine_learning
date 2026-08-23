#!/usr/bin/env python3
"""Defines a function that calculates the cumulative n-gram BLEU score"""
from collections import Counter
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence

    Args:
        references (list): list of reference translations
            each reference translation is a list of the words in the
            translation
        sentence (list): list containing the model proposed sentence
        n (int): size of the largest n-gram to use for evaluation

    Returns:
        the cumulative n-gram BLEU score
    """
    def get_ngrams(words, size):
        """Builds a list of n-grams (as tuples) from a list of words"""
        return [tuple(words[i:i + size])
                for i in range(len(words) - size + 1)]

    def ngram_precision(size):
        """Computes the clipped n-gram precision for the given size"""
        sentence_ngrams = get_ngrams(sentence, size)
        sentence_counts = Counter(sentence_ngrams)
        max_ref_counts = {}

        for reference in references:
            ref_counts = Counter(get_ngrams(reference, size))
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(
                    max_ref_counts.get(ngram, 0), count)

        clipped_count = 0
        for ngram, count in sentence_counts.items():
            clipped_count += min(count, max_ref_counts.get(ngram, 0))

        return clipped_count / len(sentence_ngrams)

    precisions = [ngram_precision(size) for size in range(1, n + 1)]
    geometric_mean = np.exp(np.sum(np.log(precisions)) / n)

    sentence_len = len(sentence)
    ref_lens = [len(reference) for reference in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (
            abs(ref_len - sentence_len), ref_len))

    if sentence_len >= closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    return brevity_penalty * geometric_mean
