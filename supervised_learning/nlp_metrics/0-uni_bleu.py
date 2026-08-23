#!/usr/bin/env python3
"""Defines a function that calculates the unigram BLEU score"""
from collections import Counter
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence

    Args:
        references (list): list of reference translations
            each reference translation is a list of the words in the
            translation
        sentence (list): list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    sentence_counts = Counter(sentence)
    max_ref_counts = {}

    for reference in references:
        ref_counts = Counter(reference)
        for word, count in ref_counts.items():
            max_ref_counts[word] = max(max_ref_counts.get(word, 0), count)

    clipped_count = 0
    for word, count in sentence_counts.items():
        clipped_count += min(count, max_ref_counts.get(word, 0))

    precision = clipped_count / len(sentence)

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
