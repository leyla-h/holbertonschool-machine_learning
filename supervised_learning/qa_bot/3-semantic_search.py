#!/usr/bin/env python3
"""Defines a function that performs semantic search on a corpus of
documents.
"""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents

    Args:
        corpus_path: the path to the corpus of reference documents on
            which to perform semantic search
        sentence: the sentence from which to perform semantic search

    Returns:
        the reference text of the document most similar to sentence
    """
    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')

    documents = [sentence]
    filenames = []

    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        filenames.append(filename)
        with open(corpus_path + '/' + filename, 'r', encoding='utf-8') as f:
            documents.append(f.read())

    embeddings = model(documents)

    correlation = np.inner(embeddings[0], embeddings[1:])
    closest = np.argmax(correlation)

    with open(corpus_path + '/' + filenames[closest],
              'r', encoding='utf-8') as f:
        return f.read()
