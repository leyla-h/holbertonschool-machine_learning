#!/usr/bin/env python3
"""Defines a function that creates a TF-IDF embedding matrix"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix

    Args:
        sentences (list): list of sentences to analyze
        vocab (list): list of the vocabulary words to use for the
            analysis. If None, all words within sentences are used

    Returns:
        embeddings, features
            embeddings is a numpy.ndarray of shape (s, f) containing
                the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features is a list of the features used for embeddings
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
