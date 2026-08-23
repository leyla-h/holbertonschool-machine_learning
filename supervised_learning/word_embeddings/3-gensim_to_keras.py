#!/usr/bin/env python3
"""Defines a function that converts a gensim word2vec model to keras"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a trainable keras Embedding layer

    Args:
        model: a trained gensim word2vec model

    Returns:
        the trainable keras Embedding
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        trainable=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([weights])

    return embedding_layer
