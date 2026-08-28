#!/usr/bin/env python3
"""Multi Head Attention module for transformers"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention"""

    def __init__(self, dm, h):
        """
        Class constructor

        dm is an integer representing the dimensionality of the model
        h is an integer representing the number of heads
        dm is divisible by h
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (h, depth) and transposes the
        result to shape (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Q is a tensor of shape (batch, seq_len_q, dk) containing the
            input to generate the query matrix
        K is a tensor of shape (batch, seq_len_v, dk) containing the
            input to generate the key matrix
        V is a tensor of shape (batch, seq_len_v, dv) containing the
            input to generate the value matrix
        mask is always None

        Returns: output, weights
            output a tensor with its last two dimensions as
                (..., seq_len_q, dm) containing the scaled dot product
                attention
            weights a tensor with its last three dimensions as
                (..., h, seq_len_q, seq_len_v) containing the attention
                weights
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                       (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, weights
