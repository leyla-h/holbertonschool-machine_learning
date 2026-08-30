#!/usr/bin/env python3
"""
Module that creates all masks needed for Transformer
training/validation.
"""
import tensorflow as tf


def create_masks(inputs, target):
    """Creates all masks for training/validation.

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) that
            contains the input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out) that
            contains the target sentence

    Returns:
        encoder_mask, combined_mask, decoder_mask
            encoder_mask: tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) to be applied in the
                encoder
            combined_mask: tf.Tensor of shape
                (batch_size, 1, seq_len_out, seq_len_out) used in
                the 1st attention block in the decoder to pad and
                mask future tokens in the input received by the
                decoder. It takes the maximum between a look ahead
                mask and the decoder target padding mask.
            decoder_mask: tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) used in the 2nd
                attention block in the decoder.
    """
    seq_len_out = target.shape[1]

    def padding_mask(seq):
        """Creates a padding mask for a batch of sequences.

        Args:
            seq: tf.Tensor of shape (batch_size, seq_len)

        Returns:
            tf.Tensor of shape (batch_size, 1, 1, seq_len)
        """
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def look_ahead_mask(size):
        """Creates a look ahead mask to mask future tokens.

        Args:
            size: the sequence length to mask

        Returns:
            tf.Tensor of shape (size, size)
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    encoder_mask = padding_mask(inputs)
    decoder_mask = padding_mask(inputs)

    look_ahead = look_ahead_mask(seq_len_out)
    decoder_target_padding_mask = padding_mask(target)
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead)

    return encoder_mask, combined_mask, decoder_mask
