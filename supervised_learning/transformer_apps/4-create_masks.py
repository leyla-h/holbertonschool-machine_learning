#!/usr/bin/env python3
"""Create Masks for training/validation"""
import tensorflow as tf


def create_masks(inputs, target):
    """Creates all masks for training/validation

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) that
            contains the input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out) that
            contains the target sentence

    Returns:
        encoder_mask, combined_mask, decoder_mask
            encoder_mask: the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) to be applied in the
                encoder
            combined_mask: the tf.Tensor of shape
                (batch_size, 1, seq_len_out, seq_len_out) used in the
                1st attention block in the decoder to pad and mask
                future tokens in the input received by the decoder.
                It takes the maximum between a look ahead mask and
                the decoder target padding mask.
            decoder_mask: the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in) used in the 2nd
                attention block in the decoder
    """
    seq_len_out = target.shape[1]

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
