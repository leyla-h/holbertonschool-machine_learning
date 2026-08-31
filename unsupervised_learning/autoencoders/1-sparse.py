#!/usr/bin/env python3
"""Module for creating a sparse autoencoder model using TensorFlow/Keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder with an encoder, decoder, and full model.
    """
    # --- Encoder ---
    input_img = keras.Input(shape=(input_dims,))
    x = input_img
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Apply L1 regularization to the latent layer for sparsity
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.Model(input_img, latent, name='encoder')

    # --- Decoder ---
    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    output_img = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_input, output_img, name='decoder')

    # --- Full Autoencoder ---
    auto_output = decoder(encoder(input_img))
    auto = keras.Model(input_img, auto_output, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
