#!/usr/bin/env python3
"""Module for creating a convolutional autoencoder using TF/Keras"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    input_dims: tuple of integers containing the dimensions of the
        model input
    filters: list containing the number of filters for each
        convolutional layer in the encoder, respectively
    latent_dims: tuple of integers containing the dimensions of the
        latent space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    # --- Encoder ---
    input_img = keras.Input(shape=input_dims)
    x = input_img
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding='same'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    latent = x
    encoder = keras.Model(input_img, latent, name='encoder')

    # --- Decoder ---
    latent_input = keras.Input(shape=latent_dims)
    x = latent_input
    reversed_filters = list(reversed(filters))
    num_layers = len(reversed_filters)
    for i, f in enumerate(reversed_filters):
        if i < num_layers - 1:
            padding = 'same'
        else:
            padding = 'valid'
        x = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding=padding
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    channels = input_dims[-1]
    output_img = keras.layers.Conv2D(
        channels, (3, 3), activation='sigmoid', padding='same'
    )(x)
    decoder = keras.Model(latent_input, output_img, name='decoder')

    # --- Full Autoencoder ---
    auto_output = decoder(encoder(input_img))
    auto = keras.Model(input_img, auto_output, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
