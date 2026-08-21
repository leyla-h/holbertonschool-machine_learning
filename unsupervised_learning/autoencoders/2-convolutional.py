#!/usr/bin/env python3
"""Module for creating a convolutional autoencoder model using TensorFlow/Keras."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder with an encoder, decoder, and full model.
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
    reversed_filters = reversed(filters)
    # All decoder conv layers except the last two
    num_layers = len(filters)
    for i, f in enumerate(reversed_filters):
        if i < num_layers - 2:
            x = keras.layers.Conv2D(
                f, (3, 3), activation='relu', padding='same'
            )(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
        elif i == num_layers - 2:
            # Second to last convolution should use valid padding
            x = keras.layers.Conv2D(
                f, (3, 3), activation='relu', padding='valid'
            )(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
        else:
            # Last convolution: same number of filters as channels in input_dims, sigmoid, no upsampling
            channels = input_dims[-1]
            x = keras.layers.Conv2D(
                channels, (3, 3), activation='sigmoid', padding='same'
            )(x)

    output_img = x
    decoder = keras.Model(latent_input, output_img, name='decoder')

    # --- Full Autoencoder ---
    auto_output = decoder(encoder(input_img))
    auto = keras.Model(input_img, auto_output, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
