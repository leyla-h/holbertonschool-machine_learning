#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    input_dims: integer, dimensions of the model input
    hidden_layers: list, number of nodes for each hidden layer
                   in the encoder (reversed for the decoder)
    latent_dims: integer, dimensions of the latent space representation

    Returns: encoder, decoder, auto
    """
    def sampling(args):
        """Reparameterization trick: sample z from mu and log_var"""
        mu, log_var = args
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mu + K.exp(log_var / 2) * epsilon

    # ------------------ Encoder ------------------
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [mu, log_var])

    encoder = keras.Model(inputs=encoder_inputs, outputs=[z, mu, log_var])

    # ------------------ Decoder ------------------
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(
        input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # ------------------ Autoencoder ------------------
    z, mu, log_var = encoder(encoder_inputs)
    auto_outputs = decoder(z)
    auto = keras.Model(inputs=encoder_inputs, outputs=auto_outputs)

    def vae_loss(inputs, outputs):
        """Reconstruction loss + KL divergence loss"""
        reconstruction_loss = keras.losses.binary_crossentropy(
            inputs, outputs) * input_dims
        kl_loss = -0.5 * K.sum(
            1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
