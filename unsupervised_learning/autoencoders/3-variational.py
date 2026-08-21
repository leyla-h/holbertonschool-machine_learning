#!/usr/bin/env python3
"""Module for creating a variational autoencoder using TensorFlow/Keras"""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    input_dims: integer containing the dimensions of the model input
    hidden_layers: list containing the number of nodes for each hidden
        layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims: integer containing the dimensions of the latent space
        representation

    Returns: encoder, decoder, auto
        encoder is the encoder model, which outputs the latent
            representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    """
    def sampling(args):
        """Reparameterization trick: samples z from mean and log variance"""
        mu, log_sig = args
        batch = K.shape(mu)[0]
        dims = K.shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dims))
        return mu + K.exp(log_sig / 2) * epsilon

    # --- Encoder ---
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sig = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling)([mu, log_sig])

    encoder = keras.Model(inputs, [z, mu, log_sig], name='encoder')

    # --- Decoder ---
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # --- Full Autoencoder ---
    z, mu, log_sig = encoder(inputs)
    auto_output = decoder(z)
    auto = keras.Model(inputs, auto_output, name='autoencoder')

    def vae_loss(y_true, y_pred):
        """Computes reconstruction loss plus KL divergence"""
        reconstruction_loss = keras.losses.binary_crossentropy(
            y_true, y_pred
        )
        reconstruction_loss *= input_dims
        kl_loss = 1 + log_sig - K.square(mu) - K.exp(log_sig)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
