#!/usr/bin/env python3
"""
Module to save and load Keras model configuration in JSON.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Args:
        network: The model whose configuration should be saved.
        filename: The path of the file that the configuration should be saved to.

    Returns:
        None
    """
    model_json = network.to_json()
    with open(filename, 'w') as f:
        f.write(model_json)


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file.

    Args:
        filename: The path of the file containing the model's configuration.

    Returns:
        The loaded model.
    """
    with open(filename, 'r') as f:
        model_json = f.read()
    
    network = K.models.model_from_json(model_json)
    return network
