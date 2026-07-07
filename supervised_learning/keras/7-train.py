#!/usr/bin/env python3
"""
Module to train a model using mini-batch gradient descent with early stopping
and learning rate decay.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    network: the model to train
    data: numpy.ndarray of shape (m, nx) containing the input data
    labels: one-hot numpy.ndarray of shape (m, classes) containing the labels
    batch_size: the size of the batch used for mini-batch gradient descent
    epochs: the number of passes through data
    validation_data: data to validate the model with
    early_stopping: boolean that indicates whether early stopping should be used
    patience: the patience used for early stopping
    learning_rate_decay: boolean that indicates whether LR decay should be used
    alpha: initial learning rate
    decay_rate: decay rate
    verbose: boolean that determines if output should be printed
    shuffle: boolean that determines whether to shuffle the batches

    Returns: the History object generated after training the model
    """
    callbacks = []

    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(callback)

    if learning_rate_decay and validation_data:
        def scheduler(epoch, lr):
            """Learning rate decay function."""
            return alpha / (1 + decay_rate * epoch)

        lr_callback = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        )
        callbacks.append(lr_callback)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
