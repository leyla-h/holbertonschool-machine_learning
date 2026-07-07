#!/usr/bin/env python3
"""
Module to train a model using mini-batch gradient descent with early stopping,
learning rate decay, and checkpointing the best model.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.
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

    if save_best and filepath:
        save_callback = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_best_only=True
        )
        callbacks.append(save_callback)

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
