#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10 dataset
using transfer learning from a Keras Application.
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10
        data, where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels
        for X

    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # to fix issue with saving keras applications
    K.learning_phase = K.backend.learning_phase

    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Input layer for the raw 32x32x3 CIFAR images
    inputs = K.Input(shape=(32, 32, 3))

    # Scale the 32x32 images up to a size ResNet50 was trained on
    resize = K.layers.Lambda(
        lambda img: K.backend.resize_images(
            img, 7, 7, data_format='channels_last'
        )
    )(inputs)

    # Load ResNet50 pre-trained on imagenet, without its top layers
    base_model = K.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False

    # Build a small model that only does: input -> resize -> frozen base
    # This lets us pre-compute (once) the frozen base's output for every
    # image instead of re-running it on every epoch, which saves a lot
    # of training time.
    feature_extractor = K.Model(inputs, base_model(resize, training=False))

    print('Extracting bottleneck features (train)...')
    train_features = feature_extractor.predict(
        X_train_p, batch_size=128, verbose=1
    )
    print('Extracting bottleneck features (test)...')
    test_features = feature_extractor.predict(
        X_test_p, batch_size=128, verbose=1
    )

    # Trainable classification head, trained on the pre-computed features
    head_input = K.Input(shape=train_features.shape[1:])
    x = K.layers.Dense(256, activation='relu')(head_input)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    head = K.Model(head_input, outputs)

    head.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3, restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=2
        ),
    ]

    head.fit(
        train_features, Y_train_p,
        validation_data=(test_features, Y_test_p),
        batch_size=128,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # Stitch the frozen base and trained head into a single end-to-end
    # model so cifar10.h5 takes raw 32x32x3 images as input, matching
    # what 0-main.py expects.
    final_output = head(base_model(resize, training=False))
    final_model = K.Model(inputs, final_output)

    final_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    final_model.save('cifar10.h5')
