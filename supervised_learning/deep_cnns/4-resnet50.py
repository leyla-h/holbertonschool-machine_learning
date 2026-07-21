#!/usr/bin/env python3
"""
ResNet-50 Architecture
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    # Stage 1: Conv 7x7, BatchNorm, ReLU, MaxPool 3x3
    X_conv = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(X)
    X_bn = K.layers.BatchNormalization(axis=3)(X_conv)
    X_act = K.layers.Activation('relu')(X_bn)
    X_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X_act)

    # Stage 2 (Conv2 block): 1 projection, 2 identity blocks
    X_proj = projection_block(X_pool, [64, 64, 256], s=1)
    X_id1 = identity_block(X_proj, [64, 64, 256])
    X_id2 = identity_block(X_id1, [64, 64, 256])

    # Stage 3 (Conv3 block): 1 projection, 3 identity blocks
    X_proj2 = projection_block(X_id2, [128, 128, 512], s=2)
    X_id3 = identity_block(X_proj2, [128, 128, 512])
    X_id4 = identity_block(X_id3, [128, 128, 512])
    X_id5 = identity_block(X_id4, [128, 128, 512])

    # Stage 4 (Conv4 block): 1 projection, 5 identity blocks
    X_proj3 = projection_block(X_id5, [256, 256, 1024], s=2)
    X_id6 = identity_block(X_proj3, [256, 256, 1024])
    X_id7 = identity_block(X_id6, [256, 256, 1024])
    X_id8 = identity_block(X_id7, [256, 256, 1024])
    X_id9 = identity_block(X_id8, [256, 256, 1024])
    X_id10 = identity_block(X_id9, [256, 256, 1024])

    # Stage 5 (Conv5 block): 1 projection, 2 identity blocks
    X_proj4 = projection_block(X_id10, [512, 512, 2048], s=2)
    X_id11 = identity_block(X_proj4, [512, 512, 2048])
    X_id12 = identity_block(X_id11, [512, 512, 2048])

    # Average Pooling and Dense Softmax Output Layer
    X_avg = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(X_id12)

    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init
    )(X_avg)

    model = K.models.Model(inputs=X, outputs=output)
    return model
