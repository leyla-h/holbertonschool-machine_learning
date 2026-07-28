#!/usr/bin/env python3
"""
Initialize NST class for Neural Style Transfer.
"""

import tensorflow as tf
import numpy as np


class NST:
    """
    Performs Neural Style Transfer.
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST.
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float, np.number)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float, np.number)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        if isinstance(alpha, int) and not isinstance(alpha, bool):
            self.alpha = alpha
        else:
            self.alpha = float(alpha)

        if isinstance(beta, int) and not isinstance(beta, bool):
            self.beta = beta
        else:
            self.beta = float(beta)

        self.model = self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim

        h_new = int(round(h * scale))
        w_new = int(round(w * scale))

        image_resized = tf.image.resize(
            image,
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        image_scaled = image_resized / 255.0
        image_scaled = tf.clip_by_value(image_scaled, 0.0, 1.0)
        image_expanded = tf.expand_dims(image_scaled, axis=0)

        return image_expanded

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19
        with average pooling instead of max pooling.
        """
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        inputs = vgg.input
        x = inputs
        layer_names = self.style_layers + [self.content_layer]
        outputs = []

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)

            if layer.name in layer_names:
                outputs.append(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of an input layer.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        shape = tf.shape(input_layer)
        h = shape[1]
        w = shape[2]
        c = shape[3]

        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)
        gram = tf.expand_dims(gram, axis=0)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        """
        style_outputs = self.model(self.style_image)
        content_output = self.model(self.content_image)

        style_features = style_outputs[:-1]
        self.gram_style_features = [
            self.gram_matrix(style_feature)
            for style_feature in style_features
        ]

        self.content_feature = content_output[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[3]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )

        shape = tf.shape(style_output)
        h = tf.cast(shape[1], tf.float32)
        w = tf.cast(shape[2], tf.float32)
        c_float = tf.cast(c, tf.float32)

        gram_output = self.gram_matrix(style_output)
        cost = tf.reduce_sum(tf.square(gram_output - gram_target))
        cost = cost / (4.0 * (c_float ** 2) * ((h * w) ** 2))
        return cost
