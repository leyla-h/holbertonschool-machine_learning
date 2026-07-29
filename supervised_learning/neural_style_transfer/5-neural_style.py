#!/usr/bin/env python3
"""
Neural Style Transfer
"""

import tensorflow as tf


class NST:
    """
    Neural Style Transfer class
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block4_2'

    def __init__(self, style_image, content_image):
        """
        Initialize the NST class
        """
        if not isinstance(style_image, tf.Tensor) and not tf.is_tensor(style_image):
            try:
                style_image = tf.convert_to_tensor(style_image, dtype=tf.float32)
            except Exception:
                raise TypeError("style_image must be a tensor of rank 3")
        
        if style_image.shape.ndims != 3 or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a tensor of rank 3")

        if not isinstance(content_image, tf.Tensor) and not tf.is_tensor(content_image):
            try:
                content_image = tf.convert_to_tensor(content_image, dtype=tf.float32)
            except Exception:
                raise TypeError("content_image must be a tensor of rank 3")
        
        if content_image.shape.ndims != 3 or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a tensor of rank 3")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.model = self.load_model()

        # Precompute style features and gram matrices for the style image
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        style_outputs = self.model(preprocessed_style)[:-1]
        self.gram_style_features = [
            self.gram_matrix(style_output) for style_output in style_outputs
        ]

        # Precompute content features for the content image
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )
        self.content_feature = self.model(preprocessed_content)[-1]

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its maximum side is 512 pixels
        and values are normalized between 0 and 1.
        """
        if not isinstance(image, tf.Tensor) and not tf.is_tensor(image):
            try:
                image = tf.convert_to_tensor(image, dtype=tf.float32)
            except Exception:
                raise TypeError("image must be a tensor of rank 3")

        if image.shape.ndims != 3 or image.shape[-1] != 3:
            raise TypeError("image must be a tensor of rank 3")

        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)

        max_dim = 512.0
        if height > width:
            new_height = max_dim
            new_width = max_dim * width / height
        else:
            new_width = max_dim
            new_height = max_dim * height / width

        new_shape = tf.cast(
            tf.stack([new_height, new_width]), tf.int32
        )
        scaled_image = tf.image.resize(image, new_shape, method='bicubic')
        scaled_image = scaled_image / 255.0
        scaled_image = tf.clip_by_value(scaled_image, 0.0, 1.0)

        return tf.cast(scaled_image, dtype=tf.float32)

    def load_model(self):
        """
        Creates the VGG19 model that outputs style and content features
        """
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        return tf.keras.models.Model(inputs=vgg.input, outputs=model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer
        """
        if not isinstance(input_layer, tf.Tensor) and not tf.is_tensor(input_layer):
            try:
                input_layer = tf.convert_to_tensor(input_layer, dtype=tf.float32)
            except Exception:
                raise TypeError("input_layer must be a tensor of rank 3")

        if input_layer.shape.ndims != 3:
            raise TypeError("input_layer must be a tensor of rank 3")

        channels = input_layer.shape[-1]
        result = tf.linalg.einsum('bij,bkj->bk', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[0] * input_shape[1], tf.float32)
        gram = result / num_locations

        return gram

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image
        """
        if not isinstance(style_outputs, list) or len(style_outputs) != len(self.style_layers):
            raise TypeError(
                f"style_outputs must be a list with a length of {len(self.style_layers)}"
            )

        weight = 1.0 / len(self.style_layers)
        cost = tf.add_n([
            weight * tf.reduce_mean(tf.square(self.gram_matrix(style_outputs[i]) - self.gram_style_features[i]))
            for i in range(len(style_outputs))
        ])

        return cost
