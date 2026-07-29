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

        self.model = None
        self.load_model()
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
        Creates the model used to calculate cost.
        The model uses VGG19 as a base, with AveragePooling2D
        layers replacing all MaxPooling2D layers, and outputs
        the style layers followed by the content layer.
        """
        VGG19_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        VGG19_model.save("VGG19_base_model")

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            "VGG19_base_model", custom_objects=custom_objects
        )

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers
        ]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]

        model = tf.keras.models.Model(vgg.input, outputs)
        model.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)

        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        Sets the public instance attributes:
            gram_style_features - a list of gram matrices calculated
                from the style layer outputs of the style image
            content_feature - the content layer output of the
                content image
        """
        vgg19 = tf.keras.applications.vgg19

        preprocess_style = vgg19.preprocess_input(self.style_image * 255)
        preprocess_content = vgg19.preprocess_input(self.content_image * 255)

        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        self.gram_style_features = [
            self.gram_matrix(style_output)
            for style_output in style_features
        ]
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c
                )
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image.
        """
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or \
                len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length
                )
            )

        weight = 1.0 / length
        style_cost = 0
        for style_output, gram_target in zip(
                style_outputs, self.gram_style_features):
            style_cost += weight * self.layer_style_cost(
                style_output, gram_target
            )

        return style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image.
        """
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
                content_output.shape != s:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s)
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image.
        """
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != s:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s)
            )

        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image.
        """
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != s:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s)
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                        beta1=0.9, beta2=0.99):
        """
        Generates the neural style transferred image.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image
            )

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    "Cost at iteration {}: {}, content {}, style {}".format(
                        i, J_total, J_content, J_style
                    )
                )

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                clipped = tf.clip_by_value(generated_image, 0.0, 1.0)
                generated_image.assign(clipped)

        best_image = best_image[0]

        return best_image, best_cost
