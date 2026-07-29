#!/usr/bin/env python3
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
