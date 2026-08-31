#!/usr/bin/env python3
"""Train a transformer for machine translation"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule for the transformer"""

    def __init__(self, dm, warmup_steps=4000):
        """Class constructor"""
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Computes the learning rate at a given step"""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, loss_object):
    """Calculates the loss, ignoring padded tokens"""
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """Calculates the accuracy, ignoring padded tokens"""
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Creates and trains a transformer model for machine translation

    Args:
        N: the number of blocks in the encoder and decoder
        dm: the dimensionality of the model
        h: the number of heads
        hidden: the number of hidden units in the fully connected
            layers
        max_len: the maximum number of tokens per sequence
        batch_size: the batch size for training
        epochs: the number of epochs to train for

    Returns:
        the trained model
    """
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden, input_vocab, target_vocab, max_len, max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, target):
        """Performs one training step"""
        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        encoder_mask, combined_mask, decoder_mask = create_masks(
            inputs, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(
                inputs, tar_inp, True, encoder_mask, combined_mask,
                decoder_mask)
            loss = loss_function(tar_real, predictions, loss_object)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, (inputs, target) in enumerate(data.data_train):
            train_step(inputs, target)

            if batch % 50 == 0:
                print(
                    'Epoch {}, Batch {}: Loss {}, Accuracy {}'.format(
                        epoch + 1, batch, train_loss.result(),
                        train_accuracy.result()))

        print(
            'Epoch {}: Loss {}, Accuracy {}'.format(
                epoch + 1, train_loss.result(), train_accuracy.result()))

    return transformer
