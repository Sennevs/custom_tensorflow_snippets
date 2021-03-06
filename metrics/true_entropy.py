import tensorflow as tf


class TrueEntropy(tf.keras.metrics.CategoricalCrossentropy):

    def __init__(self, name='true_entropy', **kwargs):
        """
        Custom Keras metric that calculates the entropy of the true distribution. This is just the CrossEntropy metric
        class with y_true as input to both the y_true and y_pred parameters.
        """

        super(TrueEntropy, self).__init__(name=name, **kwargs)

        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(TrueEntropy, self).update_state(y_true, y_true, sample_weight=sample_weight)


# Example code
# y_pred = tf.constant([[0.1, 0.9], [0, 1]])
# y_true = tf.constant([[0.8, 0.2], [0.1, 0.9]])
# sw = tf.constant([0.5, 0.5])

# te = TrueEntropy()
# te.update_state(y_true, y_pred, sample_weight=sw)
# print(f'True entropy: {te.result()}')
