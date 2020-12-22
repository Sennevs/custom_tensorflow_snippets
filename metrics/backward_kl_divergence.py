import tensorflow as tf


class BackwardKLDivergence(tf.keras.metrics.KLDivergence):

    def __init__(self, name='backward_kl_divergence', **kwargs):
        """
        Custom Keras metric that calculates backward KL-Divergence, which is just the KLDivergence metric class in Keras
        with the y_true and y_pred parameters switched.
        """

        super(BackwardKLDivergence, self).__init__(name=name, **kwargs)

        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(BackwardKLDivergence, self).update_state(y_pred, y_true, sample_weight=sample_weight)


# Example code
# y_pred = tf.constant([[0.1, 0.9], [0, 1]])
# y_true = tf.constant([[0.8, 0.2], [0.1, 0.9]])
# sw = tf.constant([0.5, 0.5])

# bkld = BackwardKLDivergence()
# bkld.update_state(y_true, y_pred, sample_weight=sw)
# print(f'Backward KL-Divergence: {bkld.result()}')
