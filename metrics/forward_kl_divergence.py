import tensorflow as tf


class ForwardKLDivergence(tf.keras.metrics.KLDivergence):

    def __init__(self, name='forward_kl_divergence', **kwargs):
        """
        Custom Keras metric that calculates forward KL-Divergence, which is just a rename of the standard
        KLDivergence metric class in Keras.
        """

        super(ForwardKLDivergence, self).__init__(name=name, **kwargs)

        return


# Example code
# y_pred = tf.constant([[0.1, 0.9], [0, 1]])
# y_true = tf.constant([[0.8, 0.2], [0.1, 0.9]])
# sw = tf.constant([0.5, 0.5])

# fkld = ForwardKLDivergence()
# fkld.update_state(y_true, y_pred, sample_weight=sw)
# print(f'Forward KL-Divergence: {fkld.result()}')
