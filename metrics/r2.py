import tensorflow as tf


class R2(tf.keras.metrics.Metric):

    def __init__(self, name='r2', **kwargs):

        """
        Custom Keras metric that calculates the R-squared (R2) incrementally based on the West weighted incremental
        variance algorithm.
        """

        super(R2, self).__init__(name=name, **kwargs)

        self.sample_weights = self.add_weight(name='count', initializer='zeros')
        self.mean = self.add_weight(name='mean', initializer='zeros')
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')

        return

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Cast input tensors
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)

        # Calculate running weights
        sw_values = tf.shape(y_true)[0] if sample_weight is None else sample_weight
        sw_values = tf.reduce_sum(tf.cast(sw_values, tf.float32))
        self.sample_weights.assign_add(sw_values)

        # Incremental R2 calculation based on West weighted incremental variance
        previous_mean = tf.identity(self.mean)

        # Calculate running average
        mean_values = (y_true - self.mean)
        mean_values = mean_values if sample_weight is None else mean_values * sample_weight
        mean_values = tf.reduce_sum(mean_values) / self.count
        self.mean.assign_add(mean_values)

        # Calculate running total variance (SST)
        tot_values = (y_true - previous_mean) * (y_true - self.mean)
        tot_values = tot_values if sample_weight is None else tot_values * sample_weight
        tot_values = tf.reduce_sum(tot_values)
        self.ss_tot.assign_add(tot_values)
        print(self.ss_tot)

        # Calculate running regression error (SSE)
        res_values = tf.math.square(y_true - y_pred)
        res_values = res_values if sample_weight is None else res_values * sample_weight
        res_values = tf.reduce_sum(res_values)
        self.ss_res.assign_add(res_values)

        return

    def result(self):
        return 1 - self.ss_res / (self.ss_tot + tf.keras.backend.epsilon())

# Example code
# r2 = R2()
# y_pred = tf.constant([2.601, 3.83, 5.059, 7.517])
# y_true = tf.constant([2.0, 4.0, 6.0, 7.0])
# sw = tf.constant([0.25, 0.25, 0.25, 0.25])
# r2.update_state(y_true, y_pred, sw)
# print(r2.result())
