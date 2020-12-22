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
        mean_values = tf.reduce_sum(mean_values) / self.sample_weights
        self.mean.assign_add(mean_values)

        # Calculate running total variance (SST)
        tot_values = (y_true - previous_mean) * (y_true - self.mean)
        tot_values = tot_values if sample_weight is None else tot_values * sample_weight
        tot_values = tf.reduce_sum(tot_values)
        self.ss_tot.assign_add(tot_values)

        # Calculate running regression error (SSE)
        res_values = tf.math.square(y_true - y_pred)
        res_values = res_values if sample_weight is None else res_values * sample_weight
        res_values = tf.reduce_sum(res_values)
        self.ss_res.assign_add(res_values)

        return

    def result(self):
        return 1 - self.ss_res / (self.ss_tot + tf.keras.backend.epsilon())


class ForwardKLDivergence(tf.keras.metrics.KLDivergence):

    def __init__(self, name='forward_kl_divergence', **kwargs):
        """
        Custom Keras metric that calculates forward KL-Divergence, which is basically just a rename of the standard
        KLDivergence class in Keras.
        """

        super(ForwardKLDivergence, self).__init__(name=name, **kwargs)

        return


class BackwardKLDivergence(tf.keras.metrics.KLDivergence):

    def __init__(self, name='backward_kl_divergence', **kwargs):
        """
        Custom Keras metric that calculates backward KL-Divergence, which is basically just the KLDivergence class in Keras
        with the y_true and y_pred parameters switched.
        """

        super(BackwardKLDivergence, self).__init__(name=name, **kwargs)

        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(BackwardKLDivergence, self).update_state(y_pred, y_true, sample_weight)


class TrueEntropy(tf.keras.metrics.CategoricalCrossentropy):

    def __init__(self, name='true_entropy', **kwargs):
        """
        Custom Keras metric that calculates backward KL-Divergence, which is basically just the KLDivergence class in Keras
        with the y_true and y_pred parameters switched.
        """

        super(TrueEntropy, self).__init__(name=name, **kwargs)

        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(TrueEntropy, self).update_state(y_true, y_true, sample_weight)


class PredictedEntropy(tf.keras.metrics.CategoricalCrossentropy):

    def __init__(self, name='true_entropy', **kwargs):
        """
        Custom Keras metric that calculates backward KL-Divergence, which is basically just the KLDivergence class in Keras
        with the y_true and y_pred parameters switched.
        """

        super(PredictedEntropy, self).__init__(name=name, **kwargs)

        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(PredictedEntropy, self).update_state(y_pred, y_pred, sample_weight)


class TrueEntropyCustom(tf.keras.metrics.KLDivergence):

    def __init__(self, name='forward_kl_divergence', **kwargs):
        """
        Custom Keras metric that calculates forward KL-Divergence, which is basically just a rename of the standard
        KLDivergence class in Keras.
        """

        super(ForwardKLDivergence, self).__init__(name=name, **kwargs)

        self.entropy = self.add_weight(name='entropy', initializer='zeros')
        self.sample_weights = self.add_weight(name='sample_weights', initializer='zeros')

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

        # Calculate running entropy average
        entropy = -y_true * tf.math.log(y_true)
        entropy_values = (entropy - self.entropy)
        entropy_values = entropy_values if sample_weight is None else entropy_values * sample_weight
        entropy_values = tf.reduce_sum(entropy_values) / self.sample_weights
        self.entropy.assign_add(entropy_values)

        return

    def result(self):
        return self.entropy


class PredictedEntropyCustom(tf.keras.metrics.KLDivergence):

    def __init__(self, name='forward_kl_divergence', **kwargs):
        """
        Custom Keras metric that calculates forward KL-Divergence, which is basically just a rename of the standard
        KLDivergence class in Keras.
        """

        super(ForwardKLDivergence, self).__init__(name=name, **kwargs)

        self.entropy = self.add_weight(name='entropy', initializer='zeros')
        self.sample_weights = self.add_weight(name='sample_weights', initializer='zeros')

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

        # Calculate running entropy average
        entropy = -y_true * tf.math.log(y_true)
        entropy_values = (entropy - self.entropy)
        entropy_values = entropy_values if sample_weight is None else entropy_values * sample_weight
        entropy_values = tf.reduce_sum(entropy_values) / self.sample_weights
        self.entropy.assign_add(entropy_values)

        return

    def result(self):
        return self.entropy


