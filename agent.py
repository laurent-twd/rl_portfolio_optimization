import tensorflow as tf
from tensorflow_probability import distributions as tfd

class Agent():

    def __init__(self, window, n_assets):
        self.window = window
        self.n_assets = n_assets
        self.actor = self.build_actor()

    def build_actor(self):
        input_features = tf.keras.Input((self.window, 3 * self.n_assets))
        input_weights = tf.keras.Input((self.n_assets, 1))
        input_cash = tf.keras.Input((1, 1))

        x = tf.keras.layers.Conv1D(
            filters = 32 * self.n_assets,
            kernel_size = 3,
            strides = 1,
            padding = 'valid',
            activation = 'relu')(input_features)
        x = tf.keras.layers.Conv1D(
            filters = 32 * self.n_assets,
            kernel_size = 28,
            strides = 1,
            padding = 'valid',
            activation = 'relu')(x)

        x = tf.keras.layers.Reshape(target_shape = (self.n_assets, 32))(x)
        x = tf.keras.layers.Concatenate(axis = -1)([x, input_weights])
        x = tf.keras.layers.Dense(1)(x)
        x = tf.keras.layers.Concatenate(axis = 1)([x, input_cash])
        x = tf.keras.layers.Dense(1, activation = 'softmax')(x)

        return tf.keras.Model(inputs = [input_features, input_weights, input_cash], outputs = x)


    def get_weights(self, features, weights, cash_bias):
        temp = tf.math.softmax(tfd.Normal(0., 1.).sample(weights.shape), axis = -1)
        return tf.cast(temp, 'float32')


        