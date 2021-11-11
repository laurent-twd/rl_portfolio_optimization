import tensorflow as tf
from tensorflow_probability import distributions as tfd

class Agent():

    def __init__(self, window, n_assets):
        self.window = window
        self.n_assets = n_assets
        self.actor = self.build_actor()

    def build_actor(self):
        input_features = tf.keras.Input((self.n_assets, self.window, 3))
        input_weights = tf.keras.Input((self.n_assets, 1, 1))

        x = tf.keras.layers.Conv2D(
            filters = 2,
            kernel_size = [1,3],
            strides = [1,1],
            padding = 'valid',
            activation = 'relu')(input_features)
            
        x = tf.keras.layers.Conv2D(
            filters = 20,
            kernel_size = [1, self.window - 2],
            strides = [1, 1],
            padding = 'valid',
            activation = 'relu')(x)

        x = tf.keras.layers.Concatenate(axis = -1)([x, input_weights])

        x = tf.keras.layers.Conv2D(
            filters = 1,
            kernel_size = [1, 1],
            strides = [1, 1],
            padding = 'valid',
            activation = 'linear')(x)

        bias = tf.ones_like(input_weights)[:, 0:1, :]
        x = tf.keras.layers.Concatenate(axis = 1)([bias, x])

        return tf.keras.Model(inputs = [input_features, input_weights], outputs = x)

    def get_weights(self, features, weights, training):
        target_weights = self.actor([features, weights[:, 1:]], training = training)
        for i in range(len(target_weights.shape) - 2):
            target_weights = tf.squeeze(target_weights, axis = -1)  
        target_weights = tf.math.softmax(target_weights, axis = -1)
 
        return target_weights



# class Agent():

#     def __init__(self, window, n_assets):
#         self.window = window
#         self.n_assets = n_assets
#         self.actor = self.build_actor()

#     def build_actor(self):
#         input_features = tf.keras.Input((self.window, 3 * self.n_assets))
#         input_weights = tf.keras.Input((self.n_assets, 1))

#         x = tf.keras.layers.LSTM(self.n_assets * 11, return_sequences=True)(input_features)
#         x = tf.keras.layers.LSTM(self.n_assets * 11, return_sequences=False)(x)
#         x = tf.keras.layers.Reshape(target_shape = (self.n_assets, 11))(x)
#         x = tf.keras.layers.Concatenate(axis = -1)([x, input_weights])
#         x = tf.keras.layers.Dense(1)(x)
#         bias = tf.ones_like(input_weights)[:, 0:1, :]
#         x = tf.keras.layers.Concatenate(axis = 1)([bias, x])

#         return tf.keras.Model(inputs = [input_features, input_weights], outputs = x)

#     def get_weights(self, features, weights, training):
#         target_weights = self.actor([features, weights[:, 1:]], training = training)
#         target_weights = tf.squeeze(target_weights, axis = -1)
#         target_weights = tf.math.softmax(target_weights, axis = -1)
#         #temp = tf.math.softmax(tfd.Normal(0., 1.).sample(weights.shape), axis = -1)
#         return target_weights