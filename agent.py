import tensorflow as tf
from tensorflow_probability import distributions as tfd

class Agent():

    def __init__(self):
        pass

    def get_weights(self, features, weights):
        return tf.math.softmax(tfd.Normal(0., 1.).sample(weights.shape), axis = -1)

        