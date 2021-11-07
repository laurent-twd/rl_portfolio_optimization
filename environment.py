import tensorflow as tf
from tensorflow_probability import distributions as tfd
from agent import Agent


class Environment():

    def __init__(self, tickers, c_s, c_p, delta = 1e-3):
        self.tickers = tickers
        self.c_s = c_s
        self.c_p = c_p
        self.c = 0.5 * (self.c_s + self.c_p)
        self.delta = delta

    def get_price_relative_vector(self, v0, v1):
        
        return tf.math.divide_no_nan(v1 , v0)
    
    def get_transaction_factor(self, closing_weights, target_weights, delta):

        def func(mu):
            x = tf.expand_dims(mu, axis = -1)
            x =  closing_weights[:, 1:] - x * target_weights[:, 1:]
            x = tf.nn.relu(x)
            x = (self.c_s + self.c_p - self.c_s * self.c_p) * tf.reduce_sum(x, axis = -1)
            x = 1. - self.c_p * closing_weights[:, 0] - x
            x = 1. / (1. - self.c_p * target_weights[:, 0]) * x
            return x
        
        mu_0 = tf.math.abs(closing_weights[:, 1:] - target_weights[:, 1:])
        mu_0 = self.c * tf.reduce_sum(mu_0, axis = -1)
        mu_1 = func(mu_0)

        k = 0
        delta = 0.01
        while tf.reduce_any(tf.math.abs(mu_0 - mu_1) > delta) and k < 20:
            mu_0 = mu_1
            mu_1 = func(mu_1)
            k+=1
        
        return mu_1
    
    def test_get_transaction_factor(self):

        closing_weights = tf.math.softmax(tfd.Normal(0., 1.).sample((32, 10)), axis = -1)
        target_weights = tf.math.softmax(tfd.Normal(0., 1.).sample((32, 10)), axis = -1)
        self = Environment([], .2, .2, delta = .1)
        delta = .1
        return self.get_transaction_factor(closing_weights, target_weights, .1)

    def run_episode(self, agent, features, prices, initial_weights):

        T = features.shape[1]
        weights = initial_weights
        reward = []
        for t in range(T):
            y_t = self.get_price_relative_vector(prices[:, t, :], prices[:, t+1, :])
            y_t = tf.concat([tf.ones((features.shape[0], 1)), y_t], axis = -1)
            closing_weights = y_t * weights / tf.reduce_sum(y_t * weights, axis = -1)[:, tf.newaxis]
            cash_bias = None
            target_weights = agent.get_weights(features[:, t, :], weights, cash_bias)
            mu_t = self.get_transaction_factor(closing_weights, target_weights, self.delta)
            return_t = tf.math.log(mu_t * tf.reduce_sum(weights * y_t, axis = -1))
            reward.append(tf.expand_dims(return_t, axis = -1))
            weights = target_weights
        
        reward = tf.concat(reward, axis = -1)
        return tf.reduce_sum(reward, axis = -1)

    def test_run_episode(self):
        
        batch_size = 32
        agent = Agent()
        d = 10
        T = 60
        initial_weights = tf.math.softmax(tfd.Normal(0., 1.).sample((batch_size, d)), axis = -1)
        prices = 100. + tfd.Normal(0., 1.).sample((batch_size, T + 1, d))
        features = 100. + tfd.Normal(0., 1.).sample((batch_size, T, d)) 
        return self.run_episode(agent, features, prices, initial_weights)
        
