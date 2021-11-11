import tensorflow as tf
import json
import pandas as pd
import numpy as np
from environment import Environment
from agent import Agent
from data_utils import get_financials_from_ticker, yfinance_get_data
from sklearn.model_selection import train_test_split

class Framework():

    def __init__(self, config, learning_rate):

        self.tickers = config['tickers']
        self.period = config['period']
        self.interval = config['interval']
        self.c_s = config['c_s']
        self.c_p = config['c_p']
        self.horizon = config['horizon']
        self.window = config['window']
        self.n_assets = len(self.tickers)
        self.learning_rate = learning_rate

        try:
            if config['fitted']:
                self.fitted = True
                pass
        except:
            pass

        self.agent = Agent(self.window, len(self.tickers))
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.env = Environment(
            tickers = self.tickers,
            c_s = self.c_s,
            c_p = self.c_p
        )

    def get_engineered_features(self, data):

        data = tf.constant(data.values)

        idx = tf.range(self.window, data.shape[0]-self.horizon)[:, tf.newaxis]
        idx = tf.tile(idx, [1, self.window + self.horizon])
        delta = tf.range(-self.window, self.horizon)[tf.newaxis, :]
        delta = tf.tile(delta, [idx.shape[0], 1])
        X = tf.gather(data, idx + delta)
        left, right = tf.split(X, num_or_size_splits = [self.window, self.horizon], axis = 1)
        prices = tf.concat([tf.gather(left, [self.window-1], axis = 1), right], axis = 1)
        
        idx = tf.range(self.horizon)
        idx = (idx[:, tf.newaxis] + idx[tf.newaxis, :])[:, :self.window]
        features = tf.gather(X, idx, axis = 1)
        shape = features.shape
        features = features / tf.expand_dims(features[:, :, -1, :], -2)
        features = tf.reshape(features, shape = [shape[0], self.n_assets, shape[1], shape[2], int(shape[-1] / self.n_assets)])

        features = tf.cast(features, 'float32')
        prices = tf.cast(prices, 'float32')

        return features, prices, len(features)

    @tf.function
    def train_step(self, features, prices):
        initial_weights = tf.zeros(features.shape[0], dtype = 'int32')
        initial_weights = tf.one_hot(initial_weights, depth = len(self.tickers) + 1)
        with tf.GradientTape() as tape:
            r_t = self.env.run_episode(self.agent, features, prices, initial_weights, training = True)
            # reward = tf.reverse(r_t, [1])
            # reward = tf.math.cumsum(reward, axis = 1)
            # G_t = tf.reverse(reward, [1])
            # G_t = tf.reduce_sum(G_t, axis = 1)
            # batch_loss = - tf.reduce_mean(G_t)

            reward = tf.reduce_mean(r_t, axis = 1)
            batch_loss = - tf.reduce_mean(reward)
            variables = self.agent.actor.trainable_variables
            gradient = tape.gradient(batch_loss, variables)
            self.optimizer.apply_gradients(zip(gradient, variables))
        
        return batch_loss
    
    def fit(self, data, batch_size = 32, epochs = 1):

        print("Loading market data...")
        features, prices, buffer_size = self.get_engineered_features(data)
        ds_train = tf.data.Dataset.from_tensor_slices((features, prices))
        ds_train = ds_train.shuffle(buffer_size)
        ds_train = ds_train.batch(batch_size)

        print("Training...")
        idx_targets = [int(prices.shape[-1] / self.n_assets)*i for i in range(self.n_assets)]
        progbar = tf.keras.utils.Progbar(epochs)
        for epoch in range(epochs):
            for features, prices in ds_train:
                prices = tf.gather(prices, idx_targets, axis = -1)
                loss = self.train_step(features, prices)
            values = [('Loss', loss)]
            progbar.add(1, values = values)
        
    def predict(self, data):

        features, prices, buffer_size = self.get_engineered_features(data)
        prices, _, _ = tf.split(prices, num_or_size_splits = [len(self.tickers) for i in range(3)], axis = -1)

        initial_weights = tf.zeros(features.shape[0], dtype = 'int32')
        initial_weights = tf.one_hot(initial_weights, depth = len(self.tickers) + 1)
        reward, weights = self.env.run_episode(self.agent, features, prices, initial_weights, training = False)

        return tf.math.exp(reward), prices, weights

    def save_model():
        pass

