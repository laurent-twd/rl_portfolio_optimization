import tensorflow as tf
import json
import pandas as pd
import numpy as np
from environment import Environment
from agent import Agent
from data_utils import get_financials_from_ticker, yfinance_get_data

class Framework():

    def __init__(self, config):

        self.tickers = config['tickers']
        self.period = config['period']
        self.interval = config['interval']
        self.c_s = config['c_s']
        self.c_p = config['c_p']
        self.horizon = config['horizon']
        self.window = config['window']
        
        try:
            if config['fitted']:
                self.fitted = True
                pass
        except:
            pass

        self.agent = Agent(self.window, len(self.tickers))
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.env = Environment(
            tickers = self.tickers,
            c_s = self.c_s,
            c_p = self.c_p
        )

    def get_engineered_features(self):

        data = yfinance_get_data(self.tickers, self.period)
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

        features = tf.cast(features, 'float32')
        prices = tf.cast(prices, 'float32')

        return tf.data.Dataset.from_tensor_slices((features, prices)), len(features)

    @tf.function
    def train_step(self, features, prices):
        initial_weights = tf.zeros(features.shape[0], dtype = 'int32')
        initial_weights = tf.one_hot(initial_weights, depth = len(self.tickers) + 1)

        with tf.GradientTape() as tape:
            reward = self.env.run_episode(self.agent, features, prices, initial_weights, training = True)
            batch_loss = - tf.reduce_mean(reward)
            variables = self.agent.actor.trainable_variables
            gradient = tape.gradient(batch_loss, variables)
            self.optimizer.apply_gradients(zip(gradient, variables))
            
        return batch_loss
    
    def fit(self, batch_size = 32, epochs = 1):

        ds, buffer_size = self.get_engineered_features()
        ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size)
        for epoch in range(epochs):
            progbar = tf.keras.utils.Progbar(buffer_size)
            for features, prices in ds:
                prices, _, _ = tf.split(prices, num_or_size_splits = [len(self.tickers) for i in range(3)], axis = -1)
                loss = self.train_step(features, prices)
                values = [('Loss', loss)]
                progbar.add(prices.shape[0], values = values)

    def save_model():
        pass

with open('config.json', 'r') as f:
    config = json.load(f)

framework = Framework(config)
framework.fit(batch_size = 32, epochs = 10)
