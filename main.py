import json
import tensorflow as tf
from framework import Framework
from data_utils import yfinance_get_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open('config.json', 'r') as f:
    config = json.load(f)

trader = Framework(config)
data = yfinance_get_data(trader.tickers, trader.period)
limit = int(0.75 * data.shape[0])

training_data = data.iloc[:limit]
validation_data = data.iloc[limit:]

batch_size = 32
epochs = 1
trader.fit(data = training_data, batch_size = batch_size, epochs = epochs)
reward, prices, weights = trader.predict(validation_data)

sharp_ratio = tf.reduce_mean(reward - 1.) / tf.math.reduce_std(reward - 1.)
total_reward = tf.math.cumprod(reward, axis = -1).numpy()
individual_return = tf.math.cumprod(prices[1:] / prices[:-1], axis = 1)

n = np.random.randint(0, prices.shape[0])
plt.plot(total_reward[n], label = 'Policy Gradient')
plt.plot(individual_return[n][:, 0], label = trader.tickers[0])
plt.plot(individual_return[n][:, 1], label = trader.tickers[1])
plt.plot(individual_return[n][:, 2], label = trader.tickers[2])
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Porfolio Value: Pf / P0')
plt.show()

