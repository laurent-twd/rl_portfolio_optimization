import json

import tensorflow as tf
from framework import Framework
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

with open('config.json', 'r') as f:
    config = json.load(f)

batch_size = 32
epochs = 5
trader = Framework(config)
reward, prices = trader.fit(batch_size = batch_size, epochs = epochs)

sharp_ratio = tf.reduce_mean(reward - 1.) / tf.math.reduce_std(reward - 1.)
total_reward = tf.math.cumprod(reward, axis = -1).numpy()
individual_return = tf.math.cumprod(prices[1:] / prices[:-1], axis = 1)

n = np.random.randint(0, prices.shape[0])
plt.plot(total_reward[n], label = 'Policy Gradient')
plt.plot(individual_return[n][:, 0], label = trader.tickers[0])
plt.plot(individual_return[n][:, 1], label = trader.tickers[1])
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Porfolio Value: Pf / P0')
plt.show()

