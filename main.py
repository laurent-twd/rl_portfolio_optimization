import json
import tensorflow as tf
from framework import Framework
from data_utils import yfinance_get_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

with open('config.json', 'r') as f:
    config = json.load(f)

data = yfinance_get_data(config['tickers'], config['period'])
limit = int(0.75 * data.shape[0])

training_data = data.iloc[:limit]
validation_data = data.iloc[limit:]


trader = Framework(config, 3e-5)
batch_size = 1
epochs = 10 # int(1e6 / (2000 / 32))
trader.fit(data = training_data, batch_size = batch_size, epochs = epochs)

reward, prices, weights = trader.predict(validation_data)

sharp_ratio = tf.reduce_mean(reward - 1.) / tf.math.reduce_std(reward - 1.)
total_reward = tf.math.cumprod(reward, axis = -1).numpy()
individual_return = tf.math.cumprod(prices[1:] / prices[:-1], axis = 1)

n = np.random.randint(0, prices.shape[0])
plt.plot(total_reward[n], label = 'Policy Gradient')
plt.plot(individual_return[n][:, 0], label = trader.tickers[0])
plt.plot(individual_return[n][:, 1], label = trader.tickers[1])
plt.plot(individual_return[n][:, -1], label = trader.tickers[-1])
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Porfolio Value: Pf / P0')
plt.show()

n = np.random.randint(0, prices.shape[0])
sns.set_theme()
plt.stackplot(range(61), weights[n, :].numpy().T, labels=['Cash'] + trader.tickers)
plt.legend(loc='upper left')
plt.show()
