# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:48:27 2022

@author: Klay
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
        
    return np.array(data), np.array(labels)

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
            
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = create_time_steps(plot_data[0].shape[0])
    
    if delta:
        future = delta
    else:
        future = 0
        
    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize = 10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
            
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    return plt

def baseline(history):
    return np.mean(history)

def plot_train_history(history):
    loss = history.history["loss"]
    val_lose = history.history["val_loss"]
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_lose, "r", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value of Loss")
    plt.legend()
    
    plt.show()

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b',
           label='True Future')
  if prediction.any():
      plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r',
               label='Predicted Future')
  plt.legend(loc='upper left')
  plt.xlabel("History")
  plt.ylabel("Normalized Score")
  plt.title("WTI & Oman")
  plt.show()

# 데이터 전처리
tf.random.set_seed(1)

df=pd.read_csv("old_data/data.csv", encoding='utf-8')

df = df[df.Dubai != '-']
df = df[df.Brent != '-']
df = df[df.WTI != '-']
df = df[df.Oman != '-']
df = df.dropna()

# 0 ~ TRAIN_SPLIT : Train / TRAIN_SPLIT ~ : Test
TRAIN_SPLIT = 1800

BATCH_SIZE = 8
BUFFER_SIZE = 10000
    
EVALUATION_INTERVAL = 100
EPOCHS = 10

# 원하는 시장 선택
# feature_considered = ["Dollar", "WTI"]
# feature_considered = ["Dollar", "Brent"]
feature_considered = ["Dollar", "Dubai", "Oman"]  # Dubai와 Oman은 시장을 공유함.

feature = df[feature_considered]
feature.index = df["Date"]
feature = feature.astype("float64")
print(feature.info())

dataset = feature.values

data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

# Normalize
dataset = (dataset - data_mean) / data_std

# feature.plot(subplots=True)

past_history = 360      # 과거 몇 일까지를 관측할 것인가
future_target = 30      # 몇 일까지 예측할 것인가
STEP = 1        # 복잡도 (클수록 단순해짐)

x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                  TRAIN_SPLIT, past_history,
                                                  future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                              TRAIN_SPLIT, None, past_history,
                                              future_target, STEP)

print("Single window of past history : {}".format(x_train_multi[0].shape))
print("Target Price to predict : {}".format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# for x, y in train_data_multi.take(1):
#   multi_step_plot(x[0], y[0], np.array([0]))

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='tanh'))
multi_step_model.add(tf.keras.layers.Dense(future_target))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

for x, y in val_data_multi.take(1):
  print (multi_step_model.predict(x).shape)
  
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=60)

plot_train_history(multi_step_history)

for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])