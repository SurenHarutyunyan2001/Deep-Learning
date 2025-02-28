import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense
import numpy as np

class AdditiveAttention(Layer):
    def __init__(self, units):
        super(AdditiveAttention, self).__init__()
        self.units = units
        # Многослойный перцептрон (MLP) для вычисления весов внимания
        self.W = Dense(units)   # Преобразование для K
        self.U = Dense(units)   # Преобразование для Q
        self.v = Dense(1)       # Финальный слой для вычисления веса внимания

    def call(self, inputs):
        query, value = inputs

        # Преобразуем Q и K
        Q = self.U(query)  # Размерность (batch_size, seq_len, units)
        K = self.W(value)  # Размерность (batch_size, seq_len, units)

        # Добавляем Q и K по последнему измерению (по осям seq_len)
        score = tf.nn.tanh(Q[:, :, tf.newaxis, :] + K[:, tf.newaxis, :, :])  # Размерность (batch_size, seq_len, seq_len, units)

        # Применяем финальный слой для вычисления внимания
        attention_weights = self.v(score)  # Размерность (batch_size, seq_len, seq_len, 1)

        # Убираем дополнительную размерность
        attention_weights = tf.squeeze(attention_weights, axis=-1)  # Размерность (batch_size, seq_len, seq_len)

        # Применяем softmax для нормализации весов
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        # Взвешиваем значения
        output = tf.matmul(attention_weights, value)  # Размерность (batch_size, seq_len, units)
        return output

# Пример использования
seq_len = 5
d_model = 10

# Пример данных: батч из 4 последовательностей длиной 5 с размерностью 10
x = tf.random.uniform((4, seq_len, d_model))  # query
y = tf.random.uniform((4, seq_len, d_model))  # value

# Создаем слой AdditiveAttention
attention_layer = AdditiveAttention(d_model)
output = attention_layer([x, y])

print("Размер выходного тензора:", output.shape)
