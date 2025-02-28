import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense
import numpy as np

class Simple_Self_Attention(Layer):
    def __init__(self, d_k, **kwargs):
        super(Simple_Self_Attention, self).__init__(**kwargs)
        self.d_k = d_k  # Размерность ключа и запроса

    def build(self, input_shape):
        # Полносвязные слои для создания запросов (Q), ключей (K) и значений (V)
        self.query_layer = Dense(self.d_k)
        self.key_layer = Dense(self.d_k)
        self.value_layer = Dense(self.d_k)
        super(Simple_Self_Attention, self).build(input_shape)

    def call(self, inputs):
        query, value = inputs

        # Преобразуем входы в запросы (Q), ключи (K) и значения (V) с помощью слоев Dense
        Q = self.query_layer(query)  # (batch_size, seq_len, d_k)
        K = self.key_layer(value)    # (batch_size, seq_len, d_k)
        V = self.value_layer(value)  # (batch_size, seq_len, d_k)

        # Вычисляем скалярное произведение между запросом и ключом
        attention_scores = tf.matmul(Q, K, transpose_b = True)  # (batch_size, seq_len, seq_len)

        # Нормализация внимания с помощью softmax
        attention_weights = tf.nn.softmax(attention_scores, axis = -1)

        # Умножаем веса внимания на значения (V)
        output = tf.matmul(attention_weights, V)  # (batch_size, seq_len, d_k)

        return output

# Пример использования
seq_len = 5
d_model = 10

# Пример данных: батч из 4 последовательностей длиной 5 с размерностью 10
x = tf.random.uniform((4, seq_len, d_model))
y = tf.random.uniform((4, seq_len, d_model))

# Создаем слой Attention
attention_layer = Simple_Self_Attention(40)
output = attention_layer([x, y])

print("Размер выходного тензора:", output.shape)
