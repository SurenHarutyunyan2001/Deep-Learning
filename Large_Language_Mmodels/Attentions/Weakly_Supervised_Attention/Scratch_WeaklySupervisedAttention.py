import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense


# Определение слоя слабого внимания
class WeaklySupervisedAttention(Layer):
    def __init__(self, units, **kwargs):
        super(WeaklySupervisedAttention, self).__init__(**kwargs)
        self.units = units
        self.query_dense = Dense(units)
        self.key_dense = Dense(units)
        self.value_dense = Dense(units)

    def call(self, inputs):
        query, value, weak_labels = inputs

        # Преобразуем запросы, ключи и значения с помощью Dense слоя
        query = self.query_dense(query)  # Преобразование запроса
        key = self.key_dense(value)      # Преобразование ключа
        value = self.value_dense(value)  # Преобразование значений

       
        adjusted_key = key + weak_labels   

        # Матричное умножение для получения внимательных весов
        scores = tf.matmul(query, adjusted_key, transpose_b=True)

        # Нормализация через Softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Применяем внимание к значению
        context_vector = tf.matmul(attention_weights, value)

        return context_vector
    












