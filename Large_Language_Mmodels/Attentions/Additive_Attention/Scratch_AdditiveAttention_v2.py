import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense

class AdditiveAttention(Layer):
    def __init__(self, units):
        super(AdditiveAttention, self).__init__()
        self.units = units

        # Веса для вычисления совместимости между запросами и ключами
        self.W_q = self.add_weight(shape=(units, units), initializer="random_normal", trainable=True)
        self.W_k = self.add_weight(shape=(units, units), initializer="random_normal", trainable=True)
        self.v_a = self.add_weight(shape=(units,), initializer="random_normal", trainable=True)

        # Смещение
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        query, key = inputs
        
        # Применяем линейные преобразования для q и k
        q_transformed = tf.matmul(query, self.W_q)  # Размер (batch_size, seq_len, units)
        k_transformed = tf.matmul(key, self.W_k)   # Размер (batch_size, seq_len, units)
        
        # Добавляем смещение
        score_input = tf.expand_dims(q_transformed, 2) + tf.expand_dims(k_transformed, 1) + self.b
        score = tf.tanh(score_input)  # Применяем активацию tanh
        
        # Применяем v_a для получения скалярных оценок совместимости
        attention_scores = tf.reduce_sum(score * self.v_a, axis=-1)  # Размер (batch_size, seq_len, seq_len)
        
        # Вычисляем веса внимания (softmax)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Взвешиваем значения (value) с помощью attention_weights
        output = tf.matmul(attention_weights, key)  # Размер (batch_size, seq_len, units)
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
