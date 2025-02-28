import sys
sys.stdout.reconfigure(encoding = 'utf-8')


import tensorflow as tf
from keras.api.layers import  Dense, Layer


class Simple_Self_Attention(Layer):
    def __init__(self, units):
        super(Simple_Self_Attention, self).__init__()
        self.units = units
        
        # Определяем линейные слои для создания Q, K, V
        self.query_layer = Dense(units)
        self.key_layer = Dense(units)
        self.value_layer = Dense(units)
        
    def call(self, inputs):
        # Создаем запросы, ключи и значения
        query, value = inputs
        Q = self.query_layer(query)  # (batch_size, seq_len, units)
        K = self.key_layer(value)    # (batch_size, seq_len, units)
        V = self.value_layer(value)  # (batch_size, seq_len, units)
        
        # Вычисляем скалярное произведение между запросами и ключами
        attention_scores = tf.matmul(Q, K, transpose_b = True)  # (batch_size, seq_len, seq_len)
        
        
        # Масштабируем для стабильности
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        attention_scores = attention_scores / tf.sqrt(d_k)
        
        # Применяем softmax для получения весов внимания
        attention_weights = tf.nn.softmax(attention_scores, axis = -1)  # (batch_size, seq_len, seq_len)
        
        # Взвешиваем значения V с учетом весов внимания
        attention_output = tf.matmul(attention_weights, V)  # (batch_size, seq_len, units)
        
        return attention_output
    
# Пример использования
seq_len = 5
d_model = 10

# Пример данных: батч из 4 последовательностей длиной 10 с размерностью 32
x = tf.random.uniform((4, seq_len, d_model))
y = tf.random.uniform((4, seq_len, d_model))

# Создаем слой Self-Attention
attention_layer = Simple_Self_Attention(d_model)
output = attention_layer([x,y])

print("Размер выходного тензора:", output.shape)

