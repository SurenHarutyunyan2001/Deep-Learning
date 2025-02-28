import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from keras.api.layers import Dense, Layer

class MemoryAttention(Layer):
    def __init__(self, units, memory_size):
        super(MemoryAttention, self).__init__()
        self.units = units
        self.memory_size = memory_size
        
        # Определяем линейные слои для создания Q, K, V
        self.query_layer = Dense(units)
        self.key_layer = Dense(units)
        self.value_layer = Dense(units)
        
        # Инициализируем память как None, чтобы она определялась при первом вызове
        self.memory = None

    def build(self, input_shape):
        # Инициализация памяти на основе размера входных данных
        batch_size = input_shape[0][0]
        self.memory = self.add_weight(
            shape=(batch_size, self.memory_size, self.units),
            initializer='zeros',
            trainable=False,
            name='memory'
        )

    def call(self, inputs):
        # Создаем запросы, ключи и значения
        query, value = inputs
        Q = self.query_layer(query)  # (batch_size, seq_len, units)
        K = self.key_layer(value)    # (batch_size, seq_len, units)
        V = self.value_layer(value)  # (batch_size, seq_len, units)
        
        # Обновляем память, добавляя новое состояние
        new_memory = tf.concat([self.memory, K], axis=1)
        new_memory = new_memory[:, -self.memory_size:, :]  # Ограничиваем размер памяти
        
        # Обновляем память в слое
        self.memory.assign(new_memory)
        
        # Конкатенируем текущие ключи и память
        K_with_memory = tf.concat([self.memory, K], axis=1)
        V_with_memory = tf.concat([self.memory, V], axis=1)
        
        # Вычисляем скалярное произведение между запросами и ключами (включая память)
        attention_scores = tf.matmul(Q, K_with_memory, transpose_b=True)  # (batch_size, seq_len, seq_len + memory_size)
        
        # Масштабируем для стабильности
        d_k = tf.cast(tf.shape(K_with_memory)[-1], tf.float32)
        attention_scores = attention_scores / tf.sqrt(d_k)
        
        # Применяем softmax для получения весов внимания
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (batch_size, seq_len, seq_len + memory_size)
        
        # Взвешиваем значения V с учетом весов внимания
        attention_output = tf.matmul(attention_weights, V_with_memory)  # (batch_size, seq_len, units)
        
        return attention_output

# Пример использования
seq_len = 5
d_model = 10
memory_size = 3

# Пример данных: батч из 4 последовательностей длиной seq_len с размерностью d_model
x = tf.random.uniform((4, seq_len, d_model))
y = tf.random.uniform((4, seq_len, d_model))

# Создаем слой Attention with Memory
attention_layer = MemoryAttention(units=d_model, memory_size=memory_size)
output = attention_layer([x, y])

print("Размер выходного тензора:", output.shape)
