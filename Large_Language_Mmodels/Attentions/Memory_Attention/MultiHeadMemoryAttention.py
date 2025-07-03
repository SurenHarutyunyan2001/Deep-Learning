import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense

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
            shape = (batch_size, self.memory_size, self.units),
            initializer = 'zeros',
            trainable = False,
            name = 'memory'
        )

    def call(self, inputs):
        # Создаем запросы, ключи и значения
        query, value = inputs
        Q = self.query_layer(query)  # (batch_size, seq_len, units)
        K = self.key_layer(value)    # (batch_size, seq_len, units)
        V = self.value_layer(value)  # (batch_size, seq_len, units)
        
        # Обновляем память, добавляя новое состояние
        new_memory = tf.concat([self.memory, K], axis = 1)
        new_memory = new_memory[:, -self.memory_size:, :]  # Ограничиваем размер памяти
        
        # Обновляем память в слое
        self.memory.assign(new_memory)
        
        # Конкатенируем текущие ключи и память
        K_with_memory = tf.concat([self.memory, K], axis = 1)
        V_with_memory = tf.concat([self.memory, V], axis = 1)
        
        # Вычисляем скалярное произведение между запросами и ключами (включая память)
        attention_scores = tf.matmul(Q, K_with_memory, transpose_b = True)  # (batch_size, seq_len, seq_len + memory_size)
        
        # Масштабируем для стабильности
        d_k = tf.cast(tf.shape(K_with_memory)[-1], tf.float32)
        attention_scores = attention_scores / tf.sqrt(d_k)
        
        # Применяем softmax для получения весов внимания
        attention_weights = tf.nn.softmax(attention_scores, axis = -1)  # (batch_size, seq_len, seq_len + memory_size)
        
        # Взвешиваем значения V с учетом весов внимания
        attention_output = tf.matmul(attention_weights, V_with_memory)  # (batch_size, seq_len, units)
        
        return attention_output
    
class MultiHeadMemoryAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, memory_size):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        self.memory_size = memory_size
        assert units % num_heads == 0
        self.depth = units // num_heads

        self.heads = [MemoryAttention(self.depth, self.memory_size) for _ in range(num_heads)]
        self.output_dense = Dense(units)

    def split_heads(self, x):
        # x: (batch_size, seq_len, units)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # (B, T, H, D)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (B, H, T, D)

    def merge_heads(self, x):
        # x: (B, H, T, D)
        x = tf.transpose(x, perm = [0, 2, 1, 3])  # (B, T, H, D)
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size, -1, self.units))  # (B, T, units)

    def call(self, inputs):
        query, key = inputs  # (B, T, units)

        # Разделение на головы
        query_heads = self.split_heads(query)  # (B, H, T, D)
        key_heads = self.split_heads(key)

        outputs = []
        for i in range(self.num_heads):
            out = self.heads[i]([query_heads[:, i], key_heads[:, i]])  # (B, T, D)
            outputs.append(out)

        # Собираем обратно
        concat = tf.stack(outputs, axis = 1)  # (B, H, T, D)
        merged = self.merge_heads(concat)     # (B, T, units)
        return self.output_dense(merged)

    
# Тестовый пример
batch_size = 4
seq_len = 5
d_model = 12

x = tf.random.uniform((batch_size, seq_len, d_model))
y = tf.random.uniform((batch_size, seq_len, d_model))

attention_layer = MultiHeadMemoryAttentionLayer(units = d_model, num_heads = 1, memory_size = 3)
output = attention_layer([x, y])
print("Размер выходного тензора:", output.shape)  # (4, 5, 12)

