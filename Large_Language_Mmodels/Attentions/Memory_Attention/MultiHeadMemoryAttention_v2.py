import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense

class MemoryAttention(Layer):
    def __init__(self, units, memory_size):
        super(MemoryAttention, self).__init__()
        self.units = units
        self.memory_size = memory_size

        # Линейные слои для Q, K, V
        self.query_layer = Dense(units)
        self.key_layer = Dense(units)
        self.value_layer = Dense(units)

    def build(self, input_shape):
        # input_shape[0] = (batch_size, seq_len, features)
        batch_size = input_shape[0][0]

        # Создаём постоянную переменную памяти
        self.memory = self.add_weight(
            shape=(batch_size, self.memory_size, self.units),
            initializer = 'zeros',
            trainable = False,
            name = 'memory_var'
        )

    def call(self, inputs):
        query, value = inputs

        Q = self.query_layer(query)
        K = self.key_layer(value)
        V = self.value_layer(value)

        # Обновление памяти (без переопределения self.memory)
        new_memory = tf.concat([self.memory, K], axis = 1)
        new_memory = new_memory[:, -self.memory_size:, :]
        self.memory.assign(new_memory)

        # Конкатенация памяти и текущих ключей/значений
        K_with_memory = tf.concat([self.memory, K], axis = 1)
        V_with_memory = tf.concat([self.memory, V], axis = 1)

        # Внимание: Q * K^T
        attention_scores = tf.matmul(Q, K_with_memory, transpose_b = True)
        d_k = tf.cast(tf.shape(K_with_memory)[-1], tf.float32)
        attention_scores /= tf.sqrt(d_k)

        attention_weights = tf.nn.softmax(attention_scores, axis = -1)
        attention_output = tf.matmul(attention_weights, V_with_memory)

        return attention_output



class MultiHeadMemoryAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, memory_size):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        self.memory_size = memory_size

        assert units % num_heads == 0          # Убедимся, что можно поделить на число голов
        self.depth = units // num_heads        # Глубина одной головы

        # Создаём список из attention-голов (по одной на каждую голову)
        self.heads = [MemoryAttention(self.depth, memory_size) for _ in range(num_heads)]

        # Финальное объединяющее преобразование после объединения голов
        self.output_dense = Dense(units)

    def split_heads(self, x):
        # Преобразуем x: (batch_size, seq_len, units) -> (batch_size, num_heads, seq_len, depth)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # (B, T, H, D)
        return tf.transpose(x, perm = [0, 2, 1, 3])                      # (B, H, T, D)

    def merge_heads(self, x):
        # Объединяем x: (batch_size, num_heads, seq_len, depth) -> (batch_size, seq_len, units)
        x = tf.transpose(x, perm = [0, 2, 1, 3])                           # (B, T, H, D)
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size, -1, self.units))                 # (B, T, units)

    def call(self, inputs):
        query, key = inputs  # Оба входа: (batch_size, seq_len, units)

        # Разбиваем входы на головы
        query_heads = self.split_heads(query)  # (B, H, T, D)
        key_heads = self.split_heads(key)

        outputs = []

        # Пропускаем каждую пару голов через свою attention-голову
        for i in range(self.num_heads):
            q_i = query_heads[:, i, :, :]      # Выбор i-й головы (B, T, D)
            k_i = key_heads[:, i, :, :]
            out = self.heads[i]([q_i, k_i])    # (B, T, D)
            outputs.append(out)

        # Складываем головы обратно: (B, H, T, D)
        concat = tf.stack(outputs, axis = 1)

        # Объединяем головы: (B, T, units)
        merged = self.merge_heads(concat)

        # Финальное проецирование
        return self.output_dense(merged)


# Пример входных данных
batch_size = 4
seq_len = 5
d_model = 12

# Генерация случайных входов
x = tf.random.uniform((batch_size, seq_len, d_model))
y = tf.random.uniform((batch_size, seq_len, d_model))

# Инициализация слоя внимания
attention_layer = MultiHeadMemoryAttentionLayer(units=d_model, num_heads = 1, memory_size = 3)

# Применение слоя
output = attention_layer([x, y])

# Проверка размера выхода
print("Размер выходного тензора:", output.shape)  # Ожидается: (4, 5, 12)
