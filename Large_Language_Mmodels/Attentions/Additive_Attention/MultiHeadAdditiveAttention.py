import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense

class AdditiveAttention(Layer):
    def __init__(self, units):
        super(AdditiveAttention, self).__init__()
        self.units = units

        # Веса для вычисления совместимости между запросами и ключами
        self.W_q = self.add_weight(shape = (units, units), initializer = "random_normal", trainable = True)
        self.W_k = self.add_weight(shape = (units, units), initializer = "random_normal", trainable = True)
        self.v_a = self.add_weight(shape = (units,), initializer = "random_normal", trainable = True)

        # Смещение
        self.b = self.add_weight(shape = (units,), initializer = "zeros", trainable = True)

    def call(self, inputs):
        query, key = inputs
        
        # Применяем линейные преобразования для q и k
        q_transformed = tf.matmul(query, self.W_q)  # Размер (batch_size, seq_len, units)
        k_transformed = tf.matmul(key, self.W_k)   # Размер (batch_size, seq_len, units)
        
        # Добавляем смещение
        score_input = tf.expand_dims(q_transformed, 2) + tf.expand_dims(k_transformed, 1) + self.b
        score = tf.tanh(score_input)  # Применяем активацию tanh
        
        # Применяем v_a для получения скалярных оценок совместимости
        attention_scores = tf.reduce_sum(score * self.v_a, axis = -1)  # Размер (batch_size, seq_len, seq_len)
        
        # Вычисляем веса внимания (softmax)
        attention_weights = tf.nn.softmax(attention_scores, axis = -1)

        # Взвешиваем значения (value) с помощью attention_weights
        output = tf.matmul(attention_weights, key)  # Размер (batch_size, seq_len, units)
        return output
    
class MultiHeadAdditiveAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        assert units % num_heads == 0
        self.depth = units // num_heads

        self.heads = [AdditiveAttention(self.depth) for _ in range(num_heads)]
        self.output_dense = Dense(units)

    def split_heads(self, x):
        # x: (batch_size, seq_len, units)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # (B, T, H, D)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (B, H, T, D)

    def merge_heads(self, x):
        # x: (B, H, T, D)
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, T, H, D)
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
        merged = self.merge_heads(concat)   # (B, T, units)
        return self.output_dense(merged)

    
# Тестовый пример
batch_size = 4
seq_len = 5
d_model = 12

x = tf.random.uniform((batch_size, seq_len, d_model))
y = tf.random.uniform((batch_size, seq_len, d_model))

attention_layer = MultiHeadAdditiveAttentionLayer(units = d_model, num_heads = 2)
output = attention_layer([x, y])
print("Размер выходного тензора:", output.shape)  # (4, 5, 12)

