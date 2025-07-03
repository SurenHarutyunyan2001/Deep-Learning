import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense

class LinearAttention(Layer):
    def __init__(self, units, n_random_features=None):
        super(LinearAttention, self).__init__()
        self.units = units
        self.n_random_features = n_random_features if n_random_features else units

        # Определяем Dense-слои для Q, K и V
        self.query_layer = Dense(units)
        self.key_layer = Dense(units)
        self.value_layer = Dense(units)

    def build(self, input_shape):
        # Определяем размерность входа для случайных признаков
        d_model = input_shape[0][-1]  # Размерность входа (d_model)
        
        # Инициализация случайных признаков для ядра
        self.random_weights = self.add_weight(
            shape = (d_model, self.n_random_features),  # (d_model, n_random_features)
            initializer = "random_normal",
            trainable = False
        )
        self.random_bias = self.add_weight(
            shape = (self.n_random_features,),  # Размерность случайных признаков
            initializer = "random_normal",
            trainable = False
        )
        
    def kernel_feature_map(self, K):
        # Приближенное представление ключей K с помощью случайных признаков.
        
        random_features = tf.matmul(K, self.random_weights) + self.random_bias
        return tf.cos(random_features)

    def call(self, inputs):
        # Создаем запросы, ключи и значения
        query, value = inputs
        Q = self.query_layer(query)  # (batch_size, seq_len, units)
        K = self.key_layer(value)    # (batch_size, seq_len, units)
        V = self.value_layer(value)  # (batch_size, seq_len, units)

        # Приближенное представление ключей K с обучаемыми случайными признаками
        K_hat = self.kernel_feature_map(K)  # (batch_size, seq_len, n_random_features)

        # Проверка совместимости матриц для умножения
        if Q.shape[-1] != K_hat.shape[-1]:
            raise ValueError(f"Размерности несовместимы: Q имеет {Q.shape[-1]}, а K_hat имеет {K_hat.shape[-1]}")

        # Вычисление внимания
        attention_scores = tf.matmul(Q, K_hat, transpose_b = True)  # Примерный расчет внимания
        attention_probs = tf.nn.softmax(attention_scores, axis = -1)

        # Вычисление выходных значений
        output = tf.matmul(attention_probs, V)
        return output
    
class MultiHeadLinearAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        assert units % num_heads == 0
        self.depth = units // num_heads

        self.heads = [LinearAttention(self.depth) for _ in range(num_heads)]
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

attention_layer = MultiHeadLinearAttentionLayer(units = d_model, num_heads = 1)
output = attention_layer([x, y])
print("Размер выходного тензора:", output.shape)  # (4, 5, 12)

