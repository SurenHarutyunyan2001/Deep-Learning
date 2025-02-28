import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from keras.api.layers import Dense, Layer

class Linear_Attention(Layer):
    def __init__(self, units, n_random_features=None):
        super(Linear_Attention, self).__init__()
        self.units = units
        self.n_random_features = n_random_features if n_random_features else units

        # Определяем Dense-слои для Q, K и V
        self.query_layer = Dense(units)
        self.key_layer = Dense(units)
        self.value_layer = Dense(units)
        
        # Пока что случайные признаки не инициализируем, сделаем это в методе build

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
        """
        Приближенное представление ключей K с помощью случайных признаков.
        """
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
        attention_scores = tf.matmul(Q, K_hat, transpose_b=True)  # Примерный расчет внимания
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # Вычисление выходных значений
        output = tf.matmul(attention_probs, V)
        return output

# Пример использования
seq_len = 5
d_model = 10

# Пример данных: батч из 4 последовательностей длиной 5 с размерностью 10
x = tf.random.uniform((4, seq_len, d_model))
y = tf.random.uniform((4, seq_len, d_model))

# Создаем слой Self-Attention
attention_layer = Linear_Attention(d_model)
output = attention_layer([x, y])

print("Размер выходного тензора:", output.shape)
