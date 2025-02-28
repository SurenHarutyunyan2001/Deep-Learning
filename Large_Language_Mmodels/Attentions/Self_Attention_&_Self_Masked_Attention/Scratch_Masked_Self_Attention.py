import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense, Softmax

class MaskedSelfAttention(Layer):
    def __init__(self, d_k):
        super(MaskedSelfAttention, self).__init__()
        self.d_k = d_k
        self.query = Dense(d_k)
        self.key = Dense(d_k)
        self.value = Dense(d_k)

    def build_mask(self, seq_len):
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask

    def call(self, inputs):
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)

        # Вычисляем внимание с маскировкой
        seq_len = tf.shape(inputs)[1]
        mask = self.build_mask(seq_len)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(self.d_k))
        masked_scores = scores + (mask * -1e9)  # Маскируем будущие токены

        attention_weights = Softmax()(masked_scores)
        output = tf.matmul(attention_weights, V)
        
        return output

# Тестирование слоя
sample_input = tf.random.uniform((2, 5, 64))
masked_attention_layer = MaskedSelfAttention(d_k=64)
output = masked_attention_layer(sample_input)
print(output.shape)  # Ожидаемый результат: (2, 5, 64)
