import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import numpy as np
import tensorflow as tf
from keras.api.layers import Layer

class Sinusoidal_Position_Embedding(Layer):
    def __init__(self, d_model, **kwargs):
        super(Sinusoidal_Position_Embedding, self).__init__(**kwargs)
        self.d_model = d_model

    def get_angles(self, pos, i):
        angle_rates = np.power(10000, -(2 * (i // 2) / np.float32(self.d_model)))
        return pos * angle_rates

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        pos = np.arange(seq_length)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]
        angle_rads = self.get_angles(pos, i)

        # Применяем sin к четным индексам и cos к нечетным
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = tf.cast(angle_rads, dtype=tf.float32)
        return inputs + pos_encoding

# Пример использования
d_model = 512
sample_input = tf.random.uniform((1, 50, d_model))  # batch_size=1, sequence_length=50, d_model=512
pos_emb_layer = Sinusoidal_Position_Embedding(d_model)
output = pos_emb_layer(sample_input)
print(output.shape)
