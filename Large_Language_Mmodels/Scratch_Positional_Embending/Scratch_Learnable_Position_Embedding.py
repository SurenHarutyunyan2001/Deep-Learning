import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import numpy as np
import tensorflow as tf
from keras.api.layers import Layer, Embedding

class LearnablePositionEmbedding(Layer):
    def __init__(self, max_length, d_model, **kwargs):
        super(LearnablePositionEmbedding, self).__init__(**kwargs)
        self.position_embeddings = Embedding(input_dim=max_length, output_dim=d_model)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        pos_emb = self.position_embeddings(positions)
        return inputs + pos_emb


# Пример использования
max_length = 100
d_model = 512
sample_input = tf.random.uniform((1, 50, d_model))  # batch_size=1, sequence_length=50, d_model=512
learnable_pos_emb_layer = LearnablePositionEmbedding(max_length, d_model)
output_learnable = learnable_pos_emb_layer(sample_input)
print(output_learnable.shape)