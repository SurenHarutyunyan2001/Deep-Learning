import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import Layer, Dense

class LinearAttention(Layer):
    def __init__(self, units, n_random_features = None):
        super(LinearAttention, self).__init__()
        self.units = units
        self.n_random_features = n_random_features or units

        self.query_layer = Dense(units)
        self.key_layer = Dense(units)
        self.value_layer = Dense(units)

    def build(self, input_shape):
        d_model = input_shape[0][-1]
        self.random_weights = self.add_weight(
            shape = (d_model, self.n_random_features),
            initializer = "random_normal",
            trainable = False,
            name = "random_weights"
        )

        self.random_bias = self.add_weight(
            shape = (self.n_random_features,),
            initializer = "random_normal",
            trainable = False,
            name = "random_bias"
        )

    def kernel_feature_map(self, x):
        projected = tf.matmul(x, self.random_weights) + self.random_bias
        return tf.cos(projected)  # Фиксированная нелинейность

    def call(self, inputs):
        query, value = inputs
        Q = self.query_layer(query)
        K = self.key_layer(value)
        V = self.value_layer(value)

        Q_hat = self.kernel_feature_map(Q)
        K_hat = self.kernel_feature_map(K)

        attention_scores = tf.matmul(Q_hat, K_hat, transpose_b = True)  # (B, T, T)
        attention_probs = tf.nn.softmax(attention_scores, axis = -1)
        output = tf.matmul(attention_probs, V)
        return output


class MultiHeadLinearAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, n_random_features = None):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        self.depth = units // num_heads
        self.n_random_features = n_random_features or self.depth

        assert units % num_heads == 0, "units must be divisible by num_heads"

        self.heads = [
            LinearAttention(self.depth, self.n_random_features) for _ in range(num_heads)
        ]
        self.output_dense = Dense(units)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])         # (B, H, T, D)

    def merge_heads(self, x):
        x = tf.transpose(x, perm = [0, 2, 1, 3])            # (B, T, H, D)
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size, -1, self.units))  # (B, T, units)

    def call(self, inputs):
        query, key = inputs  # (B, T, units)
        query_heads = self.split_heads(query)
        key_heads = self.split_heads(key)
        value_heads = self.split_heads(key)  

        outputs = []
        for i in range(self.num_heads):
            out = self.heads[i]([query_heads[:, i], value_heads[:, i]])
            outputs.append(out)

        concat = tf.stack(outputs, axis = 1)  # (B, H, T, D)
        merged = self.merge_heads(concat)     # (B, T, units)
        return self.output_dense(merged)


# Тест
batch_size = 2
seq_len = 10
d_model = 32
num_heads = 4

x = tf.random.uniform((batch_size, seq_len, d_model))
y = tf.random.uniform((batch_size, seq_len, d_model))

attn = MultiHeadLinearAttentionLayer(units = d_model, num_heads = num_heads)
out = attn([x, y])

print("Выходная форма:", out.shape)  # (2, 10, 32)
