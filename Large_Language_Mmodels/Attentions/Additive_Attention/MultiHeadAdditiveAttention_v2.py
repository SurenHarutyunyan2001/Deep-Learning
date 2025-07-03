import tensorflow as tf
from keras.api.layers import Layer, Dense

class AdditiveAttention(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.W_q = self.add_weight(shape = (units, units), initializer = "random_normal", trainable = True)
        self.W_k = self.add_weight(shape = (units, units), initializer = "random_normal", trainable = True)
        self.v_a = self.add_weight(shape = (units,), initializer = "random_normal", trainable = True)
        self.b = self.add_weight(shape = (units,), initializer = "zeros", trainable = True)

    def call(self, inputs):
        query, key = inputs
        q = tf.matmul(query, self.W_q)  # (B, T, D)
        k = tf.matmul(key, self.W_k)    # (B, T, D)

        score_input = tf.expand_dims(q, 2) + tf.expand_dims(k, 1) + self.b  # (B, T, T, D)
        score = tf.tanh(score_input)
        score = tf.reduce_sum(score * self.v_a, axis = -1)  # (B, T, T)

        attention_weights = tf.nn.softmax(score, axis = -1)
        output = tf.matmul(attention_weights, key)  # (B, T, D)
        return output

class MultiHeadAdditiveAttentionLayer(Layer):
    def __init__(self, units, num_heads):
        super().__init__()
        self.units = units
        self.num_heads = num_heads
        assert units % num_heads == 0
        self.depth = units // num_heads

        self.query_dense = Dense(units)
        self.key_dense = Dense(units)
        self.value_dense = Dense(units)
        self.attn_heads = [AdditiveAttention(self.depth) for _ in range(num_heads)]
        self.out_dense = Dense(units)

    def split_heads(self, x):
        # x: (B, T, units) → (B, num_heads, T, depth)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def merge_heads(self, x):
        # x: (B, num_heads, T, depth) → (B, T, units)
        x = tf.transpose(x, perm = [0, 2, 1, 3])
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size, -1, self.units))

    def call(self, inputs, mask = None):
        query, key, value = inputs 

        Q = self.query_dense(query)
        K = self.key_dense(key)
        V = self.value_dense(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        outputs = []
        for i in range(self.num_heads):
            head_output = self.attn_heads[i]([Q[:, i], K[:, i]])  # (B, T, depth)
            outputs.append(head_output)

        concat = tf.stack(outputs, axis = 1)  # (B, H, T, D)
        merged = self.merge_heads(concat)     # (B, T, units)
        return self.out_dense(merged)

B, T, D = 4, 5, 12
query = tf.random.uniform((B, T, D))
key = tf.random.uniform((B, T, D))
value = tf.random.uniform((B, T, D))

attention_layer = MultiHeadAdditiveAttentionLayer(units = 12, num_heads = 2)
output = attention_layer([query, key, value])
print("Output shape:", output.shape)
