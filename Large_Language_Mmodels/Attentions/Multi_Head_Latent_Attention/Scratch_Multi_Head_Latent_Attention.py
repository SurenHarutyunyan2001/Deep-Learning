import tensorflow as tf
from keras.api.layers import Layer, Dense, Dropout

class MultiHeadLatentAttention(Layer):
    def __init__(self, num_heads, d_model, latent_dim, dropout_rate = 0.1):
        super(MultiHeadLatentAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)

        self.latent_query = Dense(latent_dim)
        self.latent_key = Dense(latent_dim)

        self.dropout = Dropout(dropout_rate)
        self.output_dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values):
        batch_size = tf.shape(queries)[0]

        # Линейные преобразования
        queries = self.query_dense(queries)
        keys = self.key_dense(keys)
        values = self.value_dense(values)

        # Разбиение на несколько голов
        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        # Латентные преобразования
        latent_queries = self.latent_query(queries)
        latent_keys = self.latent_key(keys)

        # Масштабированное скалярное произведение внимания
        scores = tf.matmul(latent_queries, latent_keys, transpose_b = True)
        scaled_scores = scores / tf.math.sqrt(tf.cast(self.latent_dim, tf.float32))

        # Применение Dropout
        scaled_scores = self.dropout(scaled_scores)

        # Веса внимания
        attention_weights = tf.nn.softmax(scaled_scores, axis = -1)

        # Контекстный вектор
        context = tf.matmul(attention_weights, values)
        context = tf.transpose(context, perm = [0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))

        # Финальное линейное преобразование
        output = self.output_dense(context)
        return output

# Пример использования
num_heads = 8
d_model = 64
latent_dim = 32
dropout_rate = 0.1

attention_layer = MultiHeadLatentAttention(num_heads, d_model, latent_dim, dropout_rate)

# Пример входных данных (batch_size, seq_len, d_model)
queries = tf.random.normal((32, 10, d_model))
keys = tf.random.normal((32, 10, d_model))
values = tf.random.normal((32, 10, d_model))

output = attention_layer(queries, keys, values)
print(output.shape)  # (32, 10, 64)
