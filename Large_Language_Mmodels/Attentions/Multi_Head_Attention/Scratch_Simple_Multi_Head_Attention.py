import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from keras.api.layers import  Dense, Layer

class Simple_Multi_Head_Attention(Layer):
    def __init__(self, num_heads = 4, embed_dim = 100):
        super(Simple_Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.projection_dim = embed_dim // num_heads  # Размерность одной головы
        
        # Линейные слои для создания Q, K, V
        self.query_layer = Dense(embed_dim)
        self.key_layer = Dense(embed_dim)
        self.value_layer = Dense(embed_dim)
        
        # Линейный слой для объединения всех голов
        self.output_layer = Dense(embed_dim)

    def split_heads(self, x):
        """
        Делим размерность `embed_dim` на количество голов и перестраиваем тензор.
        Вход: (batch_size, seq_len, embed_dim)
        Выход: (batch_size, num_heads, seq_len, projection_dim)
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm = [0, 2, 1, 3])  # (batch_size, num_heads, seq_len, projection_dim)

    def call(self, inputs):
        # Ожидаем два входа: query и value (аналогично вашему Simple_Self_Attention)
        query, value = inputs
        
        # Генерируем Q, K, V
        Q = self.split_heads(self.query_layer(query))  # (batch_size, num_heads, seq_len, projection_dim)
        K = self.split_heads(self.key_layer(value))    # (batch_size, num_heads, seq_len, projection_dim)
        V = self.split_heads(self.value_layer(value))  # (batch_size, num_heads, seq_len, projection_dim)
        
        # Вычисляем скалярное произведение (scaled dot-product attention)
        attention_scores = tf.matmul(Q, K, transpose_b = True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        attention_scores /= tf.math.sqrt(d_k)  # Масштабируем
        
        # Softmax для получения весов внимания
        attention_weights = tf.nn.softmax(attention_scores, axis = -1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Умножаем веса внимания на значения V
        attention_output = tf.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len_q, projection_dim)
        
        # Объединяем головы обратно
        attention_output = tf.transpose(attention_output, perm = [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, projection_dim)
        concat_attention = tf.reshape(attention_output, (tf.shape(attention_output)[0], -1, self.embed_dim))  # (batch_size, seq_len_q, embed_dim)
        
        # Применяем линейное преобразование
        output = self.output_layer(concat_attention)  # (batch_size, seq_len_q, embed_dim)
        return output
    
# Пример использования
seq_len = 5
d_model = 100

# Пример данных: батч из 4 последовательностей длиной 10 с размерностью 32
x = tf.random.uniform((4, seq_len, d_model))
print(x.shape)
y = tf.random.uniform((4, seq_len, d_model))

# Создаем слой Self-Attention
attention_layer = Simple_Multi_Head_Attention(d_model)
output = attention_layer([x, y])

print("Размер выходного тензора:", output.shape)




