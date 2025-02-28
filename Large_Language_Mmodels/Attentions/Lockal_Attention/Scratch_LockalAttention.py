import sys
sys.stdout.reconfigure(encoding = 'utf-8')


import tensorflow as tf
from keras.api.layers import  Dense, Layer

class LocalSelfAttention(Layer):
    def __init__(self, units, window_size):
        super(LocalSelfAttention, self).__init__()
        self.units = units
        self.window_size = window_size
        
        # Определяем линейные слои для создания Q, K, V
        self.query_layer = Dense(units)
        self.key_layer = Dense(units)
        self.value_layer = Dense(units)
        
       
    def call(self, inputs):
        # Вход: (batch_size, seq_len, d_model)
        # Создаем запросы, ключи и значения
        query, value = inputs
        Q = self.query_layer(query)  # (batch_size, seq_len, units)
        K = self.key_layer(value)    # (batch_size, seq_len, units)
        V = self.value_layer(value)  # (batch_size, seq_len, units)
        batch_size, seq_len, _ = tf.shape(Q)  # (batch_size, seq_len, units)

        # Для каждого элемента последовательности вычисляем внимание только на окне
        attention_output = []
        
        for i in range(seq_len):
            # Определяем индексы для локального окна вокруг позиции i
            start_idx = max(i - self.window_size, 0)
            end_idx = min(i + self.window_size + 1, seq_len)
            
            # Вытаскиваем соответствующие части Q, K и V для текущего окна
            Q_local = Q[:, start_idx:end_idx, :]
            K_local = K[:, start_idx:end_idx, :]
            V_local = V[:, start_idx:end_idx, :]
            
            # Вычисляем скалярное произведение для локального окна
            attention_scores = tf.matmul(Q_local, K_local, transpose_b=True)  # (batch_size, window_size, window_size)
            
            # Масштабируем для стабильности
            d_k = tf.cast(tf.shape(K_local)[-1], tf.float32)
            attention_scores = attention_scores / tf.sqrt(d_k)
            
            # Применяем softmax для получения весов внимания
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (batch_size, window_size, window_size)
            
            # Взвешиваем значения V с учетом весов внимания
            attention_output_local = tf.matmul(attention_weights, V_local)  # (batch_size, window_size, units)
            
            attention_output.append(attention_output_local)

        # Объединяем выходы всех окон
        attention_output = tf.concat(attention_output, axis=1)  # (batch_size, seq_len, units)
        
        return attention_output


# Пример использования
seq_len = 5
d_model = 10

# Пример данных: батч из 4 последовательностей длиной 10 с размерностью 32
x = tf.random.uniform((4, seq_len, d_model))
y = tf.random.uniform((4, seq_len, d_model))

# Создаем слой Self-Attention
attention_layer = LocalSelfAttention(20,5)
output = attention_layer([x,y])

print("Размер выходного тензора:", output.shape)
