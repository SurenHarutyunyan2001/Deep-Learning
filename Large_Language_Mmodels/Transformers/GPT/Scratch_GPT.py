import tensorflow as tf
from keras.api.layers import Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout, Input
from keras.api.models import Model
from keras.api.datasets import imdb
from keras.api.preprocessing.sequence import pad_sequences
import numpy as np

class GPT1(Model):
    def __init__(self, vocab_size, seq_len, d_model, num_heads, ff_dim, num_layers, dropout_rate = 0.1):
        super(GPT1, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Эмбеддинги токенов и позиций
        self.token_embedding = Embedding(input_dim = vocab_size, output_dim = d_model)
        self.position_embedding = self.build_position_embedding(seq_len, d_model)

        # Слои трансформера
        self.transformer_blocks = [
            self.build_transformer_block(d_model, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]

        # Финальный слой
        self.final_layer = Dense(vocab_size)

    def build_position_embedding(self, seq_len, d_model):
        positions = np.arange(seq_len)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates

        # Применяем sin к четным индексам и cos к нечетным
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        position_embedding = tf.convert_to_tensor(angle_rads, dtype = tf.float32)
        return Embedding(input_dim = seq_len, output_dim = d_model, weights = [position_embedding], trainable = False)

    def create_attention_mask(self, seq_len):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return tf.expand_dims(mask, 0)  # (1, seq_len, seq_len)
    
    def build_transformer_block(self, d_model, num_heads, ff_dim, dropout_rate):
        inputs = Input(shape=(self.seq_len, d_model))
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
        attention_output = Dropout(dropout_rate)(attention_output)
        norm_attention = LayerNormalization(epsilon = 1e-6)(inputs + attention_output)

        ffn_output = Dense(ff_dim, activation = 'relu')(norm_attention)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)

        outputs = LayerNormalization(epsilon = 1e-6)(norm_attention + ffn_output)
        return Model(inputs, outputs)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start = 0, limit = seq_len, delta = 1)
        position_embeddings = self.position_embedding(positions)
        token_embeddings = self.token_embedding(inputs)

        x = token_embeddings + position_embeddings

        # Создание маски на каждом шаге вызова
        mask = self.create_attention_mask(seq_len)

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.final_layer(x)
        return logits

# Загрузка и подготовка данных IMDB
max_features = 10000  # Максимальное количество слов в словаре
max_len = 100  # Максимальная длина последовательности

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)

# Гиперпараметры
vocab_size = max_features
seq_len = max_len
d_model = 128
num_heads = 4
ff_dim = 512
num_layers = 4
dropout_rate = 0.1

# Создание модели
gpt1 = GPT1(vocab_size, seq_len, d_model, num_heads, ff_dim, num_layers, dropout_rate)

# Компиляция модели
gpt1.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)

# Обучение модели
gpt1.fit(
    x_train, x_train,
    batch_size = 64,
    epochs = 5,
    validation_data = (x_test, x_test)
)

# Генерация текста с нуля
generated_sequence = []
current_token = tf.constant([[1]])  # Стартовый токен (<START>)
for _ in range(50):  # Генерируем 50 токенов
    padded_input = pad_sequences([generated_sequence + [current_token.numpy()[0][0]]], maxlen = max_len)
    predictions = gpt1.predict(padded_input)
    next_token = tf.argmax(predictions[0, len(generated_sequence)]).numpy()
    generated_sequence.append(next_token)
    if next_token == 0:  # Если токен <PAD>, прекращаем генерацию
        break

print("Generated sequence:", generated_sequence)
