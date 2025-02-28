import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from keras.api.layers import *
from keras.api.models import Model
from keras.api.datasets import imdb
from keras.api.preprocessing.sequence import pad_sequences

class Scratch_T5(Model):
    def __init__(self,vocab_size, seq_len, d_model, num_heads, dff, num_layers = 4):
        super(Scratch_T5, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dff = dff
        self.num_layers = num_layers

        # Эмбеддинги
        self.embedding = Embedding(input_dim = vocab_size, output_dim = d_model)

        # Слои энкодера и декодера
        self.encoder_block = self.build_encoder_block(d_model, num_heads, dff, num_layers)
        self.decoder_block = self.build_decoder_block(d_model, num_heads, dff, num_layers)

        # Финальный слой для классификации
        self.final_dense = Dense(vocab_size, activation = 'softmax') 

    def get_positional_encoding(self, seq_len, d_model):
        pos = tf.range(seq_len, dtype = tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype = tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000., (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        angle_rads = tf.where(tf.equal(i % 2, 0), tf.sin(angle_rads), tf.cos(angle_rads))
        return angle_rads[tf.newaxis, ...]

    # Функция для создания маски для паддинга
    def create_look_ahead_mask(self, seq_len):
        # Создание маски для предотвращения использования будущей информации
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask

    def build_encoder_block(self,d_model, num_heads, dff, num_layers):
        inputs = Input(shape = (self.seq_len,))
        embedding = self.embedding(inputs)
        position_embedding = self.get_positional_encoding(self.seq_len, self.d_model)
        x = embedding + position_embedding  # Сложение эмбеддинга и позиционного кодирования
        for _ in range(self.num_layers):
            # Нормализация перед вниманием
            norm_before = LayerNormalization(epsilon = 1e-6)(x)

            # Multi-Head Attention с нормализацией до
            att = MultiHeadAttention(num_heads = self.num_heads, key_dim = self.d_model)(norm_before, norm_before)  # Внимание: query, key и value одинаковые
            
            # Нормализация после внимания
            att = LayerNormalization(epsilon = 1e-6)(att)

            # Остаточная связь: добавляем исходный вход (x) к выходу внимания
            att = att + x

            # Feed Forward Network
            ff_output = Dense(self.dff, activation = 'gelu')(att)
            ff_output = Dense(self.d_model)(ff_output)
            ff_output = Dropout(0.1)(ff_output)

            # Нормализация после Feed Forward с остаточной связью
            out = LayerNormalization(epsilon = 1e-6)(ff_output + att)

        return Model(inputs = inputs, outputs = out)
    
    def build_decoder_block(self, d_model, num_heads, dff, num_layers):
        decoder_inputs = Input(shape=(self.seq_len,))
        embedding = self.embedding(decoder_inputs)

        encoder_out = Input(shape = (self.seq_len, self.d_model))
        
        position_embedding = self.get_positional_encoding(self.seq_len, self.d_model)
        x = embedding + position_embedding  # Сложение эмбеддинга и позиционного кодирования

        # Создаем маску для декодера
        padding_mask = self.create_look_ahead_mask(self.seq_len)

        for _ in range(self.num_layers):
            # Нормализация перед вниманием
            norm_before = LayerNormalization(epsilon = 1e-6)(x)

            # Multi-Head Attention с нормализацией до
            att = MultiHeadAttention(num_heads = self.num_heads, key_dim = self.d_model)(norm_before, norm_before, attention_mask = padding_mask)  # Внимание с маской
            
            # Нормализация после внимания
            att = LayerNormalization(epsilon = 1e-6)(att)

            # Остаточная связь: добавляем исходный вход (x) к выходу внимания
            att = att + x

            # Cross-Attention: внимание с выходом кодировщика
            cross_att = MultiHeadAttention(num_heads = self.num_heads, key_dim = self.d_model)(att, encoder_out, encoder_out)
            cross_att = LayerNormalization(epsilon = 1e-6)(cross_att)
            cross_att = cross_att + att

            # Feed Forward Network
            ff_output = Dense(self.dff, activation = 'gelu')(cross_att)
            ff_output = Dense(self.d_model)(ff_output)
            ff_output = Dropout(0.1)(ff_output)

            # Нормализация после Feed Forward с остаточной связью
            out = LayerNormalization(epsilon = 1e-6)(ff_output + cross_att)

        return Model(inputs = [decoder_inputs, encoder_out], outputs = out)
    
    def call(self, inputs):
        encoder_input, decoder_input = inputs
        encoder_output = self.encoder_block(encoder_input)
        decoder_output = self.decoder_block([decoder_input, encoder_output])

        # Применяем финальный слой к выходу декодера
        final_output = self.final_dense(decoder_output)
        return final_output
    

# Загрузка и подготовка данных IMDB
max_features = 10000  # Максимальное количество слов в словаре
max_len = 100  # Максимальная длина последовательности

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)
y_train = pad_sequences(x_train, maxlen = max_len)  # Целевые последовательности
y_test = pad_sequences(x_test, maxlen = max_len)


# Гиперпараметры
vocab_size = max_features
seq_len = max_len
d_model = 128
num_heads = 4
ff_dim = 512
num_layers = 4
dropout_rate = 0.1

T5 = Scratch_T5(vocab_size = max_features, seq_len = max_len, d_model = 128, num_heads = 4, dff = 256)
# Компиляция модели
T5.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics=['accuracy']
)

T5.fit(
    [x_train, x_train], y_train,
    epochs = 5,
    batch_size = 64,
    validation_data = ([x_test, x_test], y_test)
)


# Генерация текста с нуля
generated_sequence = []
current_token = tf.constant([[1]])  # Стартовый токен (<START>)
for _ in range(50):  # Генерируем 50 токенов
    padded_input = pad_sequences([generated_sequence + [current_token.numpy()[0][0]]], maxlen = max_len)
    predictions = T5.predict(padded_input)
    next_token = tf.argmax(predictions[0, len(generated_sequence)]).numpy()
    generated_sequence.append(next_token)
    if next_token == 0:  # Если токен <PAD>, прекращаем генерацию
        break

print("Generated sequence:", generated_sequence)