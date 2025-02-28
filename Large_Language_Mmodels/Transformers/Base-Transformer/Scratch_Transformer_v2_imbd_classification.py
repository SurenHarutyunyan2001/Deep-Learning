import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import *
from keras.api.models import Model
from keras.api.datasets import imdb
from keras.api.preprocessing.sequence import pad_sequences

# Загрузка и подготовка данных IMDB
max_features = 10000  # Максимальное количество слов в словаре
max_len = 100  # Максимальная длина последовательности

# Загружаем и подготавливаем данные
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Паддинг последовательностей до одинаковой длины
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Преобразование меток в форму (batch_size, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

class Transformer(Layer):
    def __init__(self, vocab_size, seq_len, d_model, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_encoding = self.get_positional_encoding(seq_len, d_model)

        self.encoder_blocks = [self.build_encoder_block(d_model, num_heads) for _ in range(num_layers)]
        self.decoder_blocks = [self.build_decoder_block(d_model, num_heads) for _ in range(num_layers)]

        self.final_dense = Dense(1, activation='sigmoid')  # Для бинарной классификации

    def get_positional_encoding(self, seq_len, d_model):
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rads = pos / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return tf.expand_dims(pos_encoding, axis=0)

    def create_look_ahead_mask(self, batch_size, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return tf.broadcast_to(mask, [batch_size, size, size])  # (batch_size, seq_len, seq_len)

    def build_encoder_block(self, d_model, num_heads):
        inputs = Input(shape=(self.seq_len, d_model))
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
        norm_attention = LayerNormalization()(inputs + attention_output)
        dense_output = Dense(d_model, activation='relu')(norm_attention)
        encoder_output = LayerNormalization()(norm_attention + dense_output)
        return Model(inputs=inputs, outputs=encoder_output)

    def build_decoder_block(self, d_model, num_heads):
        inputs = Input(shape=(self.seq_len, d_model))
        encoder_output = Input(shape=(self.seq_len, d_model))

        look_ahead_mask = Input(shape=(self.seq_len, self.seq_len))
        masked_attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs, attention_mask=look_ahead_mask)

        norm_masked_attention = LayerNormalization()(inputs + masked_attention_output)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(norm_masked_attention, encoder_output)
        norm_attention = LayerNormalization()(norm_masked_attention + attention_output)
        dense_output = Dense(d_model, activation='relu')(norm_attention)
        decoder_output = LayerNormalization()(norm_attention + dense_output)
        return Model(inputs=[inputs, encoder_output, look_ahead_mask], outputs=decoder_output)


    def call(self, inputs):
        encoder_input, decoder_input = inputs

        # Эмбеддинг и позиционное кодирование для энкодера
        encoder_embedding = self.embedding(encoder_input)
        encoder_positional = encoder_embedding + self.pos_encoding

        # Эмбеддинг и позиционное кодирование для декодера
        decoder_embedding = self.embedding(decoder_input)
        decoder_positional = decoder_embedding + self.pos_encoding

        # Пропускаем через блоки энкодера
        encoder_output = encoder_positional
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)

        # Создаем маску и пропускаем через блоки декодера
        batch_size = tf.shape(encoder_input)[0]
        look_ahead_mask = self.create_look_ahead_mask(batch_size, self.seq_len)
        decoder_output = decoder_positional
        for block in self.decoder_blocks:
            decoder_output = block([decoder_output, encoder_output, look_ahead_mask])

        # Финальный выход
        final_output = self.final_dense(decoder_output)
        return tf.reduce_mean(final_output, axis=1)

# Создание трансформера
transformer = Transformer(vocab_size=max_features, seq_len=max_len, d_model=128, num_heads=4, num_layers=4)

# Получение выходных данных
model_input_encoder = Input(shape=(max_len,), dtype=tf.int32)
model_input_decoder = Input(shape=(max_len,), dtype=tf.int32)

transformer_output = transformer([model_input_encoder, model_input_decoder])

# Создание модели
model = Model(inputs=[model_input_encoder, model_input_decoder], outputs=transformer_output)

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit([x_train, x_train], y_train, epochs=5, batch_size=32)

# Оценка модели на тестовых данных
model.evaluate([x_test, x_test], y_test)
