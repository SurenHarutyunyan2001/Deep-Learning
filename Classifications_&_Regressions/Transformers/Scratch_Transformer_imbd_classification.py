import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from keras.api.layers import *
from keras.api.models import Model
from keras.api.datasets import imdb
from keras.api.preprocessing.sequence import pad_sequences

# Загрузка и подготовка данных IMDB
max_features = 10000  # Максимальное количество слов в словаре
max_len = 100  # Максимальная количество слов

# Загружаем и подготавливаем данные
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

# Паддинг последовательностей до одинаковой длины
x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

class Transformer(Model):
    def __init__(self, vocab_size, seq_len, d_model, num_heads):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Эмбеддинги
        self.embedding = Embedding(input_dim = vocab_size, output_dim = d_model)

        # Слои энкодера и декодера
        self.encoder_block = self.build_encoder_block(d_model, num_heads)
        self.decoder_block = self.build_decoder_block(d_model, num_heads)

        # Финальный слой для классификации
        self.final_dense = Dense(vocab_size, activation = 'softmax') 

    def create_look_ahead_mask(self, size):
        mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]  # Добавляем измерения для батча и головы
    
    def build_encoder_block(self, d_model, num_heads):
        inputs = Input(shape = (self.seq_len,))
        embedding_output = self.embedding(inputs)
        attention_output = MultiHeadAttention(num_heads = num_heads, key_dim = d_model)(embedding_output, embedding_output)
        norm_attention = LayerNormalization()(embedding_output + attention_output)
        dense_output = Dense(d_model, activation = 'relu')(norm_attention)
        encoder_output = LayerNormalization()(norm_attention + dense_output)
        return Model(inputs = inputs, outputs = encoder_output)

    def build_decoder_block(self, d_model, num_heads):
        inputs = Input(shape = (self.seq_len,))
        encoder_output = Input(shape = (self.seq_len, self.d_model))
        decoder_embedded = self.embedding(inputs)

        # Создаем маску для предотвращения утечки информации из будущих токенов
        look_ahead_mask = self.create_look_ahead_mask(self.seq_len)
        masked_attention_output = MultiHeadAttention(num_heads = num_heads, key_dim = d_model)(decoder_embedded, decoder_embedded, attention_mask = look_ahead_mask)

        norm_masked_attention = LayerNormalization()(decoder_embedded + masked_attention_output)
        attention_output = MultiHeadAttention(num_heads = num_heads, key_dim = d_model)(norm_masked_attention, encoder_output)
        norm_attention = LayerNormalization()(norm_masked_attention + attention_output)
        dense_output = Dense(d_model, activation = 'relu')(norm_attention)
        decoder_output = LayerNormalization()(norm_attention + dense_output)
        return Model(inputs = [inputs, encoder_output], outputs = decoder_output)

    def call(self, inputs):
        encoder_input, decoder_input = inputs
        encoder_output = self.encoder_block(encoder_input)
        decoder_output = self.decoder_block([decoder_input, encoder_output])

        # Применяем финальный слой к выходу декодера
        final_output = self.final_dense(decoder_output)
        return final_output


max_features = 10000  # Количество слов в словаре
max_len = 100  # Длина последовательности

model_input = tf.keras.Input(shape = (max_len,))
decoder_input = tf.keras.Input(shape = (max_len,))
transformer_output = Transformer(vocab_size = max_features, seq_len = max_len, d_model = 128, num_heads = 4)([model_input, decoder_input])
out = Dense(1, activation='sigmoid')(transformer_output)


final_model = Model(inputs = [model_input, decoder_input], outputs = out)
final_model.compile(optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy', 
                    metrics = ['accuracy'])

# Обучение
final_model.fit([x_train, x_train], y_train,
                 epochs = 5,
                 batch_size = 64,
                 validation_data = ([x_test, x_test], y_test))

# Оценка модели
output = final_model.predict([x_test, x_test])
print(f"Output shape: {output.shape}")  # Проверка выходной формы декодера
