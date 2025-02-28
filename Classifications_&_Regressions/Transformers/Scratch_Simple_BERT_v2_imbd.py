import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
import numpy as np
from keras.api.layers import *
from keras.api.models import Model
from keras.api.datasets import imdb
from keras.api.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer



class SimpleBERT(Model):
    def __init__(self, vocab_size, d_model, num_heads, num_blocks, ff_dim, max_len):
        super(SimpleBERT, self).__init__()

        # Эмбеддинги
        self.embedding = Embedding(input_dim = vocab_size, output_dim = d_model)
        
        # Позиционные кодировки
        self.position_encoding = self.positional_encoding(max_len, d_model)
        
        # Многоголовое внимание и блоки трансформера
        self.transformer_blocks = [self.TransformerBlock(d_model, num_heads, ff_dim) for _ in range(num_blocks)]
        
        # Dropout
        self.dropout = Dropout(0.1)
        
        # Финальный слой 
        self.dense =  Dense(vocab_size, activation = 'softmax') 

    def positional_encoding(self, max_len, d_model):
        position = tf.range(max_len, dtype = tf.float32)[:, tf.newaxis]  # (max_len, 1)
        div_term = tf.exp(tf.range(0, d_model, 2, dtype = tf.float32) * -(tf.math.log(10000.0) / d_model))  # (d_model // 2,)

        # Создаем пустой тензор для позиционных кодировок
        encoding = tf.zeros((max_len, d_model), dtype = tf.float32)

        # Четные индексы для синусов и нечетные индексы для косинусов
        sin_vals = tf.sin(position * div_term)  # (max_len, d_model//2)
        cos_vals = tf.cos(position * div_term)  # (max_len, d_model//2)

        # Обновляем тензор для четных индексов
        encoding = tf.concat([sin_vals, cos_vals], axis = -1)  # (max_len, d_model)

        return encoding

    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def TransformerBlock(self, d_model, num_heads, ff_dim):
        # Многоголовое внимание
        inputs = Input(shape = (None, d_model))  # Вход с размерностью (batch_size, seq_len, d_model)
        attention_output = MultiHeadAttention(num_heads = num_heads, key_dim = d_model)(inputs, inputs)
        attention_output = LayerNormalization()(attention_output + inputs)  # Residual connection

        # Feed forward network
        ffn_output = Dense(ff_dim, activation = 'relu')(attention_output)
        ffn_output = Dense(d_model)(ffn_output)
        encoder_output = LayerNormalization()(ffn_output + attention_output)  # Residual connection

        return Model(inputs = inputs, outputs = encoder_output)

    def call(self, inputs):
        # Применяем эмбеддинги и позиционные кодировки
        x = self.embedding(inputs) + self.position_encoding[:tf.shape(inputs)[1], :]

        # Применяем блоки трансформера
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Применяем слой Dropout
        x = self.dropout(x)

        # Усреднение по всем позициям
        x = tf.reduce_mean(x, axis = 1)  # Применяем глобальное усреднение

        # Финальный слой классификации
        output = self.dense(x)

        return output


# Параметры модели
vocab_size = 10000
d_model = 128
num_heads = 8
num_blocks = 4
ff_dim = 512
max_len = 512

# Параметры модели
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size  # Используем размер словаря токенизатора

# Загрузка датасета
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Подготовка данных
# Токенизация данных с помощью BertTokenizer
x_train = [tokenizer.encode(' '.join(map(str, seq)), truncation = True, max_length = max_len, padding = 'max_length', add_special_tokens = True) for seq in x_train]
x_test  = [tokenizer.encode(' '.join(map(str, seq)), truncation = True, max_length = max_len, padding = 'max_length', add_special_tokens = True) for seq in x_test]

# Обрезаем индексы, выходящие за пределы vocab_size
x_train = [[min(token, vocab_size - 1) for token in seq] for seq in x_train]
x_test  = [[min(token, vocab_size - 1) for token in seq] for seq in x_test]

# Преобразуем в тензоры
x_train = tf.convert_to_tensor(x_train)
x_test  = tf.convert_to_tensor(x_test)


# Создание модели
model_input = tf.keras.Input(shape = (max_len,))
simple_bert_model = SimpleBERT(vocab_size = vocab_size, max_len = max_len, num_blocks = 4, d_model = 128, num_heads = 4, ff_dim = 512)
bert_out = simple_bert_model(model_input)  # передаем только один вход

# Выходной слой для бинарной классификации
out = Dense(1, activation = 'sigmoid')(bert_out)
model = Model(inputs = model_input, outputs = out)

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs = 3, batch_size = 32, validation_data = (x_test, y_test))

# Оценка модели на тестовых данных
test_results = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_results[0]}, Test Accuracy: {test_results[1]}")
