import sys
sys.stdout.reconfigure(encoding = 'utf-8')

import tensorflow as tf
from keras.api.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D, Dropout, MultiHeadAttention
from keras.api.models import Model
from keras.api.datasets import imdb
from keras.api.preprocessing.sequence import pad_sequences

# Загрузка и подготовка данных IMDB
max_features = 10000  # Максимальное количество слов в словаре
max_len = 100  # Максимальная длина отзыва

# Загружаем и подготавливаем данные
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

# Паддинг последовательностей до одинаковой длины
x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)

# Создание модели с использованием слоя Attention
input_layer = tf.keras.Input(shape = (max_len,))
embedding_layer = Embedding(input_dim = max_features, output_dim = 128)(input_layer)

# LSTM слой
lstm_out = LSTM(128, return_sequences = True)(embedding_layer)

# Attention слой
attention_out = MultiHeadAttention(key_dim = 128, num_heads = 8, use_bias = True, dropout = 0.25)(
    query = lstm_out, 
    key = lstm_out, 
    value = lstm_out
)

# Применяем global pooling (по всем временным шагам)
attention_out = GlobalAveragePooling1D()(attention_out)

# Полносвязный слой
dense_out = Dense(64, activation ='relu')(attention_out)
dropout_out = Dropout(0.5)(dense_out)

# Выходной слой для бинарной классификации
output_layer = Dense(1, activation ='sigmoid')(dropout_out)

# Создание модели
model = Model(inputs=input_layer, outputs=output_layer)

# Компиляция модели
model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])

# Обучение модели
model.fit(x_train, y_train, batch_size = 64, epochs = 3, validation_data = (x_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Тестовая точность: {accuracy:.4f}")



