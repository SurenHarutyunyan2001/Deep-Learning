import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Установка кодировки для вывода в консоль
sys.stdout.reconfigure(encoding = 'utf-8')

# Гиперпараметры
batch_size = 64       # Размер батча
epochs = 100          # Количество эпох
latent_dim = 256      # Размерность скрытого состояния
num_samples = 10000   # Количество образцов для обучения

# Пути к файлам
data_path = 'spa.txt'  # Путь к файлу с данными в формате "english_sentence \t spanish_sentence \n"

# Загрузка и подготовка данных
input_texts = []
target_texts = []

with open(data_path, 'r', encoding = 'utf-8') as f:
    lines = f.read().split('\n')

# Обработка данных
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    # Добавляем символы начала и конца предложения для целевого текста
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)

# Инициализация токенизаторов для входных и целевых текстов
input_tokenizer = Tokenizer(char_level = True)
target_tokenizer = Tokenizer(char_level = True)

# Обучаем токенизаторы на входных и целевых текстах
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

# Количество уникальных символов в каждом наборе данных
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# Преобразуем текстовые данные в последовательности индексов
encoder_input_data = input_tokenizer.texts_to_sequences(input_texts)
decoder_input_data = target_tokenizer.texts_to_sequences(target_texts)

# Паддинг последовательностей для одинаковой длины
max_encoder_seq_length = max([len(seq) for seq in encoder_input_data])
max_decoder_seq_length = max([len(seq) for seq in decoder_input_data])

encoder_input_data = pad_sequences(encoder_input_data, maxlen = max_encoder_seq_length, padding = 'post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen = max_decoder_seq_length, padding = 'post')

# Подготовка данных для целевой переменной (one-hot encoding сдвига на 1 символ для обучения)
decoder_target_data = np.zeros((len(decoder_input_data), max_decoder_seq_length, num_decoder_tokens), dtype = "float32")
for i, seq in enumerate(decoder_input_data):
    for t, token_index in enumerate(seq[1:]):
        decoder_target_data[i, t, token_index] = 1.

# Построение модели кодировщика с Embedding и Bidirectional LSTM
encoder_inputs = Input(shape = (None,))
encoder_embedding = Embedding(input_dim = num_encoder_tokens, output_dim = latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(latent_dim, return_state = True, dropout = 0.3, recurrent_dropout = 0.3))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)

# Состояния кодировщика - это объединенные состояния обеих направлений
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

# Построение модели декодировщика
decoder_inputs = Input(shape = (None,))
decoder_embedding = Embedding(input_dim = num_decoder_tokens, output_dim = latent_dim, mask_zero = True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim * 2, return_sequences = True, return_state = True, dropout = 0.3, recurrent_dropout = 0.3)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state = encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Итоговая модель для обучения
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Компиляция модели
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Обучение модели
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size = batch_size,
          epochs = epochs,
          validation_split = 0.2)

# Определение модели для кодировщика
encoder_model = Model(encoder_inputs, encoder_states)

# Определение модели для декодировщика
decoder_state_input_h = Input(shape=(latent_dim * 2,))
decoder_state_input_c = Input(shape=(latent_dim * 2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state = decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Функция для предсказания перевода
def decode_sequence(input_seq):
    # Получаем состояния кодировщика для начального состояния декодера
    states_value = encoder_model.predict(input_seq)

    # Создаем начальную целевую последовательность с символом "\t"
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_tokenizer.word_index['\t']] = 1.

    # Инициализация переменных для хранения результата
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Получаем символ с наивысшей вероятностью
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word[sampled_token_index]
        decoded_sentence += sampled_char

        # Условие завершения: достигнут конец последовательности или максимальная длина
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Обновляем целевую последовательность и состояния для следующего шага
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]

    return decoded_sentence

# Тестирование на нескольких примерах
for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input:', input_texts[seq_index])
    print('Decoded:', decoded_sentence)