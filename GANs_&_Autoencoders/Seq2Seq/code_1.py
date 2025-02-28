import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Установка кодировки для вывода в консоль
sys.stdout.reconfigure(encoding='utf-8')

# Гиперпараметры
batch_size = 64       # Размер батча
epochs = 40           # Количество эпох
latent_dim = 256      # Размерность скрытого состояния
num_samples = 10000   # Количество образцов для обучения

# Пути к файлам
data_path = 'spa.txt'  # Путь к файлу с данными в формате "english_sentence \t spanish_sentence \n"

# Загрузка и подготовка данных
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# Чтение данных из файла
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

encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_encoder_seq_length, padding = 'post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_decoder_seq_length, padding = 'post')

# Подготовка данных для целевой переменной (one-hot encoding сдвига на 1 символ для обучения)
decoder_target_data = np.zeros((len(decoder_input_data), max_decoder_seq_length, num_decoder_tokens), dtype = "float32")
for i, seq in enumerate(decoder_input_data):
    for t, token_index in enumerate(seq[1:]):  # Сдвиг на 1 для целевых данных
        decoder_target_data[i, t, token_index] = 1.

# Подготовка данных с one-hot encoding для кодировщика
encoder_input_data_one_hot = np.zeros((len(encoder_input_data), max_encoder_seq_length, num_encoder_tokens), dtype = 'float32')
for i, seq in enumerate(encoder_input_data):
    for t, token_index in enumerate(seq):
        encoder_input_data_one_hot[i, t, token_index] = 1.

# Подготовка данных с one-hot encoding для декодировщика
decoder_input_data_one_hot = np.zeros((len(decoder_input_data), max_decoder_seq_length, num_decoder_tokens), dtype = 'float32')
for i, seq in enumerate(decoder_input_data):
    for t, token_index in enumerate(seq):
        decoder_input_data_one_hot[i, t, token_index] = 1.

# Построение модели кодировщика
encoder_inputs = Input(shape = (None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state = True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Построение модели декодировщика
decoder_inputs = Input(shape = (None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Итоговая модель для обучения
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Компиляция модели
model.compile(optimizer =  'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Обучение модели
model.fit([encoder_input_data_one_hot, decoder_input_data_one_hot], decoder_target_data,
          batch_size = batch_size,
          epochs = epochs,
          validation_split = 0.2)

# Определение модели для кодировщика (encoder model)
encoder_model = Model(encoder_inputs, encoder_states)

# Определение модели для декодировщика (decoder model)
decoder_state_input_h = Input(shape = (latent_dim,))
decoder_state_input_c = Input(shape = (latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Функции для предсказания перевода
def decode_sequence(input_seq):
    # Получение состояний кодировщика для начального состояния декодера
    states_value = encoder_model.predict(input_seq)

    # Создание начального целевого последовательности с символом "\t"
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

        # Обновляем целевую последовательность и состояния для следующего предсказания
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence

# Тестирование на нескольких примерах
for seq_index in range(10):
    input_seq = encoder_input_data_one_hot[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print(f"Input: {input_texts[seq_index]}")
    print(f"Decoded: {decoded_sentence}")

print("return 0;")
