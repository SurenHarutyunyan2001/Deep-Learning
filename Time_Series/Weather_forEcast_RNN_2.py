import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Настройки для графиков
mpl.rcParams['figure.figsize'] = (8, 6)  # Размер графиков
mpl.rcParams['axes.grid'] = False  # Отключение сетки на графиках

# Загрузка и распаковка датасета
zip_path = tf.keras.utils.get_file(
    origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname = 'jena_climate_2009_2016.csv.zip',
    extract = True)
csv_path, _ = os.path.splitext(zip_path)  # Убираем расширение файла .zip
df = pd.read_csv(csv_path)  # Чтение CSV файла в DataFrame

# Функция для подготовки одномерных данных
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size  # Смещение на размер истории
    if end_index is None:
        end_index = len(dataset) - target_size  # Устанавливаем конечный индекс

    # Проходим по каждому индексу, собирая данные и метки
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)  # Индексы для истории
        # Преобразование данных в форму (history_size,) -> (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])  # Добавляем целевое значение
    return np.array(data), np.array(labels)  # Возвращаем данные и метки

# Параметры для обучения
TRAIN_SPLIT = 300000  # Разделение на обучающую и тестовую выборки
BATCH_SIZE = 256  # Размер пакета
BUFFER_SIZE = 10000  # Размер буфера для перемешивания
EVALUATION_INTERVAL = 200  # Интервал для оценки модели
EPOCHS = 10  # Количество эпох

tf.random.set_seed(13)  # Установка начального состояния генератора случайных чисел

# Определение используемых признаков
features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = df[features_considered]  # Извлечение признаков из DataFrame
features.index = df['Date Time']  # Установка индекса по времени
features.head()  # Показ первых строк данных

# Подготовка данных
dataset = features.values  # Преобразование признаков в массив
data_mean = dataset[:TRAIN_SPLIT].mean(axis = 0)  # Среднее значение для нормализации
data_std = dataset[:TRAIN_SPLIT].std(axis = 0)  # Стандартное отклонение для нормализации
dataset = (dataset - data_mean) / data_std  # Нормализация данных

# Функция для подготовки многомерных данных
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step = False):
    data = []
    labels = []

    start_index = start_index + history_size  # Смещение на размер истории
    if end_index is None:
        end_index = len(dataset) - target_size  # Устанавливаем конечный индекс

    # Проходим по каждому индексу, собирая данные и метки
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)  # Индексы для истории с шагом
        data.append(dataset[indices])  # Добавляем данные

        # Условие для определения, какую метку добавлять
        if single_step:
            labels.append(target[i + target_size])  # Одношаговое предсказание
        else:
            labels.append(target[i:i + target_size])  # Многопериодное предсказание

    return np.array(data), np.array(labels)  # Возвращаем данные и метки

# Параметры для подготовки данных
past_history = 720  # Размер истории
future_target = 72  # Размер целевого значения
STEP = 6  # Шаг при извлечении данных

# Подготовка обучающих и валидационных данных
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

# Функция для создания временных меток
def create_time_steps(length):
    return list(range(-length, 0))

# Вывод информации о размерах данных
print('Single window of past history : {}'.format(x_train_multi[0].shape))
print('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

# Создание TensorFlow Dataset для обучающих данных
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Создание TensorFlow Dataset для валидационных данных
val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# Функция для визуализации многопериодного предсказания
def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize = (12, 6))  # Размер графика
    num_in = create_time_steps(len(history))  # Временные метки для истории
    num_out = len(true_future)  # Длина истинного будущего

    plt.plot(num_in, np.array(history[:, 1]), label = 'History')  # График истории
    plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'bo',
             label = 'True Future')  # График истинного будущего
    if prediction.any():  # Проверка, есть ли предсказание
        plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'ro',
                 label = 'Predicted Future')  # График предсказанного будущего
    plt.legend(loc = 'upper left')  # Позиция легенды
    plt.show()  # Показ графика

# Визуализация обучающей выборки
for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))  # Визуализация

# Создание модели с LSTM
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences = True, input_shape = x_train_multi.shape[-2:]))  # Первый слой LSTM
multi_step_model.add(tf.keras.layers.LSTM(16, activation = 'relu'))  # Второй слой LSTM
multi_step_model.add(tf.keras.layers.Dense(72))  # Полносвязный слой для предсказания 72 значений
multi_step_model.compile(optimizer = tf.keras.optimizers.RMSprop(clipvalue = 1.0), loss = 'mae')  # Компиляция модели

# Обучение модели
multi_step_history = multi_step_model.fit(train_data_multi, 
                                          epochs = EPOCHS,
                                          steps_per_epoch = EVALUATION_INTERVAL,
                                          validation_data = val_data_multi,
                                          validation_steps = 50)

# Функция для визуализации истории обучения
def plot_train_history(history, title):
    loss = history.history['loss']  # Потери на обучающей выборке
    val_loss = history.history['val_loss']  # Потери на валидационной выборке

    epochs = range(len(loss))  # Эпохи для оси X

    plt.figure()

    plt.plot(epochs, loss, 'b', label = 'Training loss')  # График потерь на обучающей выборке
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')  # График потерь на валидационной выборке
    plt.title(title)  # Заголовок графика
    plt.legend()  # Легенда
    plt.show()  # Показ графика

# Визуализация потерь во время обучения
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

# Визуализация предсказаний на валидационных данных
for x, y in val_data_multi.take(3):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])  # Визуализация предсказаний
