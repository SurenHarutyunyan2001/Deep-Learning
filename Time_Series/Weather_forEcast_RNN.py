import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.api.layers import *
from keras.api.models import *

# Настройки для графиков
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

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

dataset = features.values  # Преобразование признаков в массив
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)  # Среднее значение для нормализации
data_std = dataset[:TRAIN_SPLIT].std(axis=0)  # Стандартное отклонение для нормализации
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
x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step = True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1], TRAIN_SPLIT, None, past_history, future_target, STEP, single_step = True)

# Создание TensorFlow Dataset для обучающих данных
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Создание TensorFlow Dataset для валидационных данных
val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

# Создание модели с LSTM
single_step_model = Sequential()
single_step_model.add(LSTM(32, input_shape = x_train_single.shape[-2:]))  # Слой LSTM
single_step_model.add(Dense(1))  # Полносвязный слой
single_step_model.compile(optimizer = tf.keras.optimizers.RMSprop(), loss = 'mae')  # Компиляция модели
# Обучение модели
single_step_history = single_step_model.fit(train_data_single,
                                            epochs = EPOCHS,
                                            steps_per_epoch = EVALUATION_INTERVAL,
                                            validation_data = val_data_single,
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
plot_train_history(single_step_history, 'Single Step Training and validation loss')

# Функция для создания временных меток
def create_time_steps(length):
    return list(range(-length, 0))

# Функция для отображения графиков
def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']  # Подписи для графиков
    marker = ['.-', 'rx', 'go']  # Маркеры для графиков
    time_steps = create_time_steps(plot_data[0].shape[0])  # Создание временных меток
    if delta:
        future = delta  # Установка значения будущего
    else:
        future = 0

    plt.title(title)  # Заголовок графика
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize = 10, label = labels[i])  # График предсказаний
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label = labels[i])  # График истории
    plt.legend()  
    plt.xlim([time_steps[0], (future + 5) * 2])  # Установка границ по оси X
    plt.xlabel('Time-Step')  # Подпись по оси X
    return plt  # Возвращаем объект plt для дальнейшего использования

# Визуализация предсказаний на валидационных данных
for x, y in val_data_single.take(3):
    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                      single_step_model.predict(x)[0]], 12,
                     'Single Step Prediction')  # Заголовок для предсказания
    plot.show()   








