import os
# Устанавливаем уровень логирования TensorFlow, чтобы уменьшить количество выводимых сообщений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
# Импортируем библиотеку для работы с набором данных MNIST
from keras.datasets import mnist
from tensorflow import keras
from keras.layers import Dense, Flatten

# Загружаем набор данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация входных данных (приведение значений пикселей к диапазону [0, 1])
x_train = x_train / 255
x_test = x_test / 255

# Преобразуем метки классов в формат one-hot encoding для обучения
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Создаем модель последовательной нейронной сети
model = keras.Sequential([
    # Плоский слой для преобразования 2D изображения в 1D вектор
    Flatten(input_shape = (28, 28, 1)),
    # Полносвязный слой с 128 нейронами и активацией ReLU
    Dense(128, activation = 'relu'),
    # Выходной слой с 10 нейронами (по количеству классов) и активацией softmax
    Dense(10, activation = 'softmax')
])

# Компилируем модель с оптимизатором Adam и функцией потерь categorical_crossentropy
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

# Обучаем модель на тренировочных данных с разделением на валидационную выборку
history = model.fit(x_train, y_train_cat,
                    batch_size = 32,
                    epochs = 5,
                    validation_split = 0.2)

# Извлекаем данные о потерях для графиков
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Строим график потерь на обучающей и валидационной выборках
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Извлекаем данные об аккуратности для графиков
acc = history.history['acc']
val_acc = history.history['val_acc']

# Строим график точности на обучающей и валидационной выборках
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
