import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Уровень логирования TensorFlow

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist  # Библиотека для загрузки набора данных MNIST
import keras
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

# Преобразование меток в категориальный формат
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)  # Вывод размеров обучающего набора

# Создание модели
model = keras.Sequential([
    Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)),
    MaxPooling2D((2, 2), strides = 2),
    Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    MaxPooling2D((2, 2), strides = 2),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

print(model.summary())  # Вывод структуры нейросети в консоль

# Компиляция модели
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Обучение модели
his = model.fit(x_train, y_train_cat, 
                batch_size = 32, 
                epochs = 5, 
                validation_split = 0.2)

# Оценка модели на тестовом наборе
model.evaluate(x_test, y_test_cat)

n = 1  # Индекс тестового изображения для предсказания
x = np.expand_dims(x_test[n], axis = 0)  # Добавление размерности для модели
res = model.predict(x)  # Предсказание
print(res)  # Вывод вероятностей
print(np.argmax(res))  # Индекс предсказанного класса

# Отображение тестового изображения
plt.imshow(x_test[n], cmap = plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis = 1)  # Получение предсказанных классов

print(pred.shape)  # Размерность предсказаний

print(pred[: 20])  # Показ первых 20 предсказанных классов
print(y_test[: 20])  # Показ первых 20 истинных классов

# Выделение неверных вариантов
mask = pred == y_test  # Маска для неверных предсказаний
print(mask[: 10])  # Показ первых 10 значений маски

x_false = x_test[~ mask]  # Неверно предсказанные изображения
y_false = y_test[~ mask]  # Истинные метки для неверных предсказаний

print(x_false.shape)  # Размерность неверных изображений

# Вывод первых 25 неверных результатов
plt.figure(figsize = (10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])  # Убираем оси
    plt.yticks([])
    plt.imshow(x_false[i], cmap = plt.cm.binary)

plt.show()
