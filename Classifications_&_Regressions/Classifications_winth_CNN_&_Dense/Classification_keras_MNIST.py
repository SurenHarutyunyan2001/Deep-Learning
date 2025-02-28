import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.api.datasets import mnist
from tensorflow import keras
from keras.api.layers import Input, Dense, Flatten

# Установка кодировки вывода в UTF-8
sys.stdout.reconfigure(encoding = 'utf-8')

# Понижение уровня логирования TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация входных данных
x_train = x_train / 255.0
x_test = x_test / 255.0

# Преобразование меток в категориальные данные
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Отображение первых 25 изображений из обучающей выборки
plt.figure(figsize = (10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap = plt.cm.binary)
plt.show()

# Создание модели
model = keras.Sequential([
    Input(shape = (28, 28, 1)),  # Задание формы входных данных через Input слой
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

# Вывод структуры модели
print(model.summary())

# Компиляция модели
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Обучение модели
model.fit(x_train, y_train_cat,
          batch_size = 32,
          epochs = 5,
          validation_split = 0.2)

# Оценка модели на тестовой выборке
model.evaluate(x_test, y_test_cat)

# Пример предсказания для одного изображения
n = 1
x = np.expand_dims(x_test[n], axis = 0)
res = model.predict(x)
print(res)
print(np.argmax(res))

plt.imshow(x_test[n], cmap = plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis = 1)

print(pred.shape)
print(pred[: 20])
print(y_test[: 20])

# Выделение неверных вариантов
mask = pred == y_test
print(mask[: 10])

x_false = x_test[~ mask]
y_false = y_test[~ mask]

print(x_false.shape)

# Вывод первых 25 неверных результатов
plt.figure(figsize = (10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)
plt.show()
