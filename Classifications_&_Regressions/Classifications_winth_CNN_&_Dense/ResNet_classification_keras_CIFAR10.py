import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Уровень логирования TensorFlow

import tensorflow as tf
import keras
from keras.api.layers import *
from keras.api.models import Model
from keras.api.datasets import cifar10  # Используем CIFAR-10

tf.random.set_seed(1)  # Установка сидов для воспроизводимости

# Загрузка данных
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255  # Нормализация
x_test = x_test / 255
y_train = keras.utils.to_categorical(y_train, 10)  # Преобразование меток
y_test = keras.utils.to_categorical(y_test, 10)

# Создание модели
inputs = Input(shape = (32, 32, 3), name = "img")
x = Conv2D(32, 3, activation = "relu")(inputs)
x = Conv2D(64, 3, activation = "relu")(x)
block_1_output = MaxPooling2D(3)(x)

x = Conv2D(64, 3, activation = "relu", padding = "same")(block_1_output)
x = Conv2D(64, 3, activation = "relu", padding = "same")(x)
block_2_output = keras.layers.add([x, block_1_output])  # Сложение для остаточного соединения

x = Conv2D(64, 3, activation = "relu", padding = "same")(block_2_output)
x = Conv2D(64, 3, activation = "relu", padding = "same")(x)
block_3_output = keras.layers.add([x, block_2_output])  # Сложение для остаточного соединения

x = Conv2D(64, 3, activation = "relu")(block_3_output)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation = 'softmax')(x)

model = Model(inputs, outputs, name = "toy_resnet")  # Создание модели
model.summary()  # Вывод структуры модели

model.compile(optimizer = 'adam',  # Компиляция модели
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, 
          batch_size = 64, 
          epochs = 15, 
          validation_split = 0.2)  # Обучение модели

print(model.evaluate(x_test, y_test))  # Оценка модели
