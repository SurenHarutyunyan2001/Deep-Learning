import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np
import os

# Установить кодировку UTF-8 для вывода
os.environ["PYTHONIOENCODING"] = "utf-8"

# Создание простого датасета
X = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([0, 1, 2, 3, 4, 5])

# Создание модели
model = Sequential()
model.add(Input(shape=(1,)))  
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100)


# После обучения, извлекаем веса из model
weights = model.get_weights()

# Создание второй модели
model2 = Sequential()
model2.add(Input(shape=(1,)))  
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1))

# Устанавливаем веса в model2
model2.set_weights(weights)
