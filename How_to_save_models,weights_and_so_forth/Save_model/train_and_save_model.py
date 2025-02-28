import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np
import os

# Установить кодировку UTF-8 для вывода
os.environ["PYTHONIOENCODING"] = "utf-8"

# Создание простого датасета
x = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([0, 1, 2, 3, 4, 5])

# Создание модели
model = Sequential()
model.add(Input(shape = (1,)))  
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer = 'adam', loss = 'mse')

# Обучение модели
#model.fit(X, y, epochs=100, verbose=2)

model.fit(x, y, epochs = 100)

# Сохранение модели
model.save('simple_model.keras')
print("Model saved.")
