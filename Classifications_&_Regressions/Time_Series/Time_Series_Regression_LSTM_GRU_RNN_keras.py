import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, Dropout, GRU

# дата начала и конца
start_date = dt.datetime(2020, 4, 1)
end_date = dt.datetime(2023, 4, 1)

# загрузка данных с yahoo finance
data = yf.download("GOOGL", start=start_date, end=end_date)

pd.set_option('display.max_rows', 4)
pd.set_option('display.max_columns', 5)
print(data)

# Установка 90 процентов данных для обучения
training_data_len = math.ceil(len(data) * 0.9)

# Разделение набора данных
train_data = data[ : training_data_len]
test_data = data[training_data_len : ]
print(train_data.shape, test_data.shape)

# Выбор значений цены открытия и преобразование 1D в 2D массив
dataset_train = train_data['Open'].values.reshape(-1, 1)  # Используем [] для доступа к столбцу
dataset_test = test_data['Open'].values.reshape(-1, 1)  # Используем [] для доступа к столбцу

# Нормализация набора данных
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_train = scaler.fit_transform(dataset_train)
scaled_test = scaler.transform(dataset_test)

# Подготовка данных для обучения
x_train, y_train = [], []
for i in range(50, len(scaled_train)):
    x_train.append(scaled_train[i-50 : i, 0])
    y_train.append(scaled_train[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Преобразование формы
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
print("x_train :", x_train.shape, "y_train :", y_train.shape)

# Подготовка тестовых данных
x_test, y_test = [], []
for i in range(50, len(scaled_test)):
    x_test.append(scaled_test[i-50 : i, 0])
    y_test.append(scaled_test[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Преобразование формы
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))
print("x_test :", x_test.shape, "y_test :", y_test.shape)

# Инициализация RNN
regressor = Sequential()
regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units = 50, return_sequences = True))
regressor.add(SimpleRNN(units = 50))
regressor.add(Dense(units = 1))  # Без активации для регрессии
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Обучение модели
regressor.fit(x_train, y_train,
              epochs = 20,
              batch_size = 2)
regressor.summary()

# Инициализация модели LSTM
regressorLSTM = Sequential()
regressorLSTM.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressorLSTM.add(LSTM(50, return_sequences=False))
regressorLSTM.add(Dense(25))
regressorLSTM.add(Dense(1))  # Без активации
regressorLSTM.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Обучение модели LSTM
regressorLSTM.fit(x_train, y_train,
                  batch_size = 1,
                  epochs = 12)
regressorLSTM.summary()

# Инициализация модели GRU
regressorGRU = Sequential()
regressorGRU.add(GRU(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressorGRU.add(Dropout(0.2))
regressorGRU.add(GRU(units = 50, return_sequences = True))
regressorGRU.add(GRU(units = 50))
regressorGRU.add(Dense(units = 1))  # Без активации
regressorGRU.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Обучение модели GRU
regressorGRU.fit(x_train, y_train,
                 epochs = 20,
                 batch_size = 1)
regressorGRU.summary()

# Прогнозирование с данными X_test
y_RNN = regressor.predict(X_test)
y_LSTM = regressorLSTM.predict(X_test)
y_GRU = regressorGRU.predict(X_test)

# Преобразование обратно с 0-1 в оригинал
y_RNN_O = scaler.inverse_transform(y_RNN) 
y_LSTM_O = scaler.inverse_transform(y_LSTM) 
y_GRU_O = scaler.inverse_transform(y_GRU)

fig, axs = plt.subplots(3, figsize=(18, 12), sharex = True, sharey = True)
fig.suptitle('Предсказания модели')

# График предсказаний RNN
axs[0].plot(train_data.index[150 : ], train_data.Open[150 : ], label = "train_data", color = "b")
axs[0].plot(test_data.index, test_data.Open, label = "test_data", color = "g")
axs[0].plot(test_data.index[50 : ], y_RNN_O, label = "y_RNN", color = "brown")
axs[0].legend()
axs[0].title.set_text("Базовая RNN")

# График предсказаний LSTM
axs[1].plot(train_data.index[150 : ], train_data.Open[150 : ], label = "train_data", color = "b")
axs[1].plot(test_data.index, test_data.Open, label = "test_data", color = "g")
axs[1].plot(test_data.index[50 : ], y_LSTM_O, label = "y_LSTM", color = "orange")
axs[1].legend()
axs[1].title.set_text("LSTM")

# График предсказаний GRU
axs[2].plot(train_data.index[150 : ], train_data.Open[150 : ], label = "train_data", color = "b")
axs[2].plot(test_data.index, test_data.Open, label = "test_data", color = "g")
axs[2].plot(test_data.index[50 : ], y_GRU_O, label = "y_GRU", color = "red")
axs[2].legend()
axs[2].title.set_text("GRU")

plt.xlabel("Дни")
plt.ylabel("Цена открытия")
plt.show()