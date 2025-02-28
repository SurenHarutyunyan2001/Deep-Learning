import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import matplotlib.pyplot as plt

# Загрузка и предобработка набора данных MNIST
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0  # Векторизуем изображение
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0

# Параметры модели
input_dim = 28 * 28  # Размерность входа (784)
hidden_dim = 64      # Размерность скрытого представления (боттлнек)

# Функция для построения автоэнкодера
def build_autoencoder():
    # Энкодер
    encoder_inputs = Input(shape = (input_dim,), name = 'encoder_input')
    x = Dense(300, activation = 'relu')(encoder_inputs)
    x = Dense(150, activation = 'relu')(x)
    encoded = Dense(hidden_dim, activation = 'linear', name = 'encoded')(x)

    # Декодер
    x = Dense(150, activation = 'relu')(encoded)
    x = Dense(300, activation = 'relu')(x)
    decoded = Dense(input_dim, activation = 'sigmoid')(x)  # Сигмоид для нормализации выхода

    # Модель автоэнкодера
    encoder = Model(encoder_inputs, encoded, name = 'encoder')
    decoder = Model(encoded, decoded, name = 'decofer')
    autoencoder = Model(encoder_inputs, decoded, name = 'autoencoder')
    
    return autoencoder, encoder, decoder

# Создаем и компилируем модель
autoencoder, encoder, decoder = build_autoencoder()
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
autoencoder.summary()

# Обучение автоэнкодера
history = autoencoder.fit(
    x_train, x_train,
    epochs = 50,
    batch_size = 128,
    shuffle = True,
    validation_data = (x_test, x_test)
)

# Функция для визуализации реконструированных изображений
def plot_reconstructed_images(original, reconstructed, n = 10):
    plt.figure(figsize = (20, 4))
    for i in range(n):
        # Оригинальные изображения
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap = 'gray')
        plt.axis('off')

        # Реконструированные изображения
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap = 'gray')
        plt.axis('off')
    plt.show()

# Реконструируем изображения и выводим результат
reconstructed_images = autoencoder.predict(x_test)
plot_reconstructed_images(x_test, reconstructed_images)