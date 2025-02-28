import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Загрузка и предобработка набора данных MNIST
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0  # Векторизуем изображение
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0

# Параметры 
input_dim = 28 * 28  # Размерность входа (784)
hidden_dim = 64      # Размерность скрытого представления (боттлнек)
sparsity_level = 0.05
lambda_sparse = 0.1

# Функция для построения автоэнкодера
def build_autoencoder():
    # Энкодер
    inputs = Input(shape = (input_dim,))
    encoded = Dense(hidden_dim, activation = 'relu')(inputs)

    # Декодер
    decoded = Dense(input_dim, activation ='sigmoid')(encoded)# Сигмоид для нормализации выхода

    # Модель автоэнкодера
    autoencoder = Model(inputs, decoded, name = 'autoencoder')
    encoder = Model(inputs, encoded, name = 'encoder')  
    decoder = Model(encoded,decoded, name = 'decofer')
    
    return autoencoder, encoder, decoder



# L = (||x - ^x || ** 2) + "параметр регулиризации" * Penalty(s)
# x входные данные
# ^x реконструктурурованные данные
# Penalty(s) Функция, которая штрафует отклонения от разреженности, часто реализуется с использованием KL-дивергенции.

def sparse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(MeanSquaredError()(y_true, y_pred))
    hidden_layer_output = encoder(y_true)
    mean_activation = tf.reduce_mean(hidden_layer_output, axis=0)

    kl_divergence = tf.reduce_sum(sparsity_level * tf.math.log(sparsity_level / (mean_activation + 1e-10)) +
                                  (1 - sparsity_level) * tf.math.log((1 - sparsity_level) / (1 - mean_activation + 1e-10)))

    return mse_loss + lambda_sparse * kl_divergence


# Создаем и компилируем модель
autoencoder, encoder, decoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss=sparse_loss)

# Обучение автоэнкодера
history = autoencoder.fit(x_train, x_train,
                          epochs = 50,
                          batch_size = 256,
                          shuffle = True)


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