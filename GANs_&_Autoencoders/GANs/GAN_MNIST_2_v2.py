import sys
import io

# Установим кодировку вывода в UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.models import Model
from keras.datasets import mnist

# Загрузка данных
x, y = mnist.load_data()

# Фильтрация данных для цифры 7
x = x[y == 7]

BUFFER_SIZE = x.shape[0]
BATCH_SIZE = 100

# Обновление размера буфера
BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE
x = x[:BUFFER_SIZE]
#print(x.shape, y.shape)

# Стандартизация входных данных
x = x / 255.0
x = np.reshape(x, (len(x), 28, 28, 1))
train_dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Формирование сетей
hidden_dim = 2

def build_generator():
    generator = tf.keras.Sequential([
        Input(shape = (hidden_dim,)),  
        Dense(7 * 7 * 256, activation = 'relu'),
        BatchNormalization(),
        Reshape((7, 7, 256)),
        Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', activation = 'relu'),
        BatchNormalization(),
        Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = 'same', activation = 'relu'),
        BatchNormalization(),
        Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', activation = 'sigmoid'),
    ])
    return generator

generator = build_generator()

def build_discriminator():
    discriminator = tf.keras.Sequential([
        Input(shape = (28, 28, 1)),  
        Conv2D(64, (5, 5), strides = (2, 2), padding = 'same'),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'),
        LeakyReLU(),
        Dropout(0.3),
        Flatten(),
        Dense(1)
    ])
    return discriminator

discriminator = build_discriminator()

g_opt = Adam(1e-4)
d_opt = Adam(1e-4)

g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

class SimpleGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss
  
    def train_step(self, batch):
        real_images = batch
        fake_images = self.generator(tf.random.normal((BATCH_SIZE, hidden_dim)), training = False)

        # Обучение дискриминатора
        with tf.GradientTape() as d_type:
            yhat_real = self.discriminator(real_images, training = True) 
            yhat_fake = self.discriminator(fake_images, training = True)
            yhat = tf.concat([yhat_real, yhat_fake], axis = 0)
            y = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis = 0)
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y += tf.concat([noise_real, noise_fake], axis=0)
            total_d_loss = self.d_loss(y, yhat)

        d_grad = d_type.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        # Обучение генератора
        with tf.GradientTape() as g_type:
            gen_images = self.generator(tf.random.normal((BATCH_SIZE, hidden_dim)), training = True)
            predicted_labels = self.discriminator(gen_images, training = True)
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        g_grad = g_type.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}

# Создание и компиляция GAN
gan = SimpleGAN(generator, discriminator)
gan.compile(g_opt, d_opt, g_loss, d_loss)

# Обучение GAN
NUM_EPOCHS = 20
try:
    hist = gan.fit(train_dataset, epochs = NUM_EPOCHS)
except UnicodeEncodeError as e:
    print("Ошибка кодировки:", e)

# Генерация изображений
images = generator.predict(tf.random.uniform((BATCH_SIZE, hidden_dim)), verbose = 0)

# Отображение результатов генерации
def show_images(images):
    try:
        fig, ax = plt.subplots(nrows = 4, ncols = 8, figsize = (20, 10))

        idx = 0
        for row in range(4):
            for col in range(8):
                ax[row][col].imshow(images[idx].reshape(28, 28), cmap = 'gray')
                ax[row][col].axis('off')
                idx += 1

        fig.tight_layout()
        plt.show()
    except Exception as e:
        print("Ошибка при отображении изображений:", e)


show_images(images)

# Отображение результатов генерации с изменением координат
n = 2
total = 2 * n + 1

plt.figure(figsize=(total, total))

num = 1
for i in range(-n, n + 1):
    for j in range(-n, n + 1):
        ax = plt.subplot(total, total, num)
        num += 1
        img = generator.predict(np.expand_dims([0.5 * i / n, 0.5 * j / n], axis=0))
        plt.imshow(img[0, :, :, 0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
