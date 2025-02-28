import sys
import io

# Установим кодировку вывода в UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import *
from keras.callbacks import ModelCheckpoint
from keras.models import *

x, y = mnist.load_data()

# Фильтруем данные для цифры 7
x = x[y == 7]

BUFFER_SIZE = x.shape[0]
BATCH_SIZE = 100

# Обновляем размер буфера
BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE
x = x[:BUFFER_SIZE]
#print(x.shape, y.shape)

# Стандартизация входных данных
x = x / 255.0
x = np.reshape(x, (len(x), 28, 28, 1))
train_dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# формирование сетей
hidden_dim = 2


def dropout_and_batch():
  return Dropout(0.3)(BatchNormalization())


# генератор
generator = tf.keras.Sequential([
  Dense(7 * 7 * 256, activation='relu', input_shape = (hidden_dim,)),
  BatchNormalization(),
  Reshape((7, 7, 256)),
  Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', activation = 'relu'),
  BatchNormalization(),
  Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = 'same', activation = 'relu'),
  BatchNormalization(),
  Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', activation = 'sigmoid'),
])

# дискриминатор
discriminator = tf.keras.Sequential()
discriminator.add(Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', input_shape = [28, 28, 1]))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(1))

# потери
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)


def generator_loss(fake_output):
  loss = cross_entropy(tf.ones_like(fake_output), fake_output)
  return loss


def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# обучение
@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, hidden_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training = True)

    real_output = discriminator(images, training = True)
    fake_output = discriminator(generated_images, training = True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  return gen_loss, disc_loss


def train(dataset, epochs):
  history = []
  MAX_PRINT_LABEL = 10
  th = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LABEL)

  for epoch in range(1, epochs + 1):
    print(f'{epoch}/{EPOCHS}: ', end='')

    start = time.time()
    n = 0

    gen_loss_epoch = 0
    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)
      gen_loss_epoch += float(tf.reduce_mean(gen_loss))
      if n % th == 0:
        print('=', end='')
      n += 1

    history += [gen_loss_epoch / n]
    print(': ' + str(history[-1]))
    print('Время эпохи {} составляет {} секунд'.format(epoch, time.time() - start))

  return history


# запуск процесса обучения
EPOCHS = 20
history = train(train_dataset, EPOCHS)

plt.plot(history)
plt.grid(True)
plt.show()

# отображение результатов генерации
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