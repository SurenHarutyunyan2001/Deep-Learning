import tensorflow as tf
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

# Отключение Eager Execution для работы с TensorFlow 1.x
tf.compat.v1.disable_eager_execution()

# Параметры сети
n_inputs = 28 * 28  # Для MNIST
n_hidden1 = 512     # Увеличение числа нейронов
n_hidden2 = 256     # Увеличение числа нейронов
n_hidden3 = n_hidden1
n_outputs = n_inputs

# Загрузим датасет MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32").reshape(-1, n_inputs) / 255.0
x_test = x_test.astype("float32").reshape(-1, n_inputs) / 255.0

learning_rate = 0.001  # Уменьшение скорости обучения
l2_reg = 0.0001

# Создание placeholder для входных данных
x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs], name = "X")

# Добавляем шум в входнх данных
nois_leavel = 1.0
x_nosy = x + nois_leavel * tf.random.normal(tf.shape(x))

# Инициализация весов He и L2-регуляризация
he_init = tf.keras.initializers.VarianceScaling()
l2_regularizer = tf.keras.regularizers.L2(l2_reg)

# Частично настроенный слой Dense
my_dense_layer = partial(
    tf.keras.layers.Dense,
    activation = tf.nn.relu,  # Изменение функции активации на ReLU
    kernel_initializer = he_init,
    kernel_regularizer = l2_regularizer,
)

# Создание слоев сети
hidden1 = my_dense_layer(n_hidden1)(x_nosy)
hidden2 = my_dense_layer(n_hidden2)(hidden1)  # Кодировочный слой
hidden3 = my_dense_layer(n_hidden3)(hidden2)
outputs = my_dense_layer(n_outputs, activation=None)(hidden3)

# Функция потерь: MSE + регуляризационные потери
reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))  # MSE
reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

# Оптимизация
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.compat.v1.global_variables_initializer()

# Гиперпараметры обучения
n_epochs = 10  # Увеличение числа эпох
batch_size = 150

n_test_digits = 10  # Количество тестовых изображений для реконструкции
x_test = x_test[:n_test_digits]

# Запуск сессии
with tf.compat.v1.Session() as sess:
    sess.run(init)  # Инициализация переменных

    for epoch in range(n_epochs):
        n_batches = x_train.shape[0] // batch_size
        for iteration in range(n_batches):
            x_batch = x_train[iteration * batch_size:(iteration + 1) * batch_size]
            sess.run(training_op, feed_dict={x: x_batch})

        # Вычисление функции потерь после каждой эпохи
        train_loss = sess.run(loss, feed_dict={x: x_train})
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    # Реконструируем тестовые изображения
    reconstructed_images = sess.run(outputs, feed_dict={x: x_test})

# Визуализация оригинальных и реконструированных изображений
def plot_image(image, shape = [28, 28]):
    plt.imshow(image.reshape(shape), cmap = "Greys", interpolation = "nearest")
    plt.axis("off")

for digit_index in range(n_test_digits):
    plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
    plot_image(x_test[digit_index])
    plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
    plot_image(reconstructed_images[digit_index])

plt.show()