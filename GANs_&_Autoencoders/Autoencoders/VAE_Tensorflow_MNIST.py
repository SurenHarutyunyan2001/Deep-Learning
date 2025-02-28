import tensorflow as tf
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

# Параметры сети
n_inputs = 28 * 28
n_hidden1 = 512     
n_hidden2 = 256     
n_outputs = n_inputs

# Загрузка и подготовка данных
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32").reshape(-1, n_inputs) / 255.0
x_test = x_test.astype("float32").reshape(-1, n_inputs) / 255.0

# Создание placeholder для входных данных
x = tf.compat.v1.placeholder(tf.float32, shape = [None, n_inputs], name = "X")

# Инициализация весов He и L2-регуляризация
learning_rate = 0.001  
l2_reg = 0.0001
he_init = tf.keras.initializers.VarianceScaling()
l2_regularizer = tf.keras.regularizers.L2(l2_reg)

# Частично настроенный слой Dense
my_dense_layer = partial(
    tf.keras.layers.Dense,
    activation = tf.nn.relu,
    kernel_initializer = he_init,
    kernel_regularizer = l2_regularizer,
)

# Создание слоев сети
hidden1 = my_dense_layer(n_hidden1)(x)
hidden1_bn = tf.keras.layers.BatchNormalization()(hidden1)

hidden2_mean = my_dense_layer(n_hidden2)(hidden1_bn)
hidden2_gamma = my_dense_layer(n_hidden2)(hidden1_bn)

noise = tf.random.normal(tf.shape(hidden2_gamma), dtype = tf.float32)
h = hidden2_mean + tf.exp(0.5 * hidden2_gamma) * noise 

hidden3 = my_dense_layer(n_hidden1)(h)
hidden3_bn = tf.keras.layers.BatchNormalization()(hidden3)
logits = my_dense_layer(n_outputs, activation = None)(hidden3_bn)
outputs = tf.sigmoid(logits)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = x, logits = logits)

# Функция потерь
reconstruction_loss = tf.reduce_mean(xentropy) 
latent_loss = 0.5 * tf.reduce_mean(tf.exp(hidden2_gamma) + tf.square(hidden2_mean) - 1 - hidden2_gamma) 
loss = reconstruction_loss + latent_loss

# Оптимизация
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.compat.v1.global_variables_initializer()

# Гиперпараметры обучения
n_epochs = 50  
batch_size = 256  # Увеличьте размер батча
n_digits = 10 
x_test = x_test[:n_digits]

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        n_batches = x_train.shape[0] // batch_size
        for iteration in range(n_batches):
            x_batch = x_train[iteration * batch_size:(iteration + 1) * batch_size]
            sess.run(training_op, feed_dict = {x : x_batch})

        train_loss = sess.run(loss, feed_dict = {x: x_train})
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    reconstructed = sess.run(outputs, feed_dict={x: x_test})

# Визуализация
plt.figure(figsize=(10, 4))
for i in range(n_digits):
    ax = plt.subplot(2, n_digits, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap = "gray")
    plt.axis('off')

    ax = plt.subplot(2, n_digits, i + 1 + n_digits)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap = "gray")
    plt.axis('off')

plt.show()