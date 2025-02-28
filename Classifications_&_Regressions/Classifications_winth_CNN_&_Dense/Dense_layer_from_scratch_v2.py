import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

class DenseNN(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev = 0.1, name = "w")
            self.b = tf.zeros([self.outputs], dtype = tf.float32, name = "b")

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b, trainable = True)

            self.fl_init = True

        y = x @ self.w + self.b

        if self.activate == "relu":
            return tf.nn.relu(y)
        elif self.activate == "softmax":
            return tf.nn.softmax(y)

        return y


class SequentialModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseNN(128)
        self.layer_2 = DenseNN(10, activate = "softmax")

    def __call__(self, x):
        return self.layer_2(self.layer_1(x))


# Загружаем и подготавливаем данные MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0  # Нормализация
x_test = x_test / 255.0

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])  # Изменение формы данных
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

y_train = to_categorical(y_train, 10)  # Преобразуем метки в категориальный формат

# Инициализация модели
model = SequentialModule()

# Определение функции потерь и оптимизатора
cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.optimizers.Adam(learning_rate = 0.001)

# Параметры обучения
BATCH_SIZE = 32
EPOCHS = 10

# Создание обучающего датасета
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(BATCH_SIZE)

@tf.function
def train_batch(x_batch, y_batch):
    with tf.GradientTape() as tape:
        f_loss = cross_entropy(y_batch, model(x_batch))

    grads = tape.gradient(f_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return f_loss


# Обучение модели
for n in range(EPOCHS):
    loss = 0
    for x_batch, y_batch in train_dataset:
        loss += train_batch(x_batch, y_batch)

    print(f'Epoch {n + 1}, Loss: {loss.numpy()}')

# Оценка модели
y = model(x_test)
y_pred = tf.argmax(y, axis = 1).numpy()

# Вычисление точности
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, y_pred), tf.float32)).numpy() * 100
print(f'Accuracy: {accuracy:.2f}%')
