import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Уровень логирования TensorFlow

import tensorflow as tf
from tensorflow.keras.datasets import mnist  # Импорт набора данных MNIST
from tensorflow.keras.utils import to_categorical  # Импорт функции для преобразования меток в категориальный формат


# Определение класса для плотного слоя
class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super().__init__()
        self.units = units  # Количество выходных единиц
        self.rate = 0.01  # Коэффициент регуляризации

    def build(self, input_shape):
        # Создание весов и смещения
        self.w = self.add_weight(shape = (input_shape[-1], self.units),
                                 initializer = "random_normal",
                                 trainable = True)
        self.b = self.add_weight(shape = (self.units,), initializer = "zeros", trainable = True)

    def call(self, inputs):
        # Регуляризация: добавление потерь на веса
        regular = 100.0 * tf.reduce_mean(tf.square(self.w))
        self.add_loss(regular)

        # Прямое распространение
        return tf.matmul(inputs, self.w) + self.b


# Определение класса нейронной сети
class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseLayer(128)  # Первый плотный слой с 128 выходами
        self.layer_2 = DenseLayer(10)    # Второй плотный слой с 10 выходами (по количеству классов)

    def call(self, inputs):
        x = self.layer_1(inputs)  # Проход через первый слой
        x = tf.nn.relu(x)  # Применение активации ReLU
        x = self.layer_2(x)  # Проход через второй слой
        return tf.nn.softmax(x)  # Применение софтмакс для получения вероятностей


model = NeuralNetwork()  # Создание экземпляра модели

# Компиляция модели
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация входных данных
x_train = x_train / 255.0
x_test = x_test / 255.0

# Изменение формы входных данных для модели
x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

# Преобразование меток в категориальный формат
y_train = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Обучение модели
model.fit(x_train, y_train, 
          batch_size = 32, 
          epochs = 5)

# Оценка модели на тестовом наборе
print(model.evaluate(x_test, y_test_cat))