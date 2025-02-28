import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as display

# Загружаем изображение контента
content_path = tf.keras.utils.get_file(origin = "https://github.com/selfedu-rus/neural-network/blob/master/img.jpg?raw=true")

# Загружаем изображение стиля
style_path = tf.keras.utils.get_file(origin = "https://github.com/selfedu-rus/neural-network/blob/master/img_style.jpg?raw=true")

def img_scaler(image, max_dim = 256):
  # Преобразует тензор в новый тип.
  original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)
  # Создает коэффициент масштабирования для изображения
  scale_ratio = 4 * max_dim / max(original_shape)
  # Преобразует тензор в новый тип.
  new_shape = tf.cast(original_shape * scale_ratio, tf.int32)
  # Изменяет размер изображения в соответствии с рассчитанным коэффициентом масштабирования
  return tf.image.resize(image, new_shape)

def load_img(path_to_img):
  # Читает и выводит содержимое входного файла.
  img = tf.io.read_file(path_to_img)
  # Определяет, является ли изображение формата BMP, GIF, JPEG или PNG,
  # и выполняет соответствующее действие для преобразования входного
  # байтового потока в тензор типа dtype
  img = tf.image.decode_image(img, channels = 3)
  # Преобразует изображение в dtype, масштабируя (MinMax нормализация) его значения, если это необходимо.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # Масштабирует изображение с помощью нашей пользовательской функции
  img = img_scaler(img)
  # Добавляет четвертое измерение к тензору, так как
  # модель требует тензор 4-мерной формы
  return img[tf.newaxis, :]

# Загружаем изображение контента и стиля
content_image = load_img(content_path)
style_image = load_img(style_path)

# Отображаем загруженные изображения
plt.figure(figsize = (12, 12))
plt.subplot(1, 2, 1)
plt.imshow(content_image[0])
plt.title('Изображение контента')
plt.subplot(1, 2, 2)
plt.imshow(style_image[0])
plt.title('Изображение стиля')

plt.show()

# Создает предварительно обученную модель VGG, которая принимает входные данные и возвращает список промежуточных выходных значений
def vgg_layers(layer_names):
  """ Создает модель VGG, которая возвращает список промежуточных выходных значений. """
  # Загружаем нашу модель. Загружаем предобученную VGG, обученную на данных imagenet
  vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  # Тензорное сокращение по указанным индексам и внешнее произведение.
  # Матрица умножения
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  # Сохраняем форму входного тензора
  input_shape = tf.shape(input_tensor)
  # Преобразует тензор в новый тип.
  num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
  # Делим выход умножения матриц на количество локаций
  return result / (num_locations)

# Мы будем использовать слой block5 conv2 для контента 
content_layers = ['block5_conv2'] 
# Мы будем использовать слои conv1 из каждого блока для стиля 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()

    # Основной 
    self.vgg = vgg_layers(style_layers + content_layers)
    self.vgg.trainable = False

    # Используется как ключи в создании словаря
    self.style_layers = style_layers
    self.content_layers = content_layers
    
  def call(self, inputs):
    # Обрабатывает входное изображение
    "Ожидает входные данные с плавающей точкой в диапазоне [0,1]"
    inputs = inputs * 255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

    # Передаем предварительно обработанное изображение в модель VGG19
    outputs = self.vgg(preprocessed_input)
    # Разделяем выходные данные по стилю и контенту
    style_outputs, content_outputs = (outputs[: len(self.style_layers)], 
                                       outputs[len(self.style_layers):])
    # Обрабатываем выходные данные стиля перед созданием словаря
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    # Создаем два словаря для выходных данных контента и стиля
    content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
    style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
    
    return {'content': content_dict, 'style': style_dict}
  
extractor = StyleContentModel(style_layers, content_layers)
# Установите свои целевые значения для стиля и контента:
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Создаем оптимизатор. В статье рекомендуется LBFGS, но Adam тоже работает неплохо:
opt = tf.optimizers.Adam(learning_rate = 0.005, beta_1 = 0.99, epsilon = 1e-1)

# Чтобы оптимизировать это, используем взвешенное сочетание двух потерь, чтобы получить общую потерю:
style_weight = 1e-2
content_weight = 1e4

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_layers)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_layers)
    loss = style_loss + content_loss
    return loss

total_variation_weight = 500

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight * tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(tf.clip_by_value(image, 
                                  clip_value_min = 0.0, 
                                  clip_value_max = 1.0))
  
image = tf.Variable(content_image)
epochs = 20
steps_per_epoch = 100
step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end = '')
  display.clear_output(wait = True)
  plt.imshow(image[0])
  plt.show()
  print("Train step: {}".format(step))

# Сохраняем итоговое изображение
tf.keras.preprocessing.image.save_img('stylized-image.png', image[0])
