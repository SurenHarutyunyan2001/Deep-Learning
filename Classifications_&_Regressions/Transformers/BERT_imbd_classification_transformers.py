import tensorflow as tf
from keras.api.layers import Input, Dense, Dropout
from keras.api.models import Model
from transformers import TFBertForSequenceClassification, BertTokenizer, AdamWeightDecay
from sklearn.model_selection import train_test_split
import numpy as np

# Загрузка датасета IMDb из TensorFlow/Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Загрузка токенизатора BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Функция для токенизации текста
def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="tf")

# Преобразуем индексы слов в текст
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}
decode_review = lambda review: ' '.join([reverse_word_index.get(i - 3, '?') for i in review])

# Преобразуем данные в текст
x_train_text = [decode_review(review) for review in x_train]
x_test_text = [decode_review(review) for review in x_test]

# Токенизируем данные
train_encodings = tokenize(x_train_text)
test_encodings = tokenize(x_test_text)

# Создаем TensorFlow Dataset для обучения
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings), y_train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings), y_test
))

# Устанавливаем параметры обучения
batch_size = 16
train_dataset = train_dataset.shuffle(1000).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Загружаем модель BERT для классификации
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Компиляция модели с оптимизатором AdamWeightDecay
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Обучение модели
model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Оценка модели
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
