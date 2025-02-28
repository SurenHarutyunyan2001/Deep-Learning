import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.layers import *
from transformers import BertTokenizer, TFBertModel

# Загрузка токенизатора BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Пример данных: документы и запросы
documents = [
    "I love programming with Python",
    "Keras is a high-level neural networks API",
    "Transformers are powerful for NLP tasks",
    "Deep learning is the future of AI"
]

queries = [
    "I enjoy coding in Python",
    "What is Keras ?",
    "Tell me about Transformers in NLP",
    "AI and Deep Learning advancements"
]

# Метки релевантности: 1 - релевантный, 0 - нерелевантный
labels = [
    [1, 0, 0, 0],  # Для первого запроса релевантен первый документ
    [0, 1, 0, 0],  # Для второго запроса релевантен второй документ
    [0, 0, 1, 0],  # И так далее
    [0, 0, 0, 1]
]

labels = tf.constant(labels, dtype = tf.float32)  # Преобразуем метки в тензор

# Токенизация с использованием BERT токенизатора
def text_to_sequence(text, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = max_len,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'tf'
    )
    return encoding['input_ids'][0], encoding['attention_mask'][0]  # Возвращаем индексы и маску внимания

# Определяем максимальную длину последовательности
max_len = 256  # Можно задать по своему усмотрению

document_sequences = [text_to_sequence(doc, tokenizer, max_len) for doc in documents]
query_sequences = [text_to_sequence(query, tokenizer, max_len) for query in queries]

# Разделяем индексы и маски внимания
document_ids, document_attention_masks = zip(*document_sequences)
query_ids, query_attention_masks = zip(*query_sequences)

# Преобразуем в тип int32
document_ids = np.array(document_ids, dtype = np.int32)
document_attention_masks = np.array(document_attention_masks, dtype = np.int32)
query_ids = np.array(query_ids, dtype = np.int32)
query_attention_masks = np.array(query_attention_masks, dtype = np.int32)

# Загрузка предобученной модели BERT
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Создание модели для векторизации текста
def create_text_model():
    input_text = Input(shape = (max_len,), dtype = tf.int32)
    attention_mask = Input(shape=(max_len,), dtype = tf.int32)  # Маска внимания
    bert_output = bert_model(input_text, attention_mask = attention_mask)[0]
    vectorized_text = GlobalAveragePooling1D()(bert_output)
    return models.Model(inputs = [input_text, attention_mask], outputs = vectorized_text)

# Создаем две модели: одну для документов, другую для запросов
document_model = create_text_model()
query_model = create_text_model()

# Обучаемая функция сходства
query_input = Input(shape = (max_len,), dtype = tf.int32)
doc_input = Input(shape = (max_len,), dtype = tf.int32)
query_attention_mask_input = Input(shape = (max_len,), dtype = tf.int32)
doc_attention_mask_input = Input(shape = (max_len,), dtype = tf.int32)

query_embedding = query_model([query_input, query_attention_mask_input])
doc_embedding = document_model([doc_input, doc_attention_mask_input])

# Косинусное сходство
similarity = layers.Dot(axes=-1, normalize=True)([query_embedding, doc_embedding])
retriever_model = models.Model(inputs = [query_input, query_attention_mask_input, doc_input, doc_attention_mask_input], outputs = similarity)

# Параметры обучения
learning_rate = 1e-5  # Используем меньшую скорость обучения для BERT
batch_size = 4
epochs = 10

# Функция потерь и оптимизатор
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = False)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

# Компиляция модели
retriever_model.compile(optimizer = optimizer, loss = loss_fn, metrics = ["accuracy"])

# Подготовка данных для обучения
queries_repeated = np.repeat(query_ids, len(documents), axis=0)
documents_tiled = np.tile(document_ids, (len(queries), 1))
labels_flattened = labels.numpy().flatten()

# Генерация маски внимания (1 для действительных токенов, 0 для паддинга)
attention_mask_queries = np.where(queries_repeated != 0, 1, 0)
attention_mask_docs = np.where(documents_tiled != 0, 1, 0)

# Обучение модели
retriever_model.fit(
    [queries_repeated, attention_mask_queries, documents_tiled, attention_mask_docs],
    labels_flattened,
    batch_size = batch_size,
    epochs = epochs
)

# Вычисляем сходство для всех запросов и документов
similarities = []

for query_seq, query_mask in zip(query_ids, query_attention_masks):
    query_similarities = []
    for doc_seq, doc_mask in zip(document_ids, document_attention_masks):
        # similarity = retriever_model.predict([query_seq[None, :], query_mask[None, :], doc_seq[None, :], doc_mask[None, :]])
        # модель передает данные по одному примеру за раз
        similarity = retriever_model.predict([query_seq, query_mask, doc_seq, doc_mask])
        # модель передает весь батч данных, и модель может вычислить сходства для всех запросов и документов одновременно
        query_similarities.append(similarity[0][0])  # Получаем значение из предсказания
    similarities.append(query_similarities)

similarities = np.array(similarities)

# Выводим наиболее релевантные документы для каждого запроса
for i, query in enumerate(queries):
    relevant_doc_idx = np.argmax(similarities[i])
    print(f"Query: {query}")
    print(f"Most relevant document: {documents[relevant_doc_idx]}")
    print(f"Similarity score: {similarities[i][relevant_doc_idx]}")
    print("=" * 50)
