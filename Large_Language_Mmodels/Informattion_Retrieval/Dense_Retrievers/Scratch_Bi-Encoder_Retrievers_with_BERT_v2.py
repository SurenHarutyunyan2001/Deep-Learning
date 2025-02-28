import numpy as np
import tensorflow as tf
from keras import layers, models, backend as K
from keras.layers import *
from transformers import BertTokenizer, TFBertModel

# Загрузка токенизатора BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Загружаем предобученную модель BERT
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Пример данных: документы и запросы
documents = [
    "I love programming with Python",
    "Keras is a high-level neural networks API",
    "Transformers are powerful for NLP tasks",
    "Deep learning is the future of AI"
]

queries = [
    "I enjoy coding in Python",
    "What is Keras?",
    "Tell me about Transformers in NLP",
    "AI and Deep Learning advancements"
]

# Метки релевантности: 1 - релевантный, 0 - нерелевантный
labels = np.array([
    [1, 0, 0, 0],  
    [0, 1, 0, 0],  
    [0, 0, 1, 0],  
    [0, 0, 0, 1]   
], dtype=np.float32)

# Функция токенизации
def text_to_sequence(text, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = max_len,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'tf'
    )
    return encoding['input_ids'][0]  # Возвращаем индексы токенов

# Определяем максимальную длину последовательности
max_len = 256

document_ids = np.array([text_to_sequence(doc, tokenizer, max_len) for doc in documents], dtype = np.int32)
query_ids = np.array([text_to_sequence(query, tokenizer, max_len) for query in queries], dtype = np.int32)

# Создание модели векторизации текста с автоматической маской внимания
def create_text_model():
    input_text = Input(shape = (max_len,), dtype = tf.int32)

    # Генерация маски внутри модели (0 для паддинга, 1 для остальных токенов)
    attention_mask = Lambda(lambda x: K.cast(K.not_equal(x, 0), dtype = 'int32'))(input_text)

    bert_output = bert_model(input_text, attention_mask = attention_mask)[0]
    vectorized_text = GlobalAveragePooling1D()(bert_output)

    return models.Model(inputs = input_text, outputs = vectorized_text)

# Создаем модели для документов и запросов
document_model = create_text_model()
query_model = create_text_model()

# Входные слои для запросов и документов
query_input = Input(shape = (max_len,), dtype = tf.int32)
doc_input = Input(shape = (max_len,), dtype = tf.int32)

# Вычисление эмбеддингов через BERT
query_embedding = query_model(query_input)
doc_embedding = document_model(doc_input)

# Косинусное сходство между запросом и документом
similarity = Dot(axes = -1, normalize = True)([query_embedding, doc_embedding])

# Создание модели
retriever_model = models.Model(inputs=[query_input, doc_input], outputs = similarity)

# Параметры обучения
learning_rate = 1e-5  
batch_size = 4
epochs = 10

# Функция потерь и оптимизатор
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = False)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

# Компиляция модели
retriever_model.compile(optimizer = optimizer, loss = loss_fn, metrics = ["accuracy"])

# Подготовка данных для обучения
queries_repeated = np.repeat(query_ids, len(documents), axis = 0)
documents_tiled = np.tile(document_ids, (len(queries), 1))
labels_flattened = labels.flatten()

# Обучение модели
retriever_model.fit(
    [queries_repeated, documents_tiled],
    labels_flattened,
    batch_size = batch_size,
    epochs = epochs
)

""""
# Вычисляем сходство для всех запросов и документов
similarities = []

for query_seq in query_ids:
    query_similarities = []
    for doc_seq in document_ids:
        # similarity = retriever_model.predict([query_seq[None, :], query_mask[None, :], doc_seq[None, :], doc_mask[None, :]])
        # модель передает данные по одному примеру за раз
        similarity = retriever_model.predict([query_seq, query_mask, doc_seq, doc_mask])
        # модель передает весь батч данных, и модель может вычислить сходства для всех запросов и документов одновременно
        query_similarities.append(similarity[0][0])  # Достаем значение предсказания
    similarities.append(query_similarities)

similarities = np.array(similarities)
"""

# Одновременно предсказываем все сходства
similarities = retriever_model.predict([queries_repeated, documents_tiled])

# Преобразуем результат обратно в матрицу (queries x documents)
similarities = similarities.reshape(len(queries), len(documents))

# Выводим наиболее релевантные документы для каждого запроса
for i, query in enumerate(queries):
    relevant_doc_idx = np.argmax(similarities[i])
    print(f"Query: {query}")
    print(f"Most relevant document: {documents[relevant_doc_idx]}")
    print(f"Similarity score: {similarities[i][relevant_doc_idx]}")
    print("=" * 50)
