import numpy as np
import tensorflow as tf
from keras import layers, models

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

labels = tf.constant(labels, dtype=tf.float32)  # Преобразуем метки в тензор

# Создание словаря и преобразование текста в индексы
vocab = set(" ".join(documents + queries).split())

word_to_index = {word: idx + 1 for idx, word in enumerate(vocab)}  # индексация слов
index_to_word = {idx: word for word, idx in word_to_index.items()}

def text_to_sequence(text, word_to_index):
    return [word_to_index[word] for word in text.split() if word in word_to_index]

document_sequences = [text_to_sequence(doc, word_to_index) for doc in documents]
query_sequences = [text_to_sequence(query, word_to_index) for query in queries]

# Padding последовательностей
max_len = max(max(len(seq) for seq in document_sequences), max(len(seq) for seq in query_sequences))
document_sequences = tf.keras.preprocessing.sequence.pad_sequences(document_sequences, maxlen = max_len, padding = 'post')
query_sequences = tf.keras.preprocessing.sequence.pad_sequences(query_sequences, maxlen = max_len, padding = 'post')

# Создание модели для векторизации текста
embedding_dim = 50  # Размерность embedding

# Общий слой Embedding
embedding_layer = layers.Embedding(input_dim = len(vocab) + 1, output_dim = embedding_dim, input_length = max_len)

def create_text_model():
    input_text = layers.Input(shape = (max_len,))
    embedding = embedding_layer(input_text)
    vectorized_text = layers.GlobalAveragePooling1D()(embedding)
    return models.Model(inputs=input_text, outputs = vectorized_text)

# Создаем две модели: одну для документов, другую для запросов
document_model = create_text_model()
query_model = create_text_model()

# Обучаемая функция сходства
query_input = layers.Input(shape = (max_len,))
doc_input = layers.Input(shape = (max_len,))

query_embedding = query_model(query_input)
doc_embedding = document_model(doc_input)

# Косинусное сходство
similarity = layers.Dot(axes = -1, normalize = True)([query_embedding, doc_embedding])
retriever_model = models.Model(inputs = [query_input, doc_input], outputs = similarity)

# Параметры обучения
learning_rate = 0.001
batch_size = 4
epochs = 10

# Функция потерь и оптимизатор
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = False)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

# Компиляция модели
retriever_model.compile(optimizer = optimizer, loss = loss_fn, metrics = ["accuracy"])

# Подготовка данных для обучения
queries_repeated = np.repeat(query_sequences, len(documents), axis = 0)
documents_tiled = np.tile(document_sequences, (len(queries), 1))
labels_flattened = labels.numpy().flatten()

# Обучение модели
retriever_model.fit(
    [queries_repeated, documents_tiled],
    labels_flattened,
    batch_size = batch_size,
    epochs = epochs
)

# Вычисляем сходство для всех запросов и документов
similarities = []

for query_seq in query_sequences:
    query_similarities = []
    for doc_seq in document_sequences:
        similarity = retriever_model.predict([query_seq[None, :], doc_seq[None, :]])
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


    