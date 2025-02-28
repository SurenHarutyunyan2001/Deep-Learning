import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.metrics.pairwise import cosine_similarity

# Пример данных: несколько документов и запросов
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

# Шаг 1: Преобразование текста в индексы (для примера используем словарь)
vocab = set(" ".join(documents + queries).split())
word_to_index = {word: idx + 1 for idx, word in enumerate(vocab)}  # индексация слов (начинаем с 1, 0 - зарезервировано)
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Преобразуем документы и запросы в индексы
def text_to_sequence(text, word_to_index):
    return [word_to_index[word] for word in text.split() if word in word_to_index]

document_sequences = [text_to_sequence(doc, word_to_index) for doc in documents]
query_sequences = [text_to_sequence(query, word_to_index) for query in queries]

# Шаг 2: Padding для равной длины последовательностей
max_len = max(max(len(seq) for seq in document_sequences), max(len(seq) for seq in query_sequences))
document_sequences = tf.keras.preprocessing.sequence.pad_sequences(document_sequences, maxlen = max_len, padding = 'post')
query_sequences = tf.keras.preprocessing.sequence.pad_sequences(query_sequences, maxlen = max_len, padding = 'post')

# Шаг 3: Создаем модель
embedding_dim = 50  # Размерность embedding

input_text = layers.Input(shape = (max_len,))
embedding_layer = layers.Embedding(input_dim = len(vocab) + 1, output_dim = embedding_dim)(input_text)
vectorized_text = layers.GlobalAveragePooling1D()(embedding_layer)

# Модель для представления текста
model = models.Model(inputs = input_text, outputs = vectorized_text)

# Шаг 4: Получаем векторные представления для документов и запросов
document_embeddings = model.predict(document_sequences)
query_embeddings = model.predict(query_sequences)

# Шаг 5: Вычисляем сходство (косинусное расстояние) между запросами и документами
similarities = cosine_similarity(query_embeddings, document_embeddings)

# Шаг 6: Для каждого запроса выводим наиболее релевантный документ
for i, query in enumerate(queries):
    relevant_document_idx = np.argmax(similarities[i])  # Индекс наиболее релевантного документа
    print(f"Query: {query}")
    print(f"Most relevant document: {documents[relevant_document_idx]}")
    print(f"Similarity score: {similarities[i][relevant_document_idx]}")
    print("=" *50)
